"""
AWS Glue PySpark Job: TopK Recommender (Item-Based Retrieval + Reranking)
==========================================================================
Genera las top-K recomendaciones para cada usuario usando el patrón:
  Item-Based Retrieval (kNN por seeds diversas) + Reranking (Full Model).

Estrategia de retrieval:
  En vez de usar el user_embedding (que no correlaciona bien con el Full Model),
  se seleccionan "seed items" del historial del usuario (las películas mejor
  calificadas, diversificadas por género) y se buscan ítems similares a cada seed
  en OpenSearch. Esto produce candidatos con alta similitud de contenido multimodal
  al perfil real del usuario.

Selección de seeds (diversidad por género):
  1. Tomar las interacciones del usuario con rating >= umbral (default 4.0)
  2. Agrupar por género principal
  3. De cada grupo de género, tomar la mejor calificada
  4. Limitar a max_seeds (default 10) seeds diversas

Flujo por usuario:
  1. Seleccionar seed items diversificados por género
  2. Para cada seed → kNN en OpenSearch → top-N similares
  3. Unión + deduplicación de todos los candidatos
  4. Filtrar ítems ya vistos
  5. Full Model reranking → pred_rating_stars
  6. Top-K final

Argumentos Glue:
  --JOB_NAME, --gold_interactions_path, --config_parameter,
  --opensearch_host, --opensearch_index_name,
  --source_gold_database, --items_consolidated_table,
  --target_recommendations_database, --target_topk_table,
  --pipeline_id, --correlation_id, --aws_region
"""

import sys
import json
import logging
import time
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, StringType
)
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

# ============================================================
# ARGUMENTOS
# ============================================================
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'gold_interactions_path',
    'config_parameter',
    'opensearch_host',
    'opensearch_index_name',
    'source_gold_database',
    'items_consolidated_table',
    'target_recommendations_database',
    'target_topk_table',
    'pipeline_id',
    'correlation_id',
    'aws_region',
])

# ============================================================
# INICIALIZACIÓN SPARK + GLUE
# ============================================================
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'))
logger.addHandler(handler)

# ============================================================
# CONSTANTES
# ============================================================
REGION = args['aws_region']
PIPELINE_ID = args['pipeline_id']
CORRELATION_ID = args['correlation_id']
GOLD_INTERACTIONS_PATH = args['gold_interactions_path']
CONFIG_PARAMETER = args['config_parameter']
OPENSEARCH_HOST = args['opensearch_host']
OPENSEARCH_INDEX_NAME = args['opensearch_index_name']
SOURCE_GOLD_DATABASE = args['source_gold_database']
ITEMS_CONSOLIDATED_TABLE = args['items_consolidated_table']
RECOMMENDATIONS_DATABASE = args['target_recommendations_database']
TOPK_TABLE = args['target_topk_table']

# Clientes AWS
ssm_client = boto3.client('ssm', region_name=REGION)
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=REGION)


# ============================================================
# CONFIGURACIÓN DESDE PARAMETER STORE
# ============================================================
def load_config() -> dict:
    """Lee la configuración del job desde SSM Parameter Store."""
    logger.info(f"Cargando config desde: {CONFIG_PARAMETER}")
    response = ssm_client.get_parameter(Name=CONFIG_PARAMETER, WithDecryption=True)
    config = json.loads(response['Parameter']['Value'])
    logger.info(f"  Config: top_k={config['top_k']}, retrieval_per_seed={config.get('retrieval_per_seed', 50)}, "
                f"max_seeds={config.get('max_seeds', 10)}, seed_rating_threshold={config.get('seed_rating_threshold', 4.0)}")
    return config


# ============================================================
# OPENSEARCH CLIENT
# ============================================================
def create_opensearch_client():
    """Crea cliente OpenSearch con SigV4 auth para AOSS."""
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    credentials = boto3.Session(region_name=REGION).get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key, credentials.secret_key,
        REGION, "aoss", session_token=credentials.token
    )
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )


# ============================================================
# SAGEMAKER ENDPOINT INVOCATIONS
# ============================================================
def invoke_full_model(user_idx: int, item_idx: int, genres_multihot: list,
                      text_emb: list, img_emb: list, endpoint_name: str) -> dict:
    """Invoca Full Model endpoint → scores + attention."""
    payload = json.dumps({
        "user_idx": user_idx,
        "item_idx": item_idx,
        "genres_multihot": genres_multihot,
        "text_emb": text_emb,
        "img_emb": img_emb
    })
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload
    )
    return json.loads(response['Body'].read().decode())


# ============================================================
# SEED SELECTION (diversidad por género)
# ============================================================
def select_diverse_seeds(user_interactions: list, items_lookup: dict, config: dict) -> list:
    """
    Selecciona seed items del historial del usuario con diversidad por género.
    
    Estrategia:
      1. Filtrar interacciones con rating >= threshold
      2. Obtener género principal de cada ítem (desde items_consolidated)
      3. Agrupar por género → tomar la mejor calificada de cada grupo
      4. Ordenar por rating y limitar a max_seeds
    
    Esto asegura que las seeds representen los diferentes gustos del usuario,
    no solo su género más frecuente.
    """
    seed_rating_threshold = config.get('seed_rating_threshold', 4.0)
    max_seeds = config.get('max_seeds', 10)

    # Filtrar interacciones positivas
    positive_interactions = [
        inter for inter in user_interactions
        if inter['rating'] >= seed_rating_threshold
    ]

    if not positive_interactions:
        # Fallback: tomar las mejor calificadas sin umbral
        positive_interactions = sorted(user_interactions, key=lambda x: x['rating'], reverse=True)[:max_seeds]

    # Agrupar por género principal para diversificar
    genre_groups = defaultdict(list)
    for inter in positive_interactions:
        item_idx = inter['item_idx']
        item_data = items_lookup.get(item_idx)
        if item_data is None:
            continue

        # Obtener género principal (primer género de la lista)
        genres_text = item_data.get('genres_text', '')
        if isinstance(genres_text, list):
            primary_genre = genres_text[0] if genres_text else 'Unknown'
        elif isinstance(genres_text, str) and genres_text:
            # Puede venir como "['Action', 'Drama']" o "Action, Drama"
            try:
                parsed = json.loads(genres_text.replace("'", '"'))
                primary_genre = parsed[0] if parsed else 'Unknown'
            except (json.JSONDecodeError, IndexError):
                primary_genre = genres_text.split(',')[0].strip().strip("[]'\"")
        else:
            primary_genre = 'Unknown'

        genre_groups[primary_genre].append({
            'item_idx': item_idx,
            'rating': inter['rating'],
            'genre': primary_genre
        })

    # De cada grupo de género, tomar la mejor calificada (round-robin por género)
    seeds = []
    # Ordenar cada grupo por rating descendente
    for genre in genre_groups:
        genre_groups[genre].sort(key=lambda x: x['rating'], reverse=True)

    # Round-robin: tomar 1 de cada género hasta completar max_seeds
    genre_keys = sorted(genre_groups.keys(),
                        key=lambda g: genre_groups[g][0]['rating'], reverse=True)
    idx_per_genre = {g: 0 for g in genre_keys}

    while len(seeds) < max_seeds:
        added_this_round = False
        for genre in genre_keys:
            if len(seeds) >= max_seeds:
                break
            group = genre_groups[genre]
            idx = idx_per_genre[genre]
            if idx < len(group):
                seeds.append(group[idx])
                idx_per_genre[genre] = idx + 1
                added_this_round = True
        if not added_this_round:
            break

    logger.debug(f"  Seeds seleccionados: {len(seeds)} de {len(genre_keys)} géneros distintos")
    return seeds


# ============================================================
# ITEM-BASED RETRIEVAL (kNN por cada seed)
# ============================================================
def retrieve_candidates_item_based(os_client, seeds: list, items_lookup: dict,
                                   retrieval_per_seed: int, seen_items: set) -> list:
    """
    Para cada seed item, busca los top-N ítems más similares en OpenSearch.
    Deduplica y filtra ya vistos.
    
    Usa el item_embedding del seed como query vector (item-to-item similarity).
    """
    candidate_scores = {}  # item_idx → max retrieval_score (dedup por mejor score)

    for seed in seeds:
        seed_idx = seed['item_idx']
        seed_data = items_lookup.get(seed_idx)
        if seed_data is None or seed_data.get('item_embedding') is None:
            continue

        item_embedding = seed_data['item_embedding']
        if not isinstance(item_embedding, list):
            continue

        try:
            result = os_client.search(
                index=OPENSEARCH_INDEX_NAME,
                body={
                    "size": retrieval_per_seed,
                    "query": {
                        "knn": {
                            "item_embedding": {
                                "vector": item_embedding,
                                "k": retrieval_per_seed
                            }
                        }
                    },
                    "_source": ["item_idx", "movie_id"]
                }
            )

            for hit in result['hits']['hits']:
                src = hit['_source']
                item_idx = src['item_idx']

                # Filtrar: no el propio seed, no ya vistos
                if item_idx == seed_idx or item_idx in seen_items:
                    continue

                # Guardar el mejor retrieval_score si hay duplicados
                score = hit['_score']
                if item_idx not in candidate_scores or score > candidate_scores[item_idx]['retrieval_score']:
                    candidate_scores[item_idx] = {
                        'item_idx': item_idx,
                        'movie_id': src.get('movie_id'),
                        'retrieval_score': score,
                        'source_seed_idx': seed_idx,
                        'source_seed_genre': seed.get('genre', '')
                    }

        except Exception as e:
            logger.warning(f"  Error en kNN para seed {seed_idx}: {e}")

    candidates = list(candidate_scores.values())
    return candidates


# ============================================================
# RERANKING (Full Model)
# ============================================================
def rerank_candidates(user_idx: int, candidates: list, items_lookup: dict,
                      full_model_endpoint: str, batch_size: int = 20) -> list:
    """
    Invoca el Full Model para cada candidato.
    Ordena por pred_rating_stars (más discriminante para ítems no vistos).
    """
    scored_candidates = []

    def score_candidate(candidate):
        try:
            item_idx = candidate['item_idx']
            item_features = items_lookup.get(item_idx)
            if item_features is None:
                return None

            result = invoke_full_model(
                user_idx=user_idx,
                item_idx=item_idx,
                genres_multihot=item_features['genres_multihot'],
                text_emb=item_features['text_emb'],
                img_emb=item_features['img_emb'],
                endpoint_name=full_model_endpoint
            )
            candidate['hybrid_score'] = result.get('hybrid_score', 0.0)
            candidate['prob_interaction'] = result.get('prob_interaction', 0.0)
            candidate['pred_rating_stars'] = result.get('pred_rating_stars', 0.0)
            candidate['attention_weights'] = result.get('attention_weights', {})
            # Metadata de contenido para explicabilidad
            candidate['title'] = item_features.get('title', '')
            candidate['genres'] = item_features.get('genres_text', '')
            candidate['synopsis'] = item_features.get('synopsis', '')
            candidate['release_year'] = item_features.get('release_year', 0)
            candidate['poster_path'] = item_features.get('poster_path', '')
            candidate['director'] = item_features.get('director', '')
            return candidate
        except Exception as e:
            logger.warning(f"  Error scoring item_idx={candidate.get('item_idx')}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(score_candidate, c): c for c in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                scored_candidates.append(result)

    # Ordenar por hybrid_score (combina probabilidad de interacción + rating predicho)
    scored_candidates.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
    return scored_candidates


# ============================================================
# PIPELINE POR USUARIO
# ============================================================
def process_user(user_idx: int, user_interactions: list, config: dict,
                 os_client, items_lookup: dict, seen_items: set, now_iso: str) -> list:
    """
    Pipeline completo para un usuario:
      1. Seleccionar seeds diversas por género
      2. Item-based kNN retrieval por cada seed
      3. Full Model reranking
      4. Top-K final
    """
    top_k = config['top_k']
    retrieval_per_seed = config.get('retrieval_per_seed', 50)
    full_model_endpoint = config['full_model_endpoint']
    batch_size_reranking = config.get('batch_size_reranking', 20)

    try:
        # 1. Seleccionar seeds diversas
        seeds = select_diverse_seeds(user_interactions, items_lookup, config)
        if not seeds:
            return []

        # 2. Item-based retrieval
        candidates = retrieve_candidates_item_based(
            os_client, seeds, items_lookup, retrieval_per_seed, seen_items
        )

        if not candidates:
            return []

        # 3. Reranking con Full Model
        scored = rerank_candidates(user_idx, candidates, items_lookup,
                                   full_model_endpoint, batch_size_reranking)

        # 4. Top-K
        topk = scored[:top_k]

        # 5. Construir registros de salida
        records = []
        for rank, item in enumerate(topk, start=1):
            records.append({
                'user_idx': user_idx,
                'rank': rank,
                'item_idx': item['item_idx'],
                'movie_id': item['movie_id'],
                'hybrid_score': float(item.get('hybrid_score', 0.0)),
                'prob_interaction': float(item.get('prob_interaction', 0.0)),
                'pred_rating_stars': float(item.get('pred_rating_stars', 0.0)),
                'retrieval_score': float(item.get('retrieval_score', 0.0)),
                'attention_weights': json.dumps(item.get('attention_weights', {})),
                'title': item.get('title', ''),
                'genres': item.get('genres', ''),
                'synopsis': item.get('synopsis', ''),
                'release_year': int(item.get('release_year', 0)),
                'poster_path': item.get('poster_path', ''),
                'director': item.get('director', ''),
                'generated_at': now_iso
            })
        return records

    except Exception as e:
        logger.warning(f"  Error procesando user_idx={user_idx}: {e}")
        return []


# ============================================================
# PERSISTENCIA EN GOLD (Iceberg)
# ============================================================
def write_topk_to_gold(df_topk):
    """Escribe la tabla de top-K recomendaciones como Iceberg en Gold (recommendations DB)."""
    full_table = f"glue_catalog.{RECOMMENDATIONS_DATABASE}.{TOPK_TABLE}"
    logger.info(f"Escribiendo tabla Iceberg: {full_table}")

    df_topk.createOrReplaceTempView("tmp_topk")

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {full_table}
        USING iceberg
        AS SELECT * FROM tmp_topk WHERE 1=0
    """)

    spark.sql(f"""
        INSERT OVERWRITE {full_table}
        SELECT * FROM tmp_topk
    """)

    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table}").collect()[0]['cnt']
    logger.info(f"  → Tabla {full_table}: {count:,} registros")


# ============================================================
# MAIN
# ============================================================
def main():
    inicio = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Job: {args['JOB_NAME']} | Pipeline: {PIPELINE_ID}")
    logger.info(f"{'='*60}")

    # 1. CONFIGURACIÓN
    logger.info("\n[PASO 1/6] Cargando configuración...")
    config = load_config()

    # 2. LEER INTERACCIONES + CONSTRUIR HISTORIAL POR USUARIO
    logger.info("\n[PASO 2/6] Leyendo interacciones y construyendo perfiles...")
    df_gold = spark.read.parquet(GOLD_INTERACTIONS_PATH)

    # Construir lookup completo: user_idx → [{item_idx, rating}]
    user_interactions_map = defaultdict(list)
    user_seen_items = defaultdict(set)

    interactions_rows = df_gold.select('userId_idx', 'movieId_idx', 'rating').collect()
    for row in interactions_rows:
        uid = int(row['userId_idx'])
        mid = int(row['movieId_idx'])
        rating = float(row['rating'])
        user_interactions_map[uid].append({'item_idx': mid, 'rating': rating})
        user_seen_items[uid].add(mid)

    user_idxs = sorted(user_interactions_map.keys())
    logger.info(f"  → Usuarios: {len(user_idxs):,} | Interacciones totales: {len(interactions_rows):,}")

    # 3. LEER ITEMS CONSOLIDATED
    logger.info("\n[PASO 3/6] Leyendo hymmrec_items_consolidated...")
    full_table = f"glue_catalog.{SOURCE_GOLD_DATABASE}.{ITEMS_CONSOLIDATED_TABLE}"
    df_items = spark.sql(f"SELECT * FROM {full_table}")
    logger.info(f"  → Ítems consolidados: {df_items.count():,}")

    items_lookup = {}
    for row in df_items.collect():
        item_idx = int(row['item_idx'])
        items_lookup[item_idx] = row.asDict(recursive=True)

    # 4. CREAR OPENSEARCH CLIENT
    logger.info("\n[PASO 4/6] Inicializando OpenSearch client...")
    os_client = create_opensearch_client()

    # 5. PROCESAR USUARIOS (Item-Based Retrieval + Reranking)
    logger.info("\n[PASO 5/6] Generando recomendaciones top-K (item-based retrieval)...")
    now_iso = datetime.now(timezone.utc).isoformat()
    all_records = []
    batch_log_size = config.get('batch_size_users', 50)

    for i, user_idx in enumerate(user_idxs):
        user_interactions = user_interactions_map[user_idx]
        seen_items = user_seen_items[user_idx]

        records = process_user(
            user_idx, user_interactions, config,
            os_client, items_lookup, seen_items, now_iso
        )
        all_records.extend(records)

        if (i + 1) % batch_log_size == 0:
            logger.info(f"  Progreso: {i + 1}/{len(user_idxs)} usuarios | "
                        f"Recomendaciones: {len(all_records):,}")

    logger.info(f"  → Total recomendaciones: {len(all_records):,}")

    if not all_records:
        raise RuntimeError("No se generaron recomendaciones.")

    # 6. PERSISTIR EN GOLD
    logger.info("\n[PASO 6/6] Escribiendo tabla top-K en Gold...")
    schema = StructType([
        StructField("user_idx", IntegerType()),
        StructField("rank", IntegerType()),
        StructField("item_idx", IntegerType()),
        StructField("movie_id", IntegerType()),
        StructField("hybrid_score", FloatType()),
        StructField("prob_interaction", FloatType()),
        StructField("pred_rating_stars", FloatType()),
        StructField("retrieval_score", FloatType()),
        StructField("attention_weights", StringType()),
        StructField("title", StringType()),
        StructField("genres", StringType()),
        StructField("synopsis", StringType()),
        StructField("release_year", IntegerType()),
        StructField("poster_path", StringType()),
        StructField("director", StringType()),
        StructField("generated_at", StringType()),
    ])
    df_topk = spark.createDataFrame(all_records, schema=schema)
    write_topk_to_gold(df_topk)

    # RESUMEN
    dur = round(time.time() - inicio, 2)
    n_users = df_topk.select('user_idx').distinct().count()
    logger.info(f"\n{'='*60}")
    logger.info(f"Completado en {dur}s")
    logger.info(f"  Usuarios: {n_users:,} | Recomendaciones: {len(all_records):,}")
    logger.info(f"  Top-K: {config['top_k']} | Seeds/user: {config.get('max_seeds', 10)} | "
                f"Retrieval/seed: {config.get('retrieval_per_seed', 50)}")
    logger.info(f"{'='*60}")


# ============================================================
# EJECUCIÓN
# ============================================================
try:
    main()
except Exception as e:
    logger.error(f"Job FALLÓ: {e}")
    raise
finally:
    job.commit()
