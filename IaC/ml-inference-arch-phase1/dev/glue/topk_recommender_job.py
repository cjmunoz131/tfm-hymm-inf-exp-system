"""
AWS Glue PySpark Job: TopK Recommender (Retrieval + Reranking)
===============================================================
Genera las top-K recomendaciones para cada usuario del sistema usando
el patrón Two-Stage: Retrieval (kNN) + Reranking (Full Model).

Flujo:
  1. Lee configuración (top_k, endpoints) desde SSM Parameter Store
  2. Lee usuarios únicos desde Gold (feature_interactions.parquet)
  3. Para cada usuario:
     a. Invoca User Tower endpoint → user_embedding (128D)
     b. kNN search en OpenSearch Serverless → top-retrieval_k candidatos
     c. Invoca Full Model endpoint para cada candidato → hybrid_score
     d. Ordena por hybrid_score y se queda con top-K
  4. Consolida resultados como tabla Iceberg en Gold

Schema tabla de salida (hymmrec_topk_recommendations):
  user_idx, user_id (movie_id movieId original), rank, item_idx, movie_id,
  hybrid_score, prob_interaction, pred_rating_stars,
  attention_weights, title, genres, release_year,
  generated_at

Argumentos Glue:
  --JOB_NAME, --gold_interactions_path, --config_parameter,
  --opensearch_host, --opensearch_index_name,
  --target_gold_database, --target_topk_table,
  --pipeline_id, --correlation_id, --aws_region
"""

import sys
import json
import logging
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, FloatType, StringType, ArrayType
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
    logger.info(f"  Config: top_k={config['top_k']}, retrieval_k={config['retrieval_k']}")
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
def invoke_user_tower(user_idx: int, endpoint_name: str) -> list:
    """Invoca User Tower endpoint → user_embedding (128D)."""
    payload = json.dumps({"user_idx": user_idx})
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload
    )
    result = json.loads(response['Body'].read().decode())
    # User tower retorna {"user_embeddings": [[128D]]}
    return result['user_embeddings'][0]


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
# RETRIEVAL (kNN en OpenSearch)
# ============================================================
def retrieve_candidates(os_client, user_embedding: list, retrieval_k: int) -> list:
    """
    kNN search en OpenSearch para obtener top-retrieval_k candidatos.
    Solo retorna item_idx + retrieval_score. Las features se cruzan
    después con hymmrec_items_consolidated (tabla Iceberg en Gold).
    """
    result = os_client.search(
        index=OPENSEARCH_INDEX_NAME,
        body={
            "size": retrieval_k,
            "query": {
                "knn": {
                    "item_embedding": {
                        "vector": user_embedding,
                        "k": retrieval_k
                    }
                }
            },
            "_source": ["item_idx", "movie_id"]
        }
    )

    candidates = []
    for hit in result['hits']['hits']:
        src = hit['_source']
        candidates.append({
            'item_idx': src['item_idx'],
            'movie_id': src.get('movie_id'),
            'retrieval_score': hit['_score']
        })

    return candidates


# ============================================================
# RERANKING (Full Model)
# ============================================================
def rerank_candidates(user_idx: int, candidates: list, items_lookup: dict,
                      full_model_endpoint: str, batch_size: int = 20) -> list:
    """
    Invoca el Full Model para cada candidato usando features de items_consolidated.
    Las features (genres_multihot, text_emb, img_emb) vienen del lookup de la tabla Iceberg.
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
            # Agregar metadata de contenido para explicabilidad
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

    # Paralelizar invocaciones al endpoint
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(score_candidate, c): c for c in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                scored_candidates.append(result)

    # Ordenar por hybrid_score descendente
    scored_candidates.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
    return scored_candidates


# ============================================================
# PIPELINE POR USUARIO
# ============================================================
def process_user(user_idx: int, config: dict, os_client, items_lookup: dict, now_iso: str) -> list:
    """
    Pipeline completo para un usuario:
      1. User Tower → embedding
      2. kNN retrieval → candidatos (item_idx + retrieval_score)
      3. Full Model reranking (features de items_consolidated) → top-K
    """
    top_k = config['top_k']
    retrieval_k = config['retrieval_k']
    user_tower_endpoint = config['user_tower_endpoint']
    full_model_endpoint = config['full_model_endpoint']
    batch_size_reranking = config.get('batch_size_reranking', 20)

    try:
        # 1. Obtener user embedding
        user_embedding = invoke_user_tower(user_idx, user_tower_endpoint)

        # 2. Retrieval kNN
        candidates = retrieve_candidates(os_client, user_embedding, retrieval_k)

        if not candidates:
            return []

        # 3. Reranking con Full Model (features de items_consolidated)
        scored = rerank_candidates(user_idx, candidates, items_lookup,
                                   full_model_endpoint, batch_size_reranking)

        # 4. Top-K
        topk = scored[:top_k]

        # 5. Construir registros de salida (incluye metadata para explicabilidad)
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
    logger.info("\n[PASO 1/5] Cargando configuración...")
    config = load_config()

    # 2. LEER USUARIOS ÚNICOS
    logger.info("\n[PASO 2/6] Leyendo usuarios únicos desde Gold...")
    df_gold = spark.read.parquet(GOLD_INTERACTIONS_PATH)
    user_idxs = [row['userId_idx'] for row in
                 df_gold.select('userId_idx').distinct().orderBy('userId_idx').collect()]
    logger.info(f"  → Usuarios únicos: {len(user_idxs):,}")

    # 3. LEER ITEMS CONSOLIDATED (features para reranking + metadata)
    logger.info("\n[PASO 3/6] Leyendo hymmrec_items_consolidated desde Gold...")
    full_table = f"glue_catalog.{SOURCE_GOLD_DATABASE}.{ITEMS_CONSOLIDATED_TABLE}"
    df_items = spark.sql(f"SELECT * FROM {full_table}")
    logger.info(f"  → Ítems consolidados: {df_items.count():,}")

    # Construir lookup dict por item_idx para acceso O(1) durante reranking
    items_lookup = {}
    for row in df_items.collect():
        item_idx = int(row['item_idx'])
        items_lookup[item_idx] = row.asDict(recursive=True)

    # 4. CREAR OPENSEARCH CLIENT
    logger.info("\n[PASO 4/6] Inicializando OpenSearch client...")
    os_client = create_opensearch_client()

    # 5. PROCESAR USUARIOS (Retrieval + Reranking)
    logger.info("\n[PASO 5/6] Generando recomendaciones top-K...")
    now_iso = datetime.now(timezone.utc).isoformat()
    all_records = []
    batch_size_users = config.get('batch_size_users', 50)

    for i, user_idx in enumerate(user_idxs):
        records = process_user(user_idx, config, os_client, items_lookup, now_iso)
        all_records.extend(records)

        # Log progreso cada N usuarios
        if (i + 1) % batch_size_users == 0:
            logger.info(f"  Progreso: {i + 1}/{len(user_idxs)} usuarios | "
                        f"Recomendaciones acumuladas: {len(all_records):,}")

    logger.info(f"  → Total recomendaciones generadas: {len(all_records):,}")

    if not all_records:
        raise RuntimeError("No se generaron recomendaciones. Verificar endpoints y OpenSearch.")

    # 6. PERSISTIR EN GOLD (Recommendations DB)
    logger.info("\n[PASO 6/6] Escribiendo tabla top-K en Gold (recommendations)...")
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
    logger.info(f"  Top-K: {config['top_k']} | Retrieval-K: {config['retrieval_k']}")
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
