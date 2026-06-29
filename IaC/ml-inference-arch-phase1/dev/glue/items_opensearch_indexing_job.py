"""
AWS Glue PySpark Job: Consolidate Items + Index to OpenSearch Serverless (AOSS)
================================================================================
Consolida los inputs del Item Tower con sus embeddings generados, enriquece con
metadata de Silver, persiste como tabla Iceberg en Gold, e indexa en AOSS.

Flujo:
  1. Lee BT input original (JSONL: item_idx, movie_id, genres_multihot, text_emb, img_emb)
  2. Lee BT output (JSONL: item_embedding 64D + attention_weights)
  3. JOIN por monotonic_id (BT mantiene orden línea a línea, SingleRecord)
  4. Lee metadata de Silver (Iceberg: cleansed_movies via Glue Catalog)
  5. JOIN consolidado + metadata → DataFrame final
  6. Escribe tabla Iceberg hymmrec_items_consolidated en Gold (Glue Catalog)
  7. Indexa en OpenSearch Serverless (AOSS) para kNN retrieval

Documento OpenSearch final (todo lo necesario para retrieval + reranking):
  {
    "item_idx", "movie_id",
    "genres_multihot", "text_emb", "img_emb",           # inputs item tower (reranking)
    "item_embedding", "attention_weights",               # outputs item tower (kNN)
    "title", "genres", "synopsis", "release_year",       # metadata (UI/explicabilidad)
    "poster_path", "director", "indexed_at"
  }

Argumentos Glue:
  --JOB_NAME, --batch_transform_input_path, --batch_transform_output_path,
  --source_silver_database, --source_silver_movies_table,
  --target_gold_database, --target_consolidated_table,
  --opensearch_host, --opensearch_index_name,
  --pipeline_id, --correlation_id, --aws_region
"""

import sys
import json
import logging
import time
from datetime import datetime, timezone

import boto3
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

# ============================================================
# ARGUMENTOS
# ============================================================
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'batch_transform_input_path',
    'batch_transform_output_path',
    'source_silver_database',
    'source_silver_movies_table',
    'target_gold_database',
    'target_consolidated_table',
    'opensearch_host',
    'opensearch_index_name',
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
BT_INPUT_PATH = args['batch_transform_input_path']
BT_OUTPUT_PATH = args['batch_transform_output_path']
SILVER_DATABASE = args['source_silver_database']
SILVER_MOVIES_TABLE = args['source_silver_movies_table']
GOLD_DATABASE = args['target_gold_database']
CONSOLIDATED_TABLE = args['target_consolidated_table']
OPENSEARCH_HOST = args['opensearch_host']
OPENSEARCH_INDEX_NAME = args['opensearch_index_name']

EMBEDDING_DIM = 128  # Verificar con model_metadata.json del modelo ganador
BULK_BATCH_SIZE = 500


# ============================================================
# LECTURA DE DATOS
# ============================================================
def read_batch_transform_input():
    """Lee el JSONL input original del Batch Transform (con features: genres_multihot, text_emb, img_emb)."""
    logger.info(f"Leyendo BT input desde: {BT_INPUT_PATH}")
    df = spark.read.json(BT_INPUT_PATH)
    logger.info(f"  → Registros: {df.count():,} | Columnas: {df.columns}")
    return df


def read_batch_transform_output():
    """Lee el JSONL output del Batch Transform (item_embedding + attention_weights)."""
    logger.info(f"Leyendo BT output desde: {BT_OUTPUT_PATH}")
    df = spark.read.json(BT_OUTPUT_PATH)
    logger.info(f"  → Registros: {df.count():,} | Columnas: {df.columns}")
    return df


def read_silver_movies():
    """Lee cleansed_movies desde Glue Catalog (tabla Iceberg en Silver)."""
    logger.info(f"Leyendo Silver: {SILVER_DATABASE}.{SILVER_MOVIES_TABLE}")
    full_table = f"glue_catalog.{SILVER_DATABASE}.{SILVER_MOVIES_TABLE}"
    df = spark.sql(f"SELECT * FROM {full_table}")
    logger.info(f"  → Registros: {df.count():,}")
    return df


# ============================================================
# CONSOLIDACIÓN
# ============================================================
def consolidate_items(df_input, df_output, df_silver):
    """
    JOIN por item_idx entre BT input (features) y BT output (embeddings),
    luego enriquece con Silver metadata por movie_id.
    """
    logger.info("Consolidando BT input + output + Silver metadata...")

    # JOIN input + output por item_idx (key determinística del passthrough)
    df_bt = df_output.join(
        df_input.select("item_idx", "genres_multihot", "text_emb", "img_emb"),
        on="item_idx",
        how="inner"
    )
    logger.info(f"  → Tras JOIN input+output por item_idx: {df_bt.count():,}")

    # Preparar Silver para JOIN por movieId
    silver_cols = df_silver.columns
    movie_id_col = "movieId" if "movieId" in silver_cols else "movie_id"

    df_meta = df_silver.select(
        F.col(movie_id_col).cast("int").alias("_silver_movie_id"),
        F.col("titulo").alias("title") if "titulo" in silver_cols else F.col("title").alias("title"),
        F.col("generos").alias("genres_text") if "generos" in silver_cols else F.col("genres").alias("genres_text"),
        F.col("sinopsis").alias("synopsis") if "sinopsis" in silver_cols else F.col("synopsis").alias("synopsis"),
        F.col("fecha_lanzamiento").alias("release_date") if "fecha_lanzamiento" in silver_cols else F.col("release_date").alias("release_date"),
        F.col("poster_path"),
        F.col("director") if "director" in silver_cols else F.lit(""),
    ).dropDuplicates(["_silver_movie_id"])

    # JOIN con Silver por movie_id
    df_consolidated = df_bt.join(
        df_meta,
        df_bt["movie_id"].cast("int") == df_meta["_silver_movie_id"],
        how="left"
    ).drop("_silver_movie_id")

    # Extraer año
    df_consolidated = df_consolidated.withColumn(
        "release_year",
        F.when(F.col("release_date").isNotNull(),
               F.substring(F.col("release_date").cast("string"), 1, 4).cast("int")
        ).otherwise(0)
    ).drop("release_date")

    # Agregar timestamp de indexación
    now_iso = datetime.now(timezone.utc).isoformat()
    df_consolidated = df_consolidated.withColumn("indexed_at", F.lit(now_iso))

    logger.info(f"  → Consolidado final: {df_consolidated.count():,} documentos")
    return df_consolidated


# ============================================================
# PERSISTENCIA EN GOLD (Iceberg)
# ============================================================
def write_consolidated_to_gold(df_consolidated):
    """Escribe la tabla consolidada como Iceberg en Gold (Glue Catalog)."""
    full_table = f"glue_catalog.{GOLD_DATABASE}.{CONSOLIDATED_TABLE}"
    logger.info(f"Escribiendo tabla Iceberg: {full_table}")

    # Registrar como vista temporal
    df_consolidated.createOrReplaceTempView("tmp_consolidated")

    # CREATE TABLE IF NOT EXISTS + INSERT OVERWRITE (idempotente)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {full_table}
        USING iceberg
        AS SELECT * FROM tmp_consolidated WHERE 1=0
    """)

    spark.sql(f"""
        INSERT OVERWRITE {full_table}
        SELECT * FROM tmp_consolidated
    """)

    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table}").collect()[0]['cnt']
    logger.info(f"  → Tabla {full_table}: {count:,} registros escritos")


# ============================================================
# OPENSEARCH SERVERLESS (opensearch-py + SigV4)
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


def ensure_index_exists(client):
    """Crea el índice kNN en AOSS si no existe."""
    logger.info(f"Verificando índice: {OPENSEARCH_INDEX_NAME}")
    if client.indices.exists(index=OPENSEARCH_INDEX_NAME):
        logger.info("  → Índice existe, upsert por _id")
        return

    logger.info("  → Creando índice kNN...")
    index_body = {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
        "mappings": {
            "properties": {
                "item_idx": {"type": "integer"},
                "movie_id": {"type": "integer"},
                "genres_multihot": {"type": "float"},
                "text_emb": {"type": "float"},
                "img_emb": {"type": "float"},
                "item_embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {"name": "hnsw", "space_type": "cosinesimil",
                               "parameters": {"ef_construction": 512, "m": 16}}
                },
                "attention_weights": {"type": "object", "properties": {
                    "category": {"type": "float"}, "text": {"type": "float"}, "image": {"type": "float"}
                }},
                "title": {"type": "text"}, "genres_text": {"type": "keyword"},
                "synopsis": {"type": "text"}, "release_year": {"type": "integer"},
                "poster_path": {"type": "keyword"}, "director": {"type": "keyword"},
                "indexed_at": {"type": "date"}
            }
        }
    }
    result = client.indices.create(index=OPENSEARCH_INDEX_NAME, body=index_body)
    logger.info(f"  → Creado: {result.get('acknowledged')}")


def bulk_index_documents(client, documents: list) -> tuple:
    """Bulk index a AOSS con upsert por _id = item_idx."""
    from opensearchpy.helpers import bulk

    logger.info(f"Indexando {len(documents):,} documentos...")
    # Log estructura del primer documento para diagnóstico
    if documents:
        sample = documents[0]
        logger.info(f"  Ejemplo doc keys: {list(sample.keys())}")
        logger.info(f"  item_embedding type: {type(sample.get('item_embedding'))}, len: {len(sample.get('item_embedding', []))}")
        logger.info(f"  attention_weights type: {type(sample.get('attention_weights'))}")
    actions = [{"_index": OPENSEARCH_INDEX_NAME, "_id": str(d['item_idx']), "_source": d} for d in documents]

    total_ok, total_err = 0, 0
    for i in range(0, len(actions), BULK_BATCH_SIZE):
        batch = actions[i:i + BULK_BATCH_SIZE]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ok, errors = bulk(client, batch, raise_on_error=False)
                total_ok += ok
                if isinstance(errors, list) and errors:
                    if attempt < max_retries - 1:
                        # Reintentar tras esperar (throttling de AOSS)
                        import time
                        time.sleep(5 * (attempt + 1))
                        continue
                    total_err += len(errors)
                    if i == 0:
                        logger.error(f"  Primer error del bulk: {json.dumps(errors[0], default=str)[:500]}")
                elif isinstance(errors, int):
                    total_err += errors
                break  # Si no hay errores, salir del retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5 * (attempt + 1))
                    continue
                logger.error(f"  Batch {i // BULK_BATCH_SIZE + 1} exception: {e}")
                total_err += len(batch)
                break
        # Pequeña pausa entre batches para no saturar AOSS
        import time
        time.sleep(1)

    logger.info(f"  → OK: {total_ok:,} | Errores: {total_err:,}")
    return total_ok, total_err


# ============================================================
# MAIN
# ============================================================
def main():
    inicio = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Job: {args['JOB_NAME']} | Pipeline: {PIPELINE_ID}")
    logger.info(f"{'='*60}")

    # 1. LEER BT INPUT + OUTPUT
    logger.info("\n[PASO 1/5] Leyendo Batch Transform input y output...")
    df_input = read_batch_transform_input()
    df_output = read_batch_transform_output()

    # 2. LEER SILVER
    logger.info("\n[PASO 2/5] Leyendo Silver metadata...")
    df_silver = read_silver_movies()

    # 3. CONSOLIDAR
    logger.info("\n[PASO 3/5] Consolidando datos...")
    df_consolidated = consolidate_items(df_input, df_output, df_silver)

    # 4. PERSISTIR EN GOLD (Iceberg)
    logger.info("\n[PASO 4/5] Escribiendo tabla Iceberg en Gold...")
    write_consolidated_to_gold(df_consolidated)

    # 5. INDEXAR EN OPENSEARCH
    logger.info("\n[PASO 5/5] Indexando en OpenSearch Serverless...")
    documents = [row.asDict(recursive=True) for row in df_consolidated.collect()]
    os_client = create_opensearch_client()
    ensure_index_exists(os_client)
    total_ok, total_err = bulk_index_documents(os_client, documents)

    # RESUMEN
    dur = round(time.time() - inicio, 2)
    logger.info(f"\n{'='*60}")
    logger.info(f"Completado en {dur}s | Indexados: {total_ok:,} | Errores: {total_err:,}")
    logger.info(f"{'='*60}")

    if total_err > 0 and total_ok == 0:
        raise RuntimeError(f"Indexación falló: {total_err} errores")


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
