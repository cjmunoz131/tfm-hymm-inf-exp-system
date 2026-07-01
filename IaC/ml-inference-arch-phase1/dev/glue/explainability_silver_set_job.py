"""
AWS Glue PySpark Job: Explainability Silver Set Generation (GPT-4)
===================================================================
Genera el silver dataset de explicabilidad invocando GPT-4o para producir
3 keywords temáticas que justifican cada recomendación del sistema híbrido.

Este dataset se usa para hacer fine-tuning (QLoRA) de Llama 3.1 8B como
modelo de explicabilidad local que reemplace GPT-4 en producción.

Flujo:
  1. Lee configuración desde SSM Parameter Store
  2. Lee API key de OpenAI desde Secrets Manager
  3. Lee recomendaciones top-K (tabla Iceberg: hymmrec_topk_recommendations)
  4. Lee interacciones de Gold (feature_interactions.parquet)
  5. Lee cleansed_movies de Silver (Iceberg: título, sinopsis, géneros, palabras_clave)
  6. Construye perfil de usuario (top géneros, top keywords, historial top-5)
  7. Filtra recomendaciones elegibles (pred_rating_stars >= umbral, rank <= top_k_explain)
  8. Para cada recomendación elegible → invoca GPT-4o → 3 keywords de explicación
  9. Escribe tabla Iceberg hymmrec_explainability_silver_set en Gold
  10. Exporta JSONL (instruction/input/output) para SFT directo

Argumentos Glue:
  --JOB_NAME, --config_parameter, --secret_name,
  --source_recommendations_database, --source_topk_table,
  --source_silver_database, --source_silver_movies_table,
  --gold_interactions_path,
  --target_recommendations_database, --target_explainability_table,
  --output_jsonl_path,
  --pipeline_id, --correlation_id, --aws_region
"""

import sys
import json
import logging
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict

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
    'config_parameter',
    'secret_name',
    'source_recommendations_database',
    'source_topk_table',
    'source_silver_database',
    'source_silver_movies_table',
    'gold_interactions_path',
    'target_recommendations_database',
    'target_explainability_table',
    'output_jsonl_path',
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
CONFIG_PARAMETER = args['config_parameter']
SECRET_NAME = args['secret_name']
RECOMMENDATIONS_DB = args['source_recommendations_database']
TOPK_TABLE = args['source_topk_table']
SILVER_DATABASE = args['source_silver_database']
SILVER_MOVIES_TABLE = args['source_silver_movies_table']
GOLD_INTERACTIONS_PATH = args['gold_interactions_path']
TARGET_DB = args['target_recommendations_database']
TARGET_TABLE = args['target_explainability_table']
OUTPUT_JSONL_PATH = args['output_jsonl_path']

# Clientes AWS
ssm_client = boto3.client('ssm', region_name=REGION)
secrets_client = boto3.client('secretsmanager', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)

# ============================================================
# PROMPT ENGINEERING
# ============================================================
SYSTEM_PROMPT = (
    "You are an expert AI recommendation explainer. Your task is to extract "
    "EXACTLY 3 short, concrete keywords from the RECOMMENDED MOVIE'S SYNOPSIS "
    "and GENRES that justify why it matches the USER'S PROFILE.\n\n"
    "Constraints:\n"
    "1. Grounding: The keywords MUST be explicitly present or directly derived "
    "from the movie's synopsis and genres.\n"
    "2. Personalization: The keywords must clearly bridge the recommended movie "
    "to the user's historical preferences.\n"
    "3. Safe Generalization: Extract core plot elements, cinematic tones, or "
    "subgenres. DO NOT use highly specific proper nouns, character names, or "
    "sensitive real-world historical terms .\n"
    "4. Formatting: Output STRICTLY a comma-separated list of 3 short keywords. "
    "No introductory text or conversational filler."
)

USER_PROMPT_TEMPLATE = (
    "### USER PROFILE & VIEWING HISTORY:\n"
    "- Favorite Genres: {top_genres}\n"
    "- Global Keyword Preferences: {top_keywords}\n\n"
    "Historical Films Watched, Highly Rated & Loved (Title | Genres):\n"
    "{history_list}\n\n"
    "### RECOMMENDED MOVIE TO EXPLAIN:\n"
    "- Title: {movie_title}\n"
    "- Genres: {movie_genres}\n"
    "- Synopsis: {movie_synopsis}\n\n"
    "### 3 THEMATIC KEYWORDS EXPLANATION:"
)


# ============================================================
# CONFIGURACIÓN Y SECRETOS
# ============================================================
def load_config() -> dict:
    """Lee configuración del job desde SSM Parameter Store."""
    logger.info(f"Cargando config: {CONFIG_PARAMETER}")
    response = ssm_client.get_parameter(Name=CONFIG_PARAMETER, WithDecryption=True)
    config = json.loads(response['Parameter']['Value'])
    logger.info(f"  Config: min_rating={config.get('min_pred_rating_stars')}, "
                f"top_k_explain={config.get('top_k_explain')}, "
                f"max_samples={config.get('max_samples')}")
    return config


def get_openai_api_key() -> str:
    """Obtiene la API key de OpenAI desde Secrets Manager."""
    logger.info(f"Obteniendo secret: {SECRET_NAME}")
    response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
    secret = json.loads(response['SecretString'])
    return secret.get('api_key', secret.get('OPENAI_API_KEY', secret.get('password', '')))


# ============================================================
# GPT-4 INVOCATION
# ============================================================
def create_openai_client(api_key: str):
    """Inicializa el cliente OpenAI."""
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def invoke_gpt4(client, prompt: str, max_retries: int = 3) -> str:
    """Invoca GPT-4o con reintentos exponenciales."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=30,
                top_p=1.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"  GPT-4 error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return "ERROR_API_TIMEOUT"


# ============================================================
# LECTURA DE DATOS (Iceberg via Spark SQL)
# ============================================================
def read_topk_recommendations():
    """Lee tabla de recomendaciones top-K desde Glue Catalog (Iceberg)."""
    full_table = f"glue_catalog.{RECOMMENDATIONS_DB}.{TOPK_TABLE}"
    logger.info(f"Leyendo: {full_table}")
    df = spark.sql(f"SELECT * FROM {full_table}")
    logger.info(f"  → Registros: {df.count():,}")
    return df


def read_silver_movies():
    """Lee cleansed_movies desde Glue Catalog (Iceberg)."""
    full_table = f"glue_catalog.{SILVER_DATABASE}.{SILVER_MOVIES_TABLE}"
    logger.info(f"Leyendo: {full_table}")
    df = spark.sql(f"SELECT * FROM {full_table}")
    logger.info(f"  → Registros: {df.count():,}")
    return df


def read_gold_interactions():
    """Lee interacciones desde Gold parquet."""
    logger.info(f"Leyendo: {GOLD_INTERACTIONS_PATH}")
    df = spark.read.parquet(GOLD_INTERACTIONS_PATH)
    logger.info(f"  → Registros: {df.count():,}")
    return df


# ============================================================
# PERFIL DE USUARIO
# ============================================================
def build_user_profiles(df_interactions, df_movies, umbral_rating: float = 4.0) -> dict:
    """
    Construye perfiles de usuario con diversidad de géneros y keywords.
    Lee desde las tablas Spark y retorna dict en el driver.
    """
    logger.info("Construyendo perfiles de usuario...")

    # Collect a pandas para procesamiento en driver (610 usuarios → manejable)
    pd_interactions = df_interactions.select(
        'userId_idx', 'movieId', 'rating', 'timestamp'
    ).toPandas()

    pd_movies = df_movies.select(
        'movieId', 'titulo', 'generos', 'palabras_clave'
    ).toPandas()

    pd_merged = pd_interactions.merge(pd_movies, on='movieId', how='inner')

    profiles = {}
    for user_idx, group in pd_merged.groupby('userId_idx'):
        positives = group[group['rating'] >= umbral_rating].sort_values('timestamp')
        if positives.empty:
            continue

        # Top 3 géneros
        all_genres = []
        for gen in positives['generos']:
            if isinstance(gen, str):
                all_genres.extend([g.strip() for g in gen.split(',') if g.strip()])
        top_genres = ", ".join([g[0] for g in Counter(all_genres).most_common(3)])

        # Top 5 keywords (de palabras_clave)
        all_kw = []
        for kw in positives['palabras_clave']:
            if isinstance(kw, str) and kw.lower() not in ('nan', '', 'none'):
                all_kw.extend([k.strip() for k in kw.split(',') if k.strip()])
        top_kw = ", ".join([k[0] for k in Counter(all_kw).most_common(5)])
        if not top_kw:
            top_kw = "general entertainment"

        # Últimas 5 bien calificadas
        last5 = positives.tail(5)
        history = ""
        for _, row in last5.iterrows():
            history += f"- {row['titulo']} | Genres: {row['generos']}\n"

        profiles[int(user_idx)] = {
            'top_genres': top_genres,
            'top_keywords': top_kw,
            'history_list': history.strip()
        }

    logger.info(f"  → Perfiles: {len(profiles):,}")
    return profiles


# ============================================================
# GENERACIÓN DEL SILVER SET
# ============================================================
def generate_silver_set(df_topk, df_movies, user_profiles: dict,
                        openai_client, config: dict) -> list:
    """
    Para cada recomendación elegible invoca GPT-4o y genera las 3 keywords.
    """
    min_rating = config.get('min_pred_rating_stars', 3.5)
    top_k_explain = config.get('top_k_explain', 5)
    max_samples = config.get('max_samples', 2000)

    # Filtrar y collect
    df_eligible = df_topk.filter(
        (F.col('pred_rating_stars') >= min_rating) & (F.col('rank') <= top_k_explain)
    )
    eligible_rows = df_eligible.collect()
    logger.info(f"  Recomendaciones elegibles: {len(eligible_rows):,}")

    # Lookup de películas
    movies_rows = df_movies.select(
        'movieId', 'titulo', 'sinopsis', 'generos'
    ).collect()
    movies_lookup = {}
    for row in movies_rows:
        movies_lookup[int(row['movieId'])] = {
            'titulo': str(row['titulo'] or ''),
            'sinopsis': str(row['sinopsis'] or '')[:800],
            'generos': str(row['generos'] or ''),
        }

    dataset = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for row in eligible_rows:
        if len(dataset) >= max_samples:
            break

        user_idx = int(row['user_idx'])
        movie_id = int(row['movie_id'])

        profile = user_profiles.get(user_idx)
        if not profile:
            continue

        movie = movies_lookup.get(movie_id)
        if not movie or not movie['sinopsis']:
            continue

        # Construir prompt
        prompt = USER_PROMPT_TEMPLATE.format(
            top_genres=profile['top_genres'],
            top_keywords=profile['top_keywords'],
            history_list=profile['history_list'],
            movie_title=movie['titulo'],
            movie_genres=movie['generos'],
            movie_synopsis=movie['sinopsis']
        )

        # Invocar GPT-4
        keywords = invoke_gpt4(openai_client, prompt)
        if keywords == "ERROR_API_TIMEOUT":
            continue

        dataset.append({
            'user_idx': user_idx,
            'movie_id': movie_id,
            'title': movie['titulo'],
            'genres': movie['generos'],
            'synopsis': movie['sinopsis'],
            'pred_rating_stars': float(row['pred_rating_stars']),
            'hybrid_score': float(row['hybrid_score'] or 0.0),
            'user_top_genres': profile['top_genres'],
            'user_top_keywords': profile['top_keywords'],
            'user_history': profile['history_list'],
            'explanation_keywords': keywords,
            'instruction': SYSTEM_PROMPT,
            'input_prompt': prompt,
            'generated_at': now_iso
        })

        if len(dataset) % 100 == 0:
            logger.info(f"  Progreso: {len(dataset):,}/{max_samples}")

    logger.info(f"  → Silver set: {len(dataset):,} muestras")
    return dataset


# ============================================================
# EXPORTACIÓN JSONL (para SFT)
# ============================================================
def export_jsonl(dataset: list, s3_path: str):
    """Exporta en formato instruction/input/output JSONL para fine-tuning."""
    logger.info(f"Exportando JSONL: {s3_path}")
    lines = []
    for r in dataset:
        lines.append(json.dumps({
            "instruction": r['instruction'],
            "input": r['input_prompt'],
            "output": r['explanation_keywords'],
            "metadata": {
                "userId_idx": r['user_idx'],
                "movieId": r['movie_id'],
                "pred_rating_stars": r['pred_rating_stars']
            }
        }, ensure_ascii=False))

    body = '\n'.join(lines)
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3_client.put_object(Bucket=bucket, Key=key, Body=body.encode('utf-8'),
                         ContentType='application/jsonlines')
    logger.info(f"  → {len(lines):,} registros ({len(body)/1024:.1f} KB)")


# ============================================================
# PERSISTENCIA EN GOLD (Iceberg)
# ============================================================
def write_to_gold(dataset: list):
    """Escribe el silver set como tabla Iceberg en Gold."""
    full_table = f"glue_catalog.{TARGET_DB}.{TARGET_TABLE}"
    logger.info(f"Escribiendo: {full_table}")

    schema = StructType([
        StructField("user_idx", IntegerType()),
        StructField("movie_id", IntegerType()),
        StructField("title", StringType()),
        StructField("genres", StringType()),
        StructField("synopsis", StringType()),
        StructField("pred_rating_stars", FloatType()),
        StructField("hybrid_score", FloatType()),
        StructField("user_top_genres", StringType()),
        StructField("user_top_keywords", StringType()),
        StructField("user_history", StringType()),
        StructField("explanation_keywords", StringType()),
        StructField("instruction", StringType()),
        StructField("input_prompt", StringType()),
        StructField("generated_at", StringType()),
    ])

    df = spark.createDataFrame(dataset, schema=schema)
    df.createOrReplaceTempView("tmp_silver")

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {full_table}
        USING iceberg
        AS SELECT * FROM tmp_silver WHERE 1=0
    """)
    spark.sql(f"INSERT OVERWRITE {full_table} SELECT * FROM tmp_silver")

    count = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table}").collect()[0]['cnt']
    logger.info(f"  → {count:,} registros escritos")


# ============================================================
# MAIN
# ============================================================
def main():
    inicio = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Job: {args['JOB_NAME']} | Pipeline: {PIPELINE_ID}")
    logger.info(f"{'='*60}")

    # 1. CONFIG
    logger.info("\n[PASO 1/7] Configuración...")
    config = load_config()

    # 2. SECRET
    logger.info("\n[PASO 2/7] Obteniendo API key...")
    api_key = get_openai_api_key()
    openai_client = create_openai_client(api_key)

    # 3. DATOS
    logger.info("\n[PASO 3/7] Leyendo datos...")
    df_topk = read_topk_recommendations()
    df_movies = read_silver_movies()
    df_interactions = read_gold_interactions()

    # 4. PERFILES
    logger.info("\n[PASO 4/7] Construyendo perfiles...")
    profiles = build_user_profiles(
        df_interactions, df_movies,
        umbral_rating=config.get('profile_rating_threshold', 4.0)
    )

    # 5. GENERACIÓN
    logger.info("\n[PASO 5/7] Generando silver set con GPT-4o...")
    dataset = generate_silver_set(df_topk, df_movies, profiles, openai_client, config)
    if not dataset:
        raise RuntimeError("No se generaron muestras.")

    # 6. JSONL
    logger.info("\n[PASO 6/7] Exportando JSONL...")
    export_jsonl(dataset, OUTPUT_JSONL_PATH)

    # 7. ICEBERG
    logger.info("\n[PASO 7/7] Persistiendo en Gold...")
    write_to_gold(dataset)

    # RESUMEN
    dur = round(time.time() - inicio, 2)
    logger.info(f"\n{'='*60}")
    logger.info(f"Completado en {dur}s | Muestras: {len(dataset):,}")
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
