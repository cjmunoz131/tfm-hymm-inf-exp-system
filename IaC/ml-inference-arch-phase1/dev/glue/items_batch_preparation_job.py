"""
AWS Glue Python Shell Job: Prepare Items JSONL for Batch Transform
===================================================================
Prepara el dataset de ítems para el SageMaker Batch Transform del Item Tower.

Flujo:
  1. Lee Gold parquet (ítems únicos: movieId_idx + genres_multihot)
  2. Lee embeddings_catalog.pkl (text_emb + img_emb por item)
  3. JOIN → genera JSONL con el schema exacto del Item Tower inference
  4. Output: s3://.../inference-input/items_for_batch.jsonl

Schema de salida (una línea JSON por ítem):
  {"item_idx": N, "genres_multihot": [...20D], "text_emb": [...1024D], "img_emb": [...1024D]}

Argumentos Glue (--key value):
  - JOB_NAME
  - gold_interactions_path:  s3 path al parquet de Gold (interactions)
  - embeddings_catalog_path: s3 path al embeddings_catalog.pkl
  - output_path:             s3 path para el JSONL de salida
  - pipeline_id
  - correlation_id
  - aws_region
"""

import sys
import json
import logging
import time
import pickle
import io
from collections import defaultdict

import boto3
import pandas as pd
import numpy as np

# ============================================================
# ARGUMENTOS
# ============================================================
from awsglue.utils import getResolvedOptions

args = getResolvedOptions(sys.argv, [
    'gold_interactions_path',
    'embeddings_catalog_path',
    'output_path',
    'aws_region',
])

JOB_NAME = args.get('JOB_NAME', 'items_batch_preparation_job')
GOLD_INTERACTIONS_PATH = args['gold_interactions_path']
EMBEDDINGS_CATALOG_PATH = args['embeddings_catalog_path']
OUTPUT_PATH = args['output_path']
PIPELINE_ID = args.get('pipeline_id', 'UNKNOWN')
CORRELATION_ID = args.get('correlation_id', 'UNKNOWN')
REGION = args['aws_region']

# ============================================================
# LOGGING
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'))
logger.addHandler(handler)

# ============================================================
# CLIENTES AWS
# ============================================================
s3_client = boto3.client('s3', region_name=REGION)


# ============================================================
# FUNCIONES UTILITARIAS
# ============================================================
def parse_s3_path(s3_path: str) -> tuple:
    """Parsea s3://bucket/prefix en (bucket, prefix)."""
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def read_parquet_from_s3(s3_path: str) -> pd.DataFrame:
    """
    Lee parquet desde S3.
    Soporta tanto archivos individuales como directorios particionados.
    """
    bucket, prefix = parse_s3_path(s3_path)
    logger.info(f"Leyendo parquet desde: s3://{bucket}/{prefix}")

    # Listar objetos para manejar directorios particionados
    paginator = s3_client.get_paginator('list_objects_v2')
    parquet_files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.parquet') or key.endswith('.snappy.parquet'):
                parquet_files.append(key)

    if not parquet_files:
        # Intentar leer como archivo único
        parquet_files = [prefix]

    dfs = []
    for pf in parquet_files:
        obj = s3_client.get_object(Bucket=bucket, Key=pf)
        df_part = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    logger.info(f"  → Registros leídos: {len(df):,}")
    return df


def read_pickle_from_s3(s3_path: str):
    """Lee un archivo pickle desde S3."""
    bucket, prefix = parse_s3_path(s3_path)
    logger.info(f"Leyendo pickle desde: s3://{bucket}/{prefix}")

    obj = s3_client.get_object(Bucket=bucket, Key=prefix)
    data = pickle.loads(obj['Body'].read())
    logger.info(f"  → Tipo: {type(data).__name__}, Keys: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    return data


def write_jsonl_to_s3(records: list, s3_path: str):
    """Escribe una lista de dicts como JSONL a S3."""
    bucket, key = parse_s3_path(s3_path)
    logger.info(f"Escribiendo JSONL a: s3://{bucket}/{key}")

    # Construir JSONL en memoria
    lines = []
    for record in records:
        lines.append(json.dumps(record, separators=(',', ':')))

    body = '\n'.join(lines)
    body_bytes = body.encode('utf-8')

    # Upload
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body_bytes,
        ContentType='application/jsonlines'
    )
    size_mb = len(body_bytes) / (1024 * 1024)
    logger.info(f"  → {len(records):,} registros escritos ({size_mb:.2f} MB)")


# ============================================================
# LÓGICA PRINCIPAL
# ============================================================
def extract_unique_items(df_gold: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae los ítems únicos del Gold parquet con sus features.
    Columns esperadas: movieId_idx, movieId (original), genres_multihot
    Incluye movieId para que viaje como passthrough en el Batch Transform output.
    """
    logger.info("Extrayendo ítems únicos del Gold dataset...")

    # Verificar columnas requeridas
    required_cols = ['movieId_idx', 'genres_multihot']
    for col in required_cols:
        if col not in df_gold.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en Gold parquet. "
                             f"Columnas disponibles: {list(df_gold.columns)}")

    # Determinar nombre de columna movieId (puede ser movieId o movieid)
    movie_id_col = None
    for candidate in ['movieId', 'movieid', 'movie_id']:
        if candidate in df_gold.columns:
            movie_id_col = candidate
            break

    if movie_id_col is None:
        raise ValueError(f"No se encontró columna movieId en Gold. "
                         f"Columnas: {list(df_gold.columns)}")

    select_cols = ['movieId_idx', movie_id_col, 'genres_multihot']
    df_items = df_gold[select_cols].drop_duplicates(subset=['movieId_idx']).reset_index(drop=True)

    # Normalizar nombre a 'movie_id'
    if movie_id_col != 'movie_id':
        df_items = df_items.rename(columns={movie_id_col: 'movie_id'})

    logger.info(f"  → Ítems únicos: {len(df_items):,}")
    return df_items


def build_batch_records(df_items: pd.DataFrame, embeddings_catalog: dict) -> list:
    """
    Construye los registros JSONL para el Batch Transform del Item Tower.
    
    Hace JOIN entre:
      - df_items: movieId_idx + movie_id + genres_multihot (del Gold)
      - embeddings_catalog: {movieId_idx: {"text_emb": [...], "img_emb": [...]}}
    
    Incluye movie_id como passthrough para que el Batch Transform output
    lo conserve y el indexador pueda hacer JOIN con Silver sin encoders.
    
    Solo incluye ítems que tengan embeddings disponibles.
    """
    logger.info("Construyendo registros para Batch Transform...")

    records = []
    items_sin_emb = 0
    items_emb_incompleto = 0

    for _, row in df_items.iterrows():
        item_idx = int(row['movieId_idx'])
        movie_id = int(row['movie_id'])

        # Buscar embeddings por movieId (key real del embeddings_catalog)
        emb_data = embeddings_catalog.get(movie_id)
        if emb_data is None:
            items_sin_emb += 1
            continue

        text_emb = emb_data.get('text_emb')
        img_emb = emb_data.get('img_emb')

        if text_emb is None or img_emb is None:
            items_emb_incompleto += 1
            continue

        # Convertir a listas nativas de Python (numpy arrays → list)
        genres_multihot = row['genres_multihot']
        if isinstance(genres_multihot, np.ndarray):
            genres_multihot = genres_multihot.tolist()
        elif isinstance(genres_multihot, str):
            genres_multihot = json.loads(genres_multihot)

        if isinstance(text_emb, np.ndarray):
            text_emb = text_emb.tolist()
        if isinstance(img_emb, np.ndarray):
            img_emb = img_emb.tolist()

        # Validación de dimensiones
        assert len(genres_multihot) == 20, f"genres_multihot debe ser 20D, got {len(genres_multihot)}"
        assert len(text_emb) == 1024, f"text_emb debe ser 1024D, got {len(text_emb)}"
        assert len(img_emb) == 1024, f"img_emb debe ser 1024D, got {len(img_emb)}"

        record = {
            "item_idx": item_idx,
            "movie_id": movie_id,
            "genres_multihot": genres_multihot,
            "text_emb": text_emb,
            "img_emb": img_emb
        }
        records.append(record)

    logger.info(f"  → Registros generados: {len(records):,}")
    logger.info(f"  → Ítems sin embeddings: {items_sin_emb:,}")
    logger.info(f"  → Ítems con embeddings incompletos: {items_emb_incompleto:,}")

    if len(records) == 0:
        raise RuntimeError("No se generaron registros. Verificar que embeddings_catalog "
                           "tiene keys coincidentes con movieId_idx del Gold parquet.")

    return records


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================
def main():
    inicio = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Job: {JOB_NAME}")
    logger.info(f"Pipeline ID: {PIPELINE_ID}")
    logger.info(f"Correlation ID: {CORRELATION_ID}")
    logger.info(f"{'='*60}")

    # 1. CARGAR GOLD PARQUET (interactions → ítems únicos)
    logger.info("\n[PASO 1/4] Cargando Gold parquet (interactions)...")
    df_gold = read_parquet_from_s3(GOLD_INTERACTIONS_PATH)
    logger.info(f"  Columnas Gold: {list(df_gold.columns)}")

    # 2. EXTRAER ÍTEMS ÚNICOS
    logger.info("\n[PASO 2/4] Extrayendo ítems únicos...")
    df_items = extract_unique_items(df_gold)

    # 3. CARGAR EMBEDDINGS CATALOG
    logger.info("\n[PASO 3/4] Cargando embeddings_catalog.pkl...")
    embeddings_catalog = read_pickle_from_s3(EMBEDDINGS_CATALOG_PATH)

    # 4. CONSTRUIR Y ESCRIBIR JSONL
    logger.info("\n[PASO 4/4] Generando JSONL para Batch Transform...")
    records = build_batch_records(df_items, embeddings_catalog)
    write_jsonl_to_s3(records, OUTPUT_PATH)

    # RESUMEN
    duracion = round(time.time() - inicio, 2)
    logger.info(f"\n{'='*60}")
    logger.info(f"Job completado exitosamente en {duracion}s")
    logger.info(f"  Ítems procesados: {len(records):,}")
    logger.info(f"  Output: {OUTPUT_PATH}")
    logger.info(f"{'='*60}")


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Job FALLÓ: {e}")
        raise
