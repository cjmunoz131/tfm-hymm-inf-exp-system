"""
AWS Glue Python Shell Job: Enrich Items + Index to OpenSearch Serverless (AOSS)
================================================================================
Lee el output del Batch Transform del Item Tower, lo enriquece con metadata
legible de la capa Silver, y lo indexa en Amazon OpenSearch Serverless (AOSS)
como vector store kNN (Faiss HNSW).

Flujo:
  1. Lee output del Batch Transform (JSONL con item_embedding 64D + attention_weights)
  2. Lee metadata legible de Silver (cleansed_movies: título, géneros, sinopsis, año, poster)
  3. Lee encoders.pkl para mapear movieId_idx → movieId original
  4. JOIN: embedding + attention_weights + metadata legible
  5. Construye documentos OpenSearch completos
  6. Crea índice kNN si no existe (Faiss HNSW, cosinesimil, 64D)
  7. Bulk index a OpenSearch Serverless

Schema del documento OpenSearch:
  {
    "item_idx": int,
    "movie_id": int,
    "title": str,
    "genres": str,
    "synopsis": str,
    "release_year": int,
    "poster_path": str,
    "director": str,
    "item_embedding": [64D float],
    "attention_weights": {"category": float, "text": float, "image": float},
    "indexed_at": str (ISO timestamp)
  }

Argumentos Glue (--key value):
  - JOB_NAME
  - batch_transform_output_path: s3 path al output del Batch Transform (.jsonl.out)
  - silver_movies_path:          s3 path al parquet de cleansed_movies en Silver
  - encoders_path:               s3 path al encoders.pkl
  - opensearch_endpoint:         endpoint de AOSS collection (sin https://)
  - opensearch_index_name:       nombre del índice kNN
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
from datetime import datetime, timezone

import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

# ============================================================
# ARGUMENTOS
# ============================================================
def get_args():
    """Parse Glue job arguments from sys.argv."""
    args = {}
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--') and i + 1 < len(sys.argv):
            key = arg[2:]
            value = sys.argv[i + 1]
            if not value.startswith('--'):
                args[key] = value
    return args

args = get_args()

JOB_NAME = args.get('JOB_NAME', 'items_opensearch_indexing_job')
BATCH_TRANSFORM_OUTPUT_PATH = args['batch_transform_output_path']
SILVER_MOVIES_PATH = args['silver_movies_path']
ENCODERS_PATH = args['encoders_path']
OPENSEARCH_ENDPOINT = args['opensearch_endpoint']
OPENSEARCH_INDEX_NAME = args.get('opensearch_index_name', 'hymmrec-items-vectors')
PIPELINE_ID = args.get('pipeline_id', 'UNKNOWN')
CORRELATION_ID = args.get('correlation_id', 'UNKNOWN')
REGION = args.get('aws_region', 'us-east-1')

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

# OpenSearch usa requests con SigV4 signing
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
import urllib.request
import urllib.error

session = boto3.Session(region_name=REGION)
credentials = session.get_credentials().get_frozen_credentials()

# ============================================================
# CONSTANTES
# ============================================================
EMBEDDING_DIM = 64
BULK_BATCH_SIZE = 500  # Documentos por bulk request

# SigV4 service name: "aoss" para OpenSearch Serverless
SIGV4_SERVICE = "aoss"

# Mapping kNN para AOSS (Faiss engine, HNSW, cosinesimil)
# AOSS gestiona shards y replicas internamente — no se configuran explícitamente
OPENSEARCH_KNN_SETTINGS = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512
        }
    },
    "mappings": {
        "properties": {
            "item_idx": {"type": "integer"},
            "movie_id": {"type": "integer"},
            "title": {"type": "text", "analyzer": "standard"},
            "genres": {"type": "keyword"},
            "synopsis": {"type": "text", "analyzer": "standard"},
            "release_year": {"type": "integer"},
            "poster_path": {"type": "keyword"},
            "director": {"type": "keyword"},
            "item_embedding": {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "faiss",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            },
            "attention_weights": {
                "type": "object",
                "properties": {
                    "category": {"type": "float"},
                    "text": {"type": "float"},
                    "image": {"type": "float"}
                }
            },
            "indexed_at": {"type": "date"}
        }
    }
}


# ============================================================
# FUNCIONES UTILITARIAS S3
# ============================================================
def parse_s3_path(s3_path: str) -> tuple:
    """Parsea s3://bucket/prefix en (bucket, prefix)."""
    path = s3_path.replace("s3://", "")
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def read_jsonl_from_s3(s3_path: str) -> list:
    """
    Lee JSONL desde S3. Soporta archivos individuales y directorios
    (Batch Transform genera archivos .out en un directorio).
    """
    bucket, prefix = parse_s3_path(s3_path)
    logger.info(f"Leyendo JSONL desde: s3://{bucket}/{prefix}")

    # Listar archivos en el path
    paginator = s3_client.get_paginator('list_objects_v2')
    files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            # Batch Transform genera .out files
            if key.endswith('.out') or key.endswith('.jsonl') or key.endswith('.json'):
                files.append(key)

    # Si no encontramos archivos con extensión conocida, intentar el prefix directo
    if not files:
        files = [prefix]

    records = []
    for f in files:
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=f)
            content = obj['Body'].read().decode('utf-8')
            for line in content.strip().split('\n'):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Error leyendo {f}: {e}")

    logger.info(f"  → Registros leídos: {len(records):,}")
    return records


def read_parquet_from_s3(s3_path: str) -> pd.DataFrame:
    """Lee parquet desde S3 (soporta directorios particionados)."""
    bucket, prefix = parse_s3_path(s3_path)
    logger.info(f"Leyendo parquet desde: s3://{bucket}/{prefix}")

    paginator = s3_client.get_paginator('list_objects_v2')
    parquet_files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.parquet') or key.endswith('.snappy.parquet'):
                parquet_files.append(key)

    if not parquet_files:
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
    logger.info(f"  → Tipo: {type(data).__name__}")
    return data


# ============================================================
# FUNCIONES OPENSEARCH (SigV4 signed requests)
# ============================================================
def _sign_request(method: str, url: str, body: str = None, headers: dict = None) -> dict:
    """Firma una request HTTP con SigV4 para OpenSearch Serverless (AOSS)."""
    if headers is None:
        headers = {'Content-Type': 'application/json'}

    request = AWSRequest(method=method, url=url, data=body, headers=headers)
    SigV4Auth(credentials, SIGV4_SERVICE, REGION).add_auth(request)
    return dict(request.headers)


def opensearch_request(method: str, path: str, body: dict = None) -> dict:
    """Ejecuta una request HTTP firmada contra OpenSearch Serverless."""
    url = f"https://{OPENSEARCH_ENDPOINT}/{path}"
    body_str = json.dumps(body) if body else None

    signed_headers = _sign_request(method, url, body_str)

    req = urllib.request.Request(
        url=url,
        data=body_str.encode('utf-8') if body_str else None,
        headers=signed_headers,
        method=method
    )

    try:
        with urllib.request.urlopen(req) as response:
            response_body = response.read().decode('utf-8')
            if not response_body:
                return {"status": response.status}
            return json.loads(response_body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.readable() else ''
        logger.error(f"OpenSearch {method} {path} → {e.code}: {error_body}")
        raise


def opensearch_bulk_request(path: str, bulk_body: str) -> dict:
    """Ejecuta un bulk request firmado contra OpenSearch Serverless."""
    url = f"https://{OPENSEARCH_ENDPOINT}/{path}"
    headers = {'Content-Type': 'application/x-ndjson'}

    signed_headers = _sign_request('POST', url, bulk_body, headers)

    req = urllib.request.Request(
        url=url,
        data=bulk_body.encode('utf-8'),
        headers=signed_headers,
        method='POST'
    )

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.readable() else ''
        logger.error(f"OpenSearch BULK → {e.code}: {error_body}")
        raise


def ensure_index_exists():
    """
    Crea el índice kNN en AOSS si no existe.
    
    En OpenSearch Serverless NO se puede eliminar un índice vía API (a diferencia
    de managed). La estrategia de refresh es idempotente: usamos _id por item_idx,
    así cada ejecución sobrescribe documentos existentes sin duplicar.
    """
    logger.info(f"Verificando/creando índice: {OPENSEARCH_INDEX_NAME}")

    # Verificar si el índice ya existe
    try:
        opensearch_request('HEAD', OPENSEARCH_INDEX_NAME)
        logger.info(f"  → Índice ya existe, se usará upsert por _id (idempotente)")
        return
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.info(f"  → Índice no existe, creando...")
        else:
            raise

    # Crear índice con mappings kNN (Faiss HNSW)
    result = opensearch_request('PUT', OPENSEARCH_INDEX_NAME, OPENSEARCH_KNN_SETTINGS)
    logger.info(f"  → Índice creado: {result.get('acknowledged', False)}")


def bulk_index_documents(documents: list):
    """
    Indexa documentos en OpenSearch usando bulk API.
    Procesa en batches de BULK_BATCH_SIZE para evitar timeouts.
    """
    logger.info(f"Indexando {len(documents):,} documentos en batches de {BULK_BATCH_SIZE}...")

    total_indexed = 0
    total_errors = 0

    for i in range(0, len(documents), BULK_BATCH_SIZE):
        batch = documents[i:i + BULK_BATCH_SIZE]

        # Construir bulk body (NDJSON)
        lines = []
        for doc in batch:
            action = {"index": {"_index": OPENSEARCH_INDEX_NAME, "_id": str(doc['item_idx'])}}
            lines.append(json.dumps(action))
            lines.append(json.dumps(doc))

        bulk_body = '\n'.join(lines) + '\n'

        # Ejecutar bulk
        try:
            result = opensearch_bulk_request('_bulk', bulk_body)
            errors = result.get('errors', False)
            if errors:
                error_items = [item for item in result.get('items', [])
                               if 'error' in item.get('index', {})]
                total_errors += len(error_items)
                if error_items:
                    logger.warning(f"  Batch {i // BULK_BATCH_SIZE + 1}: {len(error_items)} errores")
                    logger.warning(f"  Ejemplo: {error_items[0]}")
            total_indexed += len(batch) - (len(error_items) if errors else 0)
        except Exception as e:
            logger.error(f"  Error en batch {i // BULK_BATCH_SIZE + 1}: {e}")
            total_errors += len(batch)

        # Log progreso cada 5 batches
        if (i // BULK_BATCH_SIZE + 1) % 5 == 0:
            logger.info(f"  Progreso: {min(i + BULK_BATCH_SIZE, len(documents)):,}/{len(documents):,}")

    logger.info(f"  → Total indexados: {total_indexed:,} | Errores: {total_errors:,}")
    return total_indexed, total_errors


# ============================================================
# LÓGICA DE ENRIQUECIMIENTO
# ============================================================
def build_idx_to_movieid_map(encoders: dict) -> dict:
    """
    Extrae el mapeo inverso movieId_idx → movieId desde los encoders.
    Los encoders contienen un LabelEncoder que mapea movieId → movieId_idx.
    """
    logger.info("Construyendo mapeo idx → movieId desde encoders...")

    # El encoder de movies suele estar en 'movieId_encoder' o 'movie_encoder'
    movie_encoder = None
    for key in ['movieId_encoder', 'movie_encoder', 'item_encoder']:
        if key in encoders:
            movie_encoder = encoders[key]
            break

    if movie_encoder is None:
        # Intentar con la key directa 'movieId' si es un dict de mappings
        if 'movieId' in encoders:
            movie_encoder = encoders['movieId']

    if movie_encoder is None:
        logger.warning(f"Keys disponibles en encoders: {list(encoders.keys())}")
        raise ValueError("No se encontró el encoder de movieId en encoders.pkl. "
                         "Keys esperadas: movieId_encoder, movie_encoder, item_encoder")

    # Si es un LabelEncoder de sklearn
    if hasattr(movie_encoder, 'classes_'):
        idx_to_movieid = {idx: int(mid) for idx, mid in enumerate(movie_encoder.classes_)}
    # Si es un dict {movieId: idx}
    elif isinstance(movie_encoder, dict):
        idx_to_movieid = {v: k for k, v in movie_encoder.items()}
    else:
        raise ValueError(f"Tipo de encoder no soportado: {type(movie_encoder)}")

    logger.info(f"  → Mapeos construidos: {len(idx_to_movieid):,}")
    return idx_to_movieid


def enrich_with_metadata(
    batch_output: list,
    df_silver_movies: pd.DataFrame,
    idx_to_movieid: dict
) -> list:
    """
    Enriquece el output del Batch Transform con metadata legible de Silver.
    
    Args:
        batch_output: Lista de dicts del Batch Transform output
                      {item_embedding: [...], attention_weights: {...}}
        df_silver_movies: DataFrame de cleansed_movies
        idx_to_movieid: Mapeo movieId_idx → movieId
    
    Returns:
        Lista de documentos OpenSearch completos
    """
    logger.info("Enriqueciendo embeddings con metadata de Silver...")

    # Crear lookup de metadata por movieId
    metadata_lookup = {}
    for _, row in df_silver_movies.iterrows():
        mid = int(row.get('movieId', 0))
        metadata_lookup[mid] = {
            'title': str(row.get('titulo', row.get('title', ''))),
            'genres': str(row.get('generos', row.get('genres', ''))),
            'synopsis': str(row.get('sinopsis', row.get('synopsis', row.get('overview', ''))))[:1000],
            'release_year': _extract_year(row.get('fecha_lanzamiento', row.get('release_date', ''))),
            'poster_path': str(row.get('poster_path', '')),
            'director': str(row.get('director', '')),
        }

    documents = []
    items_sin_metadata = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for record in batch_output:
        item_idx = record.get('item_idx')
        item_embedding = record.get('item_embedding')
        attention_weights = record.get('attention_weights', {})

        if item_idx is None or item_embedding is None:
            continue

        item_idx = int(item_idx)

        # Mapear idx → movieId
        movie_id = idx_to_movieid.get(item_idx)
        if movie_id is None:
            items_sin_metadata += 1
            continue

        # Obtener metadata
        meta = metadata_lookup.get(movie_id, {})
        if not meta:
            items_sin_metadata += 1
            # Aún así indexamos con metadata vacía (el embedding es lo importante)
            meta = {'title': '', 'genres': '', 'synopsis': '',
                    'release_year': 0, 'poster_path': '', 'director': ''}

        # Construir documento OpenSearch
        doc = {
            'item_idx': item_idx,
            'movie_id': movie_id,
            'title': meta['title'],
            'genres': meta['genres'],
            'synopsis': meta['synopsis'],
            'release_year': meta['release_year'],
            'poster_path': meta['poster_path'],
            'director': meta['director'],
            'item_embedding': item_embedding,
            'attention_weights': attention_weights,
            'indexed_at': now_iso
        }
        documents.append(doc)

    logger.info(f"  → Documentos enriquecidos: {len(documents):,}")
    logger.info(f"  → Ítems sin metadata Silver: {items_sin_metadata:,}")

    return documents


def _extract_year(date_value) -> int:
    """Extrae el año de una fecha (string o datetime)."""
    if pd.isna(date_value) or date_value is None or date_value == '':
        return 0
    try:
        if isinstance(date_value, str):
            return int(date_value[:4])
        elif hasattr(date_value, 'year'):
            return int(date_value.year)
    except (ValueError, TypeError):
        pass
    return 0


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================
def main():
    inicio = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"Job: {JOB_NAME}")
    logger.info(f"Pipeline ID: {PIPELINE_ID}")
    logger.info(f"Correlation ID: {CORRELATION_ID}")
    logger.info(f"OpenSearch Endpoint: {OPENSEARCH_ENDPOINT}")
    logger.info(f"OpenSearch Index: {OPENSEARCH_INDEX_NAME}")
    logger.info(f"{'='*60}")

    # 1. LEER OUTPUT DEL BATCH TRANSFORM
    logger.info("\n[PASO 1/5] Leyendo output del Batch Transform...")
    batch_output = read_jsonl_from_s3(BATCH_TRANSFORM_OUTPUT_PATH)

    if not batch_output:
        raise RuntimeError(f"No se encontraron registros en: {BATCH_TRANSFORM_OUTPUT_PATH}")

    logger.info(f"  Ejemplo primer registro (keys): {list(batch_output[0].keys())}")

    # 2. LEER METADATA DE SILVER
    logger.info("\n[PASO 2/5] Leyendo metadata de Silver (cleansed_movies)...")
    df_silver_movies = read_parquet_from_s3(SILVER_MOVIES_PATH)
    logger.info(f"  Columnas Silver: {list(df_silver_movies.columns)}")

    # 3. LEER ENCODERS (para mapeo idx → movieId)
    logger.info("\n[PASO 3/5] Leyendo encoders.pkl...")
    encoders = read_pickle_from_s3(ENCODERS_PATH)
    idx_to_movieid = build_idx_to_movieid_map(encoders)

    # 4. ENRIQUECER DOCUMENTOS
    logger.info("\n[PASO 4/5] Enriqueciendo documentos con metadata...")
    documents = enrich_with_metadata(batch_output, df_silver_movies, idx_to_movieid)

    if not documents:
        raise RuntimeError("No se generaron documentos enriquecidos. "
                           "Verificar consistencia entre Batch Transform output, "
                           "encoders y Silver movies.")

    # 5. INDEXAR EN OPENSEARCH
    logger.info("\n[PASO 5/5] Indexando en OpenSearch...")
    ensure_index_exists()
    total_indexed, total_errors = bulk_index_documents(documents)

    # RESUMEN
    duracion = round(time.time() - inicio, 2)
    logger.info(f"\n{'='*60}")
    logger.info(f"Job completado exitosamente en {duracion}s")
    logger.info(f"  Documentos indexados: {total_indexed:,}")
    logger.info(f"  Errores: {total_errors:,}")
    logger.info(f"  Índice: {OPENSEARCH_INDEX_NAME}")
    logger.info(f"{'='*60}")

    if total_errors > 0 and total_indexed == 0:
        raise RuntimeError(f"Indexación falló completamente: {total_errors} errores, 0 indexados")


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Job FALLÓ: {e}")
        raise
