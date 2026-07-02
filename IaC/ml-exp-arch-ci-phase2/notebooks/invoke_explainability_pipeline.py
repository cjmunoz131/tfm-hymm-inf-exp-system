"""
==============================================================================
HYMM-REC Explainability CI Pipeline: Orquestador desde SageMaker Notebook
==============================================================================
Script para ejecutar desde SageMaker Notebook Instance el pipeline completo
de CI del modelo de explicabilidad (Llama 3.1 8B con QLoRA).

Flujo:
  Step 1: Processing Job — Clean Gold Set + Split (train/val/test)
  Step 2: Training Job — Fine-tuning QLoRA (SFTTrainer en ml.g5.2xlarge)
  Step 3: Deploy Endpoint Temporal (ml.g5.2xlarge) + Evaluación desde notebook
  Step 4: Delete Endpoint Temporal
  Step 5: Model Registry — Registrar si métricas superan umbral

Estrategia de evaluación:
  - Se despliega un endpoint TEMPORAL con el modelo fine-tuned
  - El notebook (ml.t3.medium, sin GPU) invoca el endpoint vía SageMaker Runtime
  - Se calculan métricas localmente (ROUGE-L, Exact Match, Keyword Overlap)
  - Se elimina el endpoint al finalizar

Pre-requisitos:
  - gold_dataset_clean.jsonl en S3 (post human-in-the-loop review)
  - HuggingFace token con acceso a Meta-Llama-3.1-8B-Instruct
  - SageMaker Execution Role con permisos a S3, ECR, SageMaker
  - Quota: ml.g5.2xlarge for endpoint usage >= 2
  - Scripts subidos a S3 por Terraform (ml-exp-arch-ci-phase2)
"""

# ==============================================================================
# CELDA 1: CONFIGURACIÓN
# ==============================================================================
import boto3
import sagemaker
import os
import json
import time
import re
import numpy as np
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.huggingface import HuggingFace, HuggingFaceModel

REGION = "us-east-1"
ROLE = get_execution_role()
SESSION = sagemaker.Session()

# Buckets
PLATINUM_BUCKET = "hymmrec-sagemaker-assets"
GOLD_BUCKET = "hymmrec-dilkehousegold01"

# S3 Paths
S3_GOLD_SET_INPUT = f"s3://{GOLD_BUCKET}/data/ml_recommendations/explainability/"
S3_EXPLAINABILITY_PREFIX = f"s3://{PLATINUM_BUCKET}/hymmrec/explainability"
S3_SCRIPTS = f"{S3_EXPLAINABILITY_PREFIX}/scripts"
S3_SPLITS_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/datasets/splits/"
S3_TRAINING_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/training-output/"
S3_EVAL_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/evaluation/"
S3_MODEL_ARTIFACTS = f"{S3_EXPLAINABILITY_PREFIX}/model-artifacts/"

# Scripts locales (relativos al notebook)
LOCAL_PROCESSING_DIR = "../dev/processing/"
LOCAL_TRAINING_DIR = "../dev/training/"
LOCAL_EVALUATION_DIR = "../dev/evaluation/"

# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set via environment variable

# Instance Types
PROCESSING_INSTANCE_TYPE = "ml.m5.large"       # Clean + Split (CPU, no GPU needed)
TRAINING_INSTANCE_TYPE = "ml.g5.2xlarge"       # QLoRA Fine-Tuning (GPU A10G)
ENDPOINT_INSTANCE_TYPE = "ml.g5.2xlarge"       # Endpoint temporal para evaluación

# Endpoint temporal
EVAL_ENDPOINT_NAME = "hymmrec-exp-eval-temp"

# Model Registry
MODEL_PACKAGE_GROUP_NAME = "hymmrec-explainability-llama"

print(f"Role: {ROLE}")
print(f"Region: {REGION}")
print(f"Gold Set: {S3_GOLD_SET_INPUT}")
print(f"Training Output: {S3_TRAINING_OUTPUT}")
print(f"Model: {MODEL_ID}")


# ==============================================================================
# CELDA 2: STEP 1 — PROCESSING JOB (Clean Gold Set + Split)
# ==============================================================================
# Input: gold_dataset_clean.jsonl (post human-in-the-loop)
# Output: train.jsonl, val.jsonl, test.jsonl (80/10/10)

print("\n" + "=" * 60)
print("STEP 1: PROCESSING JOB — Clean Gold Set + Split")
print("=" * 60)

clean_split_processor = SKLearnProcessor(
    role=ROLE,
    instance_type=PROCESSING_INSTANCE_TYPE,
    instance_count=1,
    framework_version="1.2-1",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-exp-clean-split",
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "explainability-processing"},
    ],
)

clean_split_processor.run(
    code=f"{LOCAL_PROCESSING_DIR}clean_and_split_job.py",
    arguments=[
        "--train-ratio", "0.8",
        "--val-ratio", "0.1",
        "--seed", "42",
    ],
    inputs=[
        ProcessingInput(
            source=S3_GOLD_SET_INPUT,
            destination="/opt/ml/processing/input/gold",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/splits",
            destination=S3_SPLITS_OUTPUT,
            output_name="splits",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/metrics",
            destination=f"{S3_EXPLAINABILITY_PREFIX}/processing-metrics/",
            output_name="metrics",
        ),
    ],
    logs=True,
    wait=True,
)

print("Step 1 completado: Clean + Split")


# ==============================================================================
# CELDA 3: VERIFICAR SPLITS
# ==============================================================================
print("\nVerificando splits en S3...")
s3_client = boto3.client("s3")

for split_name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    key = f"hymmrec/explainability/datasets/splits/{split_name}"
    try:
        response = s3_client.head_object(Bucket=PLATINUM_BUCKET, Key=key)
        size_kb = response["ContentLength"] / 1024
        print(f"  {split_name}: {size_kb:.1f} KB")
    except s3_client.exceptions.ClientError:
        print(f"  {split_name}: NO ENCONTRADO")

# Leer métricas del split
metrics_key = "hymmrec/explainability/processing-metrics/split_metrics.json"
try:
    obj = s3_client.get_object(Bucket=PLATINUM_BUCKET, Key=metrics_key)
    split_metrics = json.loads(obj["Body"].read().decode())
    print(f"\nMétricas del split:")
    print(f"  Total registros: {split_metrics['total_raw_records']}")
    print(f"  Aprobados: {split_metrics['approved_records']} ({split_metrics['approval_rate']*100:.1f}%)")
    print(f"  Train: {split_metrics['train_size']} | Val: {split_metrics['val_size']} | Test: {split_metrics['test_size']}")
except Exception as e:
    print(f"  No se pudieron leer métricas: {e}")


# ==============================================================================
# CELDA 4: STEP 2 — TRAINING JOB (QLoRA Fine-Tuning)
# ==============================================================================
# Fine-tuning de Llama 3.1 8B con QLoRA usando HuggingFace Estimator.
# El script requiere: transformers, peft, trl, bitsandbytes, accelerate, datasets

print("\n" + "=" * 60)
print("STEP 2: TRAINING JOB — QLoRA Fine-Tuning Llama 3.1 8B")
print("=" * 60)

# HuggingFace Estimator con dependencias necesarias para QLoRA
huggingface_estimator = HuggingFace(
    entry_point="finetune_llama_qlora.py",
    source_dir=LOCAL_TRAINING_DIR,
    role=ROLE,
    instance_type=TRAINING_INSTANCE_TYPE,
    instance_count=1,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
    sagemaker_session=SESSION,
    base_job_name="hymmrec-exp-qlora-train",
    output_path=S3_TRAINING_OUTPUT,
    hyperparameters={
        "model-id": MODEL_ID,
        "hf-token": HF_TOKEN,
        "epochs": 3,
        "batch-size": 1,
        "gradient-accumulation-steps": 8,
        "learning-rate": 2e-4,
        "lora-r": 16,
        "lora-alpha": 32,
        "lora-dropout": 0.05,
        "seed": 42,
    },
    environment={
        "HF_TOKEN": HF_TOKEN,
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
    },
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "explainability-training"},
    ],
)

huggingface_estimator.fit(
    inputs={
        "train": S3_SPLITS_OUTPUT,
    },
    wait=True,
    logs="All",
)

print("Step 2 completado: Training QLoRA")
print(f"Model artifact: {huggingface_estimator.model_data}")

# Guardar la ruta del modelo para los pasos siguientes
S3_TRAINED_MODEL = huggingface_estimator.model_data
print(f"Modelo entrenado en: {S3_TRAINED_MODEL}")


# ==============================================================================
# CELDA 5: STEP 3 — DEPLOY ENDPOINT TEMPORAL PARA EVALUACIÓN
# ==============================================================================
# Desplegamos el modelo fine-tuned como endpoint temporal en ml.g5.2xlarge.
# El notebook (sin GPU) invoca el endpoint vía SageMaker Runtime API.
# Al finalizar la evaluación, se elimina el endpoint.

print("\n" + "=" * 60)
print("STEP 3: DEPLOY ENDPOINT TEMPORAL — ml.g5.2xlarge")
print("=" * 60)

# Crear HuggingFaceModel desde el artefacto del training job
huggingface_model = HuggingFaceModel(
    model_data=S3_TRAINED_MODEL,
    role=ROLE,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
    sagemaker_session=SESSION,
    env={
        "HF_TOKEN": HF_TOKEN,
        "HF_MODEL_ID": MODEL_ID,
        "SM_NUM_GPUS": "1",
        "MAX_INPUT_LENGTH": "1024",
        "MAX_TOTAL_TOKENS": "1100",
    },
)

print(f"Desplegando endpoint: {EVAL_ENDPOINT_NAME}")
print(f"Instancia: {ENDPOINT_INSTANCE_TYPE}")
print(f"Modelo: {S3_TRAINED_MODEL}")
print("Esto puede tomar 8-12 minutos...")

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type=ENDPOINT_INSTANCE_TYPE,
    endpoint_name=EVAL_ENDPOINT_NAME,
    tags=[
        {"Key": "project", "Value": "hymmrec"},
        {"Key": "phase", "Value": "explainability-evaluation"},
        {"Key": "lifecycle", "Value": "temporal"},
    ],
    wait=True,
)

print(f"Endpoint desplegado: {EVAL_ENDPOINT_NAME}")


# ==============================================================================
# CELDA 6: STEP 3b — EVALUACIÓN VÍA ENDPOINT (desde notebook sin GPU)
# ==============================================================================
# El notebook descarga el test.jsonl de S3, invoca el endpoint para cada ejemplo,
# y calcula métricas localmente (ROUGE-L, Exact Match, Keyword Overlap).
# No requiere GPU en el notebook — toda la inferencia la hace el endpoint.

print("\n" + "=" * 60)
print("STEP 3b: EVALUACIÓN — Invocando endpoint con test set")
print("=" * 60)

# --- Descargar test.jsonl ---
test_key = "hymmrec/explainability/datasets/splits/test.jsonl"
obj = s3_client.get_object(Bucket=PLATINUM_BUCKET, Key=test_key)
test_lines = obj["Body"].read().decode("utf-8").strip().split("\n")
test_data = [json.loads(line) for line in test_lines]
print(f"Test set cargado: {len(test_data)} ejemplos")

# --- Funciones de métricas (corren localmente en el notebook, sin GPU) ---

def compute_exact_match(predicted_keywords, gold_keywords):
    """1 si las 3 keywords coinciden exactamente (case-insensitive, order-insensitive)."""
    pred_set = set(k.strip().lower() for k in predicted_keywords)
    gold_set = set(k.strip().lower() for k in gold_keywords)
    return 1.0 if pred_set == gold_set else 0.0

def compute_keyword_overlap(predicted_keywords, gold_keywords):
    """Proporción de keywords gold que aparecen en la predicción."""
    pred_set = set(k.strip().lower() for k in predicted_keywords)
    gold_set = set(k.strip().lower() for k in gold_keywords)
    if len(gold_set) == 0:
        return 0.0
    return len(pred_set.intersection(gold_set)) / len(gold_set)

def compute_rouge_l(prediction, reference):
    """ROUGE-L F1 basado en Longest Common Subsequence."""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        table = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[m][n]

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# --- Invocar endpoint para cada ejemplo del test set ---
runtime_client = boto3.client("sagemaker-runtime", region_name=REGION)

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

exact_matches = []
keyword_overlaps = []
rouge_l_scores = []
results = []

print(f"\nEvaluando {len(test_data)} ejemplos...")

for i, muestra in enumerate(test_data):
    # Construir prompt (sin respuesta — el modelo la genera)
    prompt = PROMPT_TEMPLATE.format(
        instruction=muestra["instruction"],
        input=muestra["input"],
    )

    # Invocar endpoint
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 20,
            "temperature": 0.1,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        }
    })

    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=EVAL_ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload,
        )
        result = json.loads(response["Body"].read().decode())

        # Extraer texto generado
        if isinstance(result, list) and len(result) > 0:
            prediccion = result[0].get("generated_text", "").strip()
        elif isinstance(result, dict):
            prediccion = result.get("generated_text", "").strip()
        else:
            prediccion = str(result).strip()

    except Exception as e:
        print(f"  Error en ejemplo {i}: {e}")
        prediccion = ""

    # Post-procesamiento: limpiar y forzar 3 keywords
    prediccion = prediccion.split("\n")[0].strip()  # Solo primera línea
    prediccion = prediccion.rstrip(".")
    prediccion = re.sub(r"<\|eot_id\|>", "", prediccion).strip()
    etiquetas = [e.strip() for e in prediccion.split(",")]
    prediccion_final = ", ".join(etiquetas[:3])

    # Calcular métricas
    gold = muestra["output"]
    pred_keywords = [k.strip() for k in prediccion_final.split(",")]
    gold_keywords = [k.strip() for k in gold.split(",")]

    em = compute_exact_match(pred_keywords, gold_keywords)
    ko = compute_keyword_overlap(pred_keywords, gold_keywords)
    rl = compute_rouge_l(prediccion_final, gold)

    exact_matches.append(em)
    keyword_overlaps.append(ko)
    rouge_l_scores.append(rl)

    results.append({
        "idx": i,
        "gold": gold,
        "predicted": prediccion_final,
        "exact_match": em,
        "keyword_overlap": ko,
        "rouge_l": rl,
    })

    if (i + 1) % 25 == 0:
        print(f"  Evaluados: {i+1}/{len(test_data)} | "
              f"EM={np.mean(exact_matches):.3f} | "
              f"KO={np.mean(keyword_overlaps):.3f} | "
              f"ROUGE-L={np.mean(rouge_l_scores):.3f}")

# --- Calcular métricas agregadas ---
eval_metrics = {
    "total_examples": len(test_data),
    "exact_match": round(float(np.mean(exact_matches)), 4),
    "keyword_overlap": round(float(np.mean(keyword_overlaps)), 4),
    "rouge_l_f1": round(float(np.mean(rouge_l_scores)), 4),
    "exact_match_std": round(float(np.std(exact_matches)), 4),
    "keyword_overlap_std": round(float(np.std(keyword_overlaps)), 4),
    "rouge_l_std": round(float(np.std(rouge_l_scores)), 4),
}

print(f"\n{'='*60}")
print("RESULTADOS DE EVALUACIÓN")
print(f"{'='*60}")
print(f"  Exact Match:     {eval_metrics['exact_match']:.4f} (+/- {eval_metrics['exact_match_std']:.4f})")
print(f"  Keyword Overlap: {eval_metrics['keyword_overlap']:.4f} (+/- {eval_metrics['keyword_overlap_std']:.4f})")
print(f"  ROUGE-L F1:      {eval_metrics['rouge_l_f1']:.4f} (+/- {eval_metrics['rouge_l_std']:.4f})")
print(f"  Total ejemplos:  {eval_metrics['total_examples']}")

# --- Guardar reporte en S3 ---
eval_report_key = "hymmrec/explainability/evaluation/evaluation_report.json"
s3_client.put_object(
    Bucket=PLATINUM_BUCKET,
    Key=eval_report_key,
    Body=json.dumps(eval_metrics, indent=2).encode(),
    ContentType="application/json",
)
print(f"\nReporte guardado: s3://{PLATINUM_BUCKET}/{eval_report_key}")

# Guardar detalles (primeros 50 ejemplos)
eval_details_key = "hymmrec/explainability/evaluation/evaluation_details.json"
s3_client.put_object(
    Bucket=PLATINUM_BUCKET,
    Key=eval_details_key,
    Body=json.dumps(results[:50], indent=2, ensure_ascii=False).encode(),
    ContentType="application/json",
)


# ==============================================================================
# CELDA 7: STEP 4 — DELETE ENDPOINT TEMPORAL
# ==============================================================================
# Eliminamos el endpoint para dejar de incurrir en costos.

print("\n" + "=" * 60)
print("STEP 4: DELETE ENDPOINT TEMPORAL")
print("=" * 60)

try:
    predictor.delete_endpoint(delete_endpoint_config=True)
    print(f"Endpoint eliminado: {EVAL_ENDPOINT_NAME}")
except Exception as e:
    print(f"Error eliminando endpoint: {e}")
    print("Intenta eliminar manualmente desde la consola de SageMaker.")


# ==============================================================================
# CELDA 8: STEP 5 — MODEL REGISTRY (Condicional)
# ==============================================================================
# Registra el modelo en SageMaker Model Registry si supera umbrales.

print("\n" + "=" * 60)
print("STEP 5: MODEL REGISTRY")
print("=" * 60)

# Umbrales mínimos para aprobación
THRESHOLD_ROUGE_L = 0.5
THRESHOLD_KEYWORD_OVERLAP = 0.6

sm_client = boto3.client("sagemaker")

# Crear Model Package Group si no existe
try:
    sm_client.describe_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME
    )
    print(f"Model Package Group ya existe: {MODEL_PACKAGE_GROUP_NAME}")
except sm_client.exceptions.ClientError:
    sm_client.create_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        ModelPackageGroupDescription=(
            "HYMM-REC Explainability: Llama 3.1 8B fine-tuned con QLoRA "
            "para generar 3 keywords de explicación por recomendación."
        ),
        Tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "domain", "Value": "explainability"},
        ],
    )
    print(f"Model Package Group creado: {MODEL_PACKAGE_GROUP_NAME}")

# Determinar status basado en umbrales
rouge_ok = eval_metrics.get("rouge_l_f1", 0) >= THRESHOLD_ROUGE_L
overlap_ok = eval_metrics.get("keyword_overlap", 0) >= THRESHOLD_KEYWORD_OVERLAP

if rouge_ok and overlap_ok:
    approval_status = "Approved"
    print(f"\nMétricas superan umbrales:")
    print(f"  ROUGE-L >= {THRESHOLD_ROUGE_L}: {eval_metrics['rouge_l_f1']:.4f}")
    print(f"  Keyword Overlap >= {THRESHOLD_KEYWORD_OVERLAP}: {eval_metrics['keyword_overlap']:.4f}")
    print(f"  --> Aprobando automáticamente")
else:
    approval_status = "PendingManualApproval"
    print(f"\nMétricas NO superan umbrales:")
    print(f"  ROUGE-L: {eval_metrics.get('rouge_l_f1', 0):.4f} (umbral: {THRESHOLD_ROUGE_L})")
    print(f"  Keyword Overlap: {eval_metrics.get('keyword_overlap', 0):.4f} (umbral: {THRESHOLD_KEYWORD_OVERLAP})")
    print(f"  --> Requiere aprobación manual")

# Registrar modelo
metrics_uri = f"s3://{PLATINUM_BUCKET}/{eval_report_key}"

# Container de inferencia (HuggingFace LLM)
huggingface_inference_image = sagemaker.image_uris.retrieve(
    framework="huggingface",
    region=REGION,
    version="4.37.0",
    py_version="py310",
    instance_type="ml.g5.xlarge",
    image_scope="inference",
)

model_package_response = sm_client.create_model_package(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
    ModelPackageDescription=(
        f"Llama 3.1 8B QLoRA | "
        f"ROUGE-L={eval_metrics.get('rouge_l_f1', 0):.4f} | "
        f"Exact Match={eval_metrics.get('exact_match', 0):.4f} | "
        f"Keyword Overlap={eval_metrics.get('keyword_overlap', 0):.4f}"
    ),
    InferenceSpecification={
        "Containers": [
            {
                "Image": huggingface_inference_image,
                "ModelDataUrl": S3_TRAINED_MODEL,
                "Framework": "HUGGINGFACE",
            }
        ],
        "SupportedTransformInstanceTypes": ["ml.g5.xlarge", "ml.g5.2xlarge"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.g5.xlarge", "ml.g5.2xlarge"],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus=approval_status,
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": metrics_uri,
            }
        }
    },
    CustomerMetadataProperties={
        "model_id": MODEL_ID,
        "technique": "QLoRA-4bit",
        "lora_r": "16",
        "lora_alpha": "32",
        "rouge_l_f1": str(round(eval_metrics.get("rouge_l_f1", 0), 4)),
        "exact_match": str(round(eval_metrics.get("exact_match", 0), 4)),
        "keyword_overlap": str(round(eval_metrics.get("keyword_overlap", 0), 4)),
    },
)

model_package_arn = model_package_response["ModelPackageArn"]
print(f"\nModelo registrado:")
print(f"  ARN: {model_package_arn}")
print(f"  Status: {approval_status}")


# ==============================================================================
# CELDA 9: RESUMEN FINAL
# ==============================================================================
print("\n" + "=" * 60)
print("PIPELINE CI EXPLAINABILITY COMPLETADO")
print("=" * 60)

print(f"""
Model: {MODEL_ID}
Technique: QLoRA 4-bit (LoRA r=16, alpha=32)
Instance: {TRAINING_INSTANCE_TYPE}

Metrics:
  Exact Match:     {eval_metrics.get('exact_match', 'N/A')}
  Keyword Overlap: {eval_metrics.get('keyword_overlap', 'N/A')}
  ROUGE-L F1:      {eval_metrics.get('rouge_l_f1', 'N/A')}

Model Registry:
  Package Group: {MODEL_PACKAGE_GROUP_NAME}
  ARN: {model_package_arn}
  Status: {approval_status}

Artifacts:
  Splits: {S3_SPLITS_OUTPUT}
  Model: {S3_TRAINED_MODEL}
  Adapter: {S3_MODEL_ADAPTER}
  Evaluation: {S3_EVAL_OUTPUT}

Proximos pasos:
  1. Si status=Approved → Deploy CD pipeline (ml-exp-arch-cd-phase2)
  2. Si status=PendingManualApproval → Revisar métricas y aprobar manualmente
  3. Para inferencia batch: integrar en Step Function del inference pipeline
""")
