"""
==============================================================================
HYMM-REC Explainability: SageMaker Pipeline Definition (CI/CD Ready)
==============================================================================
Script que genera la definicion del SageMaker Pipeline para despliegue
via Terraform + Azure DevOps CI/CD.

DAG del Pipeline:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CleanAndSplit (SKLearn, ml.m5.large CPU)                               │
  │         │                                                               │
  │         ▼                                                               │
  │  QLoRAFineTuning (HuggingFace, ml.g5.12xlarge GPU)                      │
  │         │                                                               │
  │         ▼                                                               │
  │  EvaluateExplainability (PyTorch, ml.g5.2xlarge GPU)                    │
  │         │   (Base BF16 + Adapter LoRA en memoria, sin merge)            │
  │         ▼                                                               │
  │  QualityGateCheck (ConditionStep)                                       │
  │         ┌────┴────┐                                                     │
  │    Pass │         │ Fail                                                │
  │         ▼         ▼                                                     │
  │  RegisterModel   FailStep                                               │
  └─────────────────────────────────────────────────────────────────────────┘

Despliegue (CI/CD):
  1. Terraform sube scripts a S3 (aws_s3_object en main.tf)
  2. Terraform aplica infra (Model Package Group, IAM, etc.)
  3. Generar JSON:  python define_explainability_pipeline.py --definition
  4. Upsert:        python define_explainability_pipeline.py --upsert
  5. Ejecutar:      python define_explainability_pipeline.py --execute

Uso desde CLI:
  # Generar JSON para Terraform (guarda en sm-dag-pipelines/)
  python define_explainability_pipeline.py --definition --output ../sm-dag-pipelines/hymmrec-explainability-sm-pipeline-dev.json

  # Upsert directo (sin Terraform)
  python define_explainability_pipeline.py --upsert

  # Ejecutar pipeline existente
  python define_explainability_pipeline.py --execute --hf-token hf_xxx

NOTA: Los scripts se referencian como paths locales (source_dir/code).
      El SDK los empaqueta y sube a S3 automaticamente al llamar
      pipeline.definition() o pipeline.upsert().
==============================================================================
"""

import argparse
import json
import logging
import os
import sys

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.huggingface import HuggingFace
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTES Y CONFIGURACION
# ==============================================================================

PIPELINE_NAME = "hymmrec-explainability-ci"

PIPELINE_DESCRIPTION = (
    "Pipeline MLOps CI para HYMM-REC Explainability. "
    "Fine-tuning Llama 3.1 8B con QLoRA + Evaluacion (Adapter LoRA sin merge) + "
    "Quality Gate + Model Registry."
)

# --- Buckets ---
DEFAULT_GOLD_BUCKET = "hymmrec-dilkehousegold01"
DEFAULT_PLATINUM_BUCKET = "hymmrec-sagemaker-assets"

# --- S3 Paths ---
S3_GOLD_SET_INPUT = f"s3://{DEFAULT_GOLD_BUCKET}/data/ml_recommendations/explainability/goldset/"
S3_EXPLAINABILITY_PREFIX = f"s3://{DEFAULT_PLATINUM_BUCKET}/hymmrec/explainability"
S3_SPLITS_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/datasets/splits/"
S3_TRAINING_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/training-output/"
S3_EVAL_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/evaluation/"

# --- Scripts locales (relativos a este archivo) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSING_SCRIPT = os.path.join(SCRIPT_DIR, "..", "dev", "processing", "clean_and_split_job.py")
TRAINING_SOURCE_DIR = os.path.join(SCRIPT_DIR, "..", "dev", "training")
EVALUATION_SCRIPT = os.path.join(SCRIPT_DIR, "..", "dev", "evaluation", "evaluate_explainability.py")

# --- Instancias ---
PROCESSING_INSTANCE = "ml.m5.large"
TRAINING_INSTANCE = "ml.g5.12xlarge"
EVALUATION_INSTANCE = "ml.g5.12xlarge"

# --- Model Registry ---
MODEL_PACKAGE_GROUP_NAME = "hymmrec-explainability-llama"

# --- PyTorch Inference Image (para registro en Model Registry) ---
# El deploy real usa PyTorchModel con code/inference.py
PYTORCH_INFERENCE_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker"


# ==============================================================================
# PIPELINE PARAMETERS
# ==============================================================================

def define_pipeline_parameters():
    """Define parametros configurables en runtime del pipeline."""

    # Model
    param_model_id = ParameterString(
        name="ModelId",
        default_value="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    param_hf_token = ParameterString(
        name="HfToken",
        default_value="",
    )

    # Hyperparameters
    param_epochs = ParameterString(name="Epochs", default_value="3")
    param_learning_rate = ParameterString(name="LearningRate", default_value="0.0002")
    param_lora_r = ParameterString(name="LoraR", default_value="16")
    param_lora_alpha = ParameterString(name="LoraAlpha", default_value="32")
    param_batch_size = ParameterString(name="BatchSize", default_value="1")
    param_grad_accum = ParameterString(name="GradientAccumulationSteps", default_value="8")

    # Quality Gate Thresholds
    param_threshold_ko = ParameterFloat(name="ThresholdKeywordOverlap", default_value=0.4)
    param_threshold_rouge = ParameterFloat(name="ThresholdRougeL", default_value=0.3)

    return {
        "model_id": param_model_id,
        "hf_token": param_hf_token,
        "epochs": param_epochs,
        "learning_rate": param_learning_rate,
        "lora_r": param_lora_r,
        "lora_alpha": param_lora_alpha,
        "batch_size": param_batch_size,
        "grad_accum": param_grad_accum,
        "threshold_ko": param_threshold_ko,
        "threshold_rouge": param_threshold_rouge,
    }


# ==============================================================================
# STEP 1: Clean Gold Set + Split (SKLearn Processing, CPU)
# ==============================================================================

def create_step_clean_split(params, role, session):
    """
    Processing Job: Valida y limpia el gold set, split train/val/test 80/10/10.
    Input: gold_dataset_clean.jsonl (post human-in-the-loop)
    Output: train.jsonl, val.jsonl, test.jsonl
    """
    sklearn_processor = SKLearnProcessor(
        role=role,
        instance_type=PROCESSING_INSTANCE,
        instance_count=1,
        framework_version="1.2-1",
        sagemaker_session=session,
        base_job_name="hymmrec-exp-clean-split",
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "explainability-processing"},
        ],
    )

    step = ProcessingStep(
        name="CleanAndSplit",
        processor=sklearn_processor,
        code=PROCESSING_SCRIPT,
        inputs=[
            ProcessingInput(
                source=S3_GOLD_SET_INPUT,
                destination="/opt/ml/processing/input/gold",
                input_name="gold",
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
        job_arguments=[
            "--train-ratio", "0.8",
            "--val-ratio", "0.1",
            "--seed", "42",
        ],
    )

    return step


# ==============================================================================
# STEP 2: QLoRA Fine-Tuning + Save Adapter (HuggingFace Training, GPU)
# ==============================================================================

def create_step_training(params, role, session, step_clean_split):
    """
    Training Job: Fine-tuning Llama 3.1 8B con QLoRA.
    Guarda solo adapter weights + code/inference.py en model.tar.gz.
    El model.tar.gz es directamente desplegable con PyTorchModel.
    """
    huggingface_estimator = HuggingFace(
        entry_point="finetune_llama_qlora.py",
        source_dir=TRAINING_SOURCE_DIR,
        role=role,
        instance_type=TRAINING_INSTANCE,
        instance_count=1,
        transformers_version="4.36.0",
        pytorch_version="2.1.0",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-exp-qlora-train",
        output_path=S3_TRAINING_OUTPUT,
        hyperparameters={
            "model-id": params["model_id"],
            "hf-token": params["hf_token"],
            "epochs": params["epochs"],
            "batch-size": params["batch_size"],
            "gradient-accumulation-steps": params["grad_accum"],
            "learning-rate": params["learning_rate"],
            "lora-r": params["lora_r"],
            "lora-alpha": params["lora_alpha"],
            "lora-dropout": "0.05",
            "seed": "42",
        },
        environment={
            "HF_TOKEN": params["hf_token"],
            "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        },
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "explainability-training"},
        ],
    )

    step = TrainingStep(
        name="QLoRAFineTuning",
        estimator=huggingface_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=S3_SPLITS_OUTPUT,
                content_type="application/jsonlines",
            ),
        },
    )

    step.add_depends_on([step_clean_split])
    return step, huggingface_estimator


# ==============================================================================
# STEP 3: Evaluation (PyTorch Processing Job, GPU)
# ==============================================================================

def create_step_evaluation(params, role, session, step_training):
    """
    Processing Job (GPU): Evalua el modelo fine-tuned.
    Carga base model BF16 + adapter LoRA en memoria (sin merge).
    Calcula: Exact Match, Keyword Overlap, ROUGE-L F1.
    """
    pytorch_processor = PyTorchProcessor(
        role=role,
        instance_type=EVALUATION_INSTANCE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name="hymmrec-exp-evaluation",
        env={
            "HF_TOKEN": params["hf_token"],
        },
        tags=[
            {"Key": "project", "Value": "hymmrec"},
            {"Key": "phase", "Value": "explainability-evaluation"},
        ],
    )

    # PropertyFile para leer metricas del output (ConditionStep)
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="metrics",
        path="evaluation_report.json",
    )

    step = ProcessingStep(
        name="EvaluateExplainability",
        processor=pytorch_processor,
        code=EVALUATION_SCRIPT,
        inputs=[
            ProcessingInput(
                source=step_training.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
                input_name="model",
            ),
            ProcessingInput(
                source=S3_SPLITS_OUTPUT,
                destination="/opt/ml/processing/input/test",
                input_name="test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/metrics",
                destination=S3_EVAL_OUTPUT,
                output_name="metrics",
            ),
        ],
        job_arguments=[
            "--model-dir", "/opt/ml/processing/input/model",
            "--test-dir", "/opt/ml/processing/input/test",
            "--output-dir", "/opt/ml/processing/output/metrics",
        ],
        property_files=[evaluation_report],
    )

    return step, evaluation_report


# ==============================================================================
# STEP 4: Quality Gate (ConditionStep)
# ==============================================================================

def create_step_quality_gate(params, step_eval, evaluation_report, step_register, step_fail):
    """
    Condition Step: Si metricas superan umbrales -> RegisterModel, si no -> Fail.
    Condiciones: keyword_overlap >= threshold AND rouge_l_f1 >= threshold.
    """
    cond_ko = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="keyword_overlap",
        ),
        right=params["threshold_ko"],
    )

    cond_rouge = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="rouge_l_f1",
        ),
        right=params["threshold_rouge"],
    )

    step = ConditionStep(
        name="QualityGateCheck",
        conditions=[cond_ko, cond_rouge],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return step


# ==============================================================================
# STEP 5: Register Model (Model Registry)
# ==============================================================================

def create_step_register(params, role, session, step_training, huggingface_estimator):
    """
    Registra el modelo aprobado en SageMaker Model Registry.
    El artefacto contiene adapter + code/inference.py (deploy con PyTorchModel).
    """
    step = RegisterModel(
        name="RegisterExplainabilityModel",
        estimator=huggingface_estimator,
        model_data=step_training.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.g5.xlarge", "ml.g5.2xlarge"],
        transform_instances=["ml.g5.xlarge", "ml.g5.2xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="Approved",
        image_uri=PYTORCH_INFERENCE_IMAGE,
    )

    return step


# ==============================================================================
# FAIL STEP
# ==============================================================================

def create_step_fail():
    """Step que marca el pipeline como fallido si no pasa el quality gate."""
    return FailStep(
        name="QualityGateFailed",
        error_message=(
            "Model did not pass quality gate. "
            "Keyword Overlap or ROUGE-L below threshold. "
            "Review evaluation_report.json for details."
        ),
    )


# ==============================================================================
# PIPELINE ASSEMBLY
# ==============================================================================

def create_pipeline(role=None, session=None):
    """
    Crea el SageMaker Pipeline completo.

    Args:
        role: SageMaker execution role ARN
        session: SageMaker Session

    Returns:
        Pipeline object
    """
    if session is None:
        session = sagemaker.Session()
    if role is None:
        role = get_execution_role()

    # Parametros
    params = define_pipeline_parameters()

    # Steps
    step_clean_split = create_step_clean_split(params, role, session)
    step_training, hf_estimator = create_step_training(params, role, session, step_clean_split)
    step_eval, eval_report = create_step_evaluation(params, role, session, step_training)
    step_register = create_step_register(params, role, session, step_training, hf_estimator)
    step_fail = create_step_fail()
    step_condition = create_step_quality_gate(params, step_eval, eval_report, step_register, step_fail)

    # Pipeline
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=list(params.values()),
        steps=[
            step_clean_split,
            step_training,
            step_eval,
            step_condition,
        ],
        sagemaker_session=session,
    )

    return pipeline


# ==============================================================================
# MAIN — CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HYMM-REC Explainability SageMaker Pipeline Management"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--definition",
        action="store_true",
        help="Genera el JSON de definicion del pipeline (para Terraform)",
    )
    group.add_argument(
        "--upsert",
        action="store_true",
        help="Crea o actualiza el pipeline directamente en SageMaker",
    )
    group.add_argument(
        "--execute",
        action="store_true",
        help="Ejecuta el pipeline (debe existir previamente)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path del archivo JSON de salida (solo con --definition). "
             "Default: ../sm-dag-pipelines/hymmrec-explainability-sm-pipeline-dev.json",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="HuggingFace token (para --execute)",
    )
    parser.add_argument(
        "--role-arn",
        type=str,
        default=None,
        help="SageMaker execution role ARN (override)",
    )

    args = parser.parse_args()

    # Setup
    session = sagemaker.Session()
    if args.role_arn:
        role = args.role_arn
    else:
        role = get_execution_role()

    if args.definition:
        # --definition: Genera el JSON para Terraform
        logger.info("Generando pipeline definition JSON...")
        pipeline = create_pipeline(role, session)

        definition_json = pipeline.definition()

        # Determinar output path
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                SCRIPT_DIR, "..", "sm-dag-pipelines",
                "hymmrec-explainability-sm-pipeline-dev.json"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(definition_json)

        logger.info(f"Pipeline definition guardado en: {output_path}")
        logger.info(f"Tamano: {os.path.getsize(output_path) / 1024:.1f} KB")
        logger.info("Los scripts fueron empaquetados y subidos a S3 por el SDK.")
        logger.info("Usa este JSON con el recurso aws_sagemaker_pipeline de Terraform.")

    elif args.upsert:
        # --upsert: Crea/actualiza el pipeline directamente
        logger.info("Creando/actualizando pipeline en SageMaker...")
        pipeline = create_pipeline(role, session)
        pipeline.upsert(role_arn=role)
        logger.info(f"Pipeline '{PIPELINE_NAME}' creado/actualizado exitosamente.")
        logger.info(f"ARN: {pipeline.describe()['PipelineArn']}")

    elif args.execute:
        # --execute: Ejecuta un pipeline existente
        logger.info(f"Ejecutando pipeline: {PIPELINE_NAME}")

        parameters = {}
        if args.hf_token:
            parameters["HfToken"] = args.hf_token

        sm_client = boto3.client("sagemaker")
        response = sm_client.start_pipeline_execution(
            PipelineName=PIPELINE_NAME,
            PipelineParameters=[
                {"Name": k, "Value": str(v)} for k, v in parameters.items()
            ],
        )
        execution_arn = response["PipelineExecutionArn"]
        logger.info(f"Pipeline execution started: {execution_arn}")


if __name__ == "__main__":
    main()
