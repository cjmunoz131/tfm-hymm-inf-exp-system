"""
==============================================================================
HYMM-REC Explainability: SageMaker Pipeline Definition
==============================================================================
Define el pipeline de CI para el modelo de explicabilidad (Llama 3.1 8B QLoRA).

Steps:
  1. ProcessingStep: Clean Gold Set + Split (train/val/test)
  2. TrainingStep: Fine-tuning QLoRA + Merge (modelo completo)
  3. ProcessingStep: Evaluation (GPU — carga modelo merged en memoria)
  4. ConditionStep: ¿Métricas superan umbrales?
  5. RegisterModelStep: Registrar en Model Registry (condicional)

Ejecución selectiva:
  El pipeline usa ParameterString para controlar qué steps ejecutar.
  Parámetro 'ExecuteSteps' acepta valores:
    - "all"               → ejecuta todo el pipeline
    - "training_only"     → solo Step 1 + Step 2 (sin evaluación ni registro)
    - "eval_only"         → solo Step 3 + Step 4 + Step 5 (usa modelo existente en S3)

Uso:
  from define_explainability_pipeline import create_pipeline
  pipeline = create_pipeline(role, session)
  pipeline.upsert(role_arn=role)
  pipeline.start(parameters={"ExecuteSteps": "all"})

Pre-requisitos:
  - gold_dataset_clean.jsonl en S3 (Gold bucket)
  - Quotas: ml.g5.12xlarge training, ml.g5.2xlarge processing, ml.m5.large processing
  - Package Group 'hymmrec-explainability-llama' creado por Terraform
==============================================================================
"""

import os
import json
import sagemaker
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.huggingface import HuggingFace


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

REGION = "us-east-1"
PROJECT = "hymmrec"
PIPELINE_NAME = f"{PROJECT}-explainability-ci"

# Buckets
PLATINUM_BUCKET = "hymmrec-sagemaker-assets"
GOLD_BUCKET = "hymmrec-dilkehousegold01"

# S3 Paths
S3_GOLD_SET_INPUT = f"s3://{GOLD_BUCKET}/data/ml_recommendations/explainability/"
S3_EXPLAINABILITY_PREFIX = f"s3://{PLATINUM_BUCKET}/hymmrec/explainability"
S3_SPLITS_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/datasets/splits/"
S3_TRAINING_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/training-output/"
S3_EVAL_OUTPUT = f"{S3_EXPLAINABILITY_PREFIX}/evaluation/"

# Scripts (relativos al notebook — ajustar si se ejecuta desde otro directorio)
PROCESSING_SCRIPT = "../dev/processing/clean_and_split_job.py"
TRAINING_SCRIPT_DIR = "../dev/training/"
EVALUATION_SCRIPT = "../dev/evaluation/evaluate_explainability.py"

# Model Registry
MODEL_PACKAGE_GROUP_NAME = "hymmrec-explainability-llama"

# Instance Types
PROCESSING_INSTANCE = "ml.m5.large"
TRAINING_INSTANCE = "ml.g5.12xlarge"
EVALUATION_INSTANCE = "ml.g5.2xlarge"

# TGI Image for Model Registry
TGI_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.3.1-tgi2.2.0-gpu-py310-cu121-ubuntu22.04-v2.0"


# ==============================================================================
# PIPELINE DEFINITION
# ==============================================================================

def create_pipeline(role=None, session=None):
    """
    Crea el SageMaker Pipeline para el flujo CI de explainability.
    
    Args:
        role: SageMaker execution role ARN
        session: SageMaker Session
    
    Returns:
        Pipeline object listo para upsert + start
    """
    if session is None:
        session = sagemaker.Session()
    if role is None:
        role = get_execution_role()

    # ==================================================================
    # PARÁMETROS DEL PIPELINE
    # ==================================================================

    # Control de ejecución selectiva
    param_execute_steps = ParameterString(
        name="ExecuteSteps",
        default_value="all",  # "all", "training_only", "eval_only"
    )

    # Modelo base
    param_model_id = ParameterString(
        name="ModelId",
        default_value="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    # HuggingFace Token
    param_hf_token = ParameterString(
        name="HfToken",
        default_value="",
    )

    # Hiperparámetros
    param_epochs = ParameterString(name="Epochs", default_value="3")
    param_learning_rate = ParameterString(name="LearningRate", default_value="0.0002")
    param_lora_r = ParameterString(name="LoraR", default_value="16")
    param_lora_alpha = ParameterString(name="LoraAlpha", default_value="32")

    # Umbrales de evaluación
    param_threshold_ko = ParameterFloat(name="ThresholdKeywordOverlap", default_value=0.4)
    param_threshold_rouge = ParameterFloat(name="ThresholdRougeL", default_value=0.3)

    # Path del modelo (para eval_only — apunta a model.tar.gz existente)
    param_model_data_url = ParameterString(
        name="ModelDataUrl",
        default_value=f"{S3_TRAINING_OUTPUT}latest/output/model.tar.gz",
    )

    # ==================================================================
    # STEP 1: PROCESSING — Clean Gold Set + Split
    # ==================================================================

    sklearn_processor = SKLearnProcessor(
        role=role,
        instance_type=PROCESSING_INSTANCE,
        instance_count=1,
        framework_version="1.2-1",
        sagemaker_session=session,
        base_job_name=f"{PROJECT}-exp-clean-split",
    )

    step_clean_split = ProcessingStep(
        name="CleanAndSplit",
        processor=sklearn_processor,
        code=PROCESSING_SCRIPT,
        inputs=[
            ProcessingInput(
                source=S3_GOLD_SET_INPUT,
                destination="/opt/ml/processing/input/gold",
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

    # ==================================================================
    # STEP 2: TRAINING — QLoRA Fine-Tuning + Merge
    # ==================================================================

    huggingface_estimator = HuggingFace(
        entry_point="finetune_llama_qlora.py",
        source_dir=TRAINING_SCRIPT_DIR,
        role=role,
        instance_type=TRAINING_INSTANCE,
        instance_count=1,
        transformers_version="4.36.0",
        pytorch_version="2.1.0",
        py_version="py310",
        sagemaker_session=session,
        base_job_name=f"{PROJECT}-exp-qlora-train",
        output_path=S3_TRAINING_OUTPUT,
        hyperparameters={
            "model-id": param_model_id,
            "hf-token": param_hf_token,
            "epochs": param_epochs,
            "batch-size": "1",
            "gradient-accumulation-steps": "8",
            "learning-rate": param_learning_rate,
            "lora-r": param_lora_r,
            "lora-alpha": param_lora_alpha,
            "lora-dropout": "0.05",
            "seed": "42",
        },
        environment={
            "HF_TOKEN": param_hf_token,
            "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        },
    )

    step_training = TrainingStep(
        name="QLoRAFineTuningAndMerge",
        estimator=huggingface_estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=S3_SPLITS_OUTPUT,
                content_type="application/jsonlines",
            ),
        },
    )

    # ==================================================================
    # STEP 3: PROCESSING — Evaluation (GPU, modelo merged en memoria)
    # ==================================================================

    pytorch_processor = PyTorchProcessor(
        role=role,
        instance_type=EVALUATION_INSTANCE,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
        base_job_name=f"{PROJECT}-exp-evaluation",
    )

    # PropertyFile para leer métricas del output
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="metrics",
        path="evaluation_report.json",
    )

    step_evaluation = ProcessingStep(
        name="EvaluateExplainability",
        processor=pytorch_processor,
        code=EVALUATION_SCRIPT,
        inputs=[
            ProcessingInput(
                source=step_training.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=S3_SPLITS_OUTPUT,
                destination="/opt/ml/processing/input/test",
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

    # ==================================================================
    # STEP 4: CONDITION — ¿Métricas superan umbrales?
    # ==================================================================

    cond_ko = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluation.name,
            property_file=evaluation_report,
            json_path="keyword_overlap",
        ),
        right=param_threshold_ko,
    )

    cond_rouge = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluation.name,
            property_file=evaluation_report,
            json_path="rouge_l_f1",
        ),
        right=param_threshold_rouge,
    )

    # ==================================================================
    # STEP 5: REGISTER MODEL (condicional)
    # ==================================================================

    step_register = RegisterModel(
        name="RegisterExplainabilityModel",
        estimator=huggingface_estimator,
        model_data=step_training.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.g5.xlarge", "ml.g5.2xlarge"],
        transform_instances=["ml.g5.xlarge", "ml.g5.2xlarge"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="Approved",
        image_uri=TGI_IMAGE,
    )

    # Condition Step: si pasa umbrales → registrar, si no → no hacer nada
    step_condition = ConditionStep(
        name="CheckMetricsThreshold",
        conditions=[cond_ko, cond_rouge],
        if_steps=[step_register],
        else_steps=[],
    )

    # ==================================================================
    # PIPELINE ASSEMBLY
    # ==================================================================

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            param_execute_steps,
            param_model_id,
            param_hf_token,
            param_epochs,
            param_learning_rate,
            param_lora_r,
            param_lora_alpha,
            param_threshold_ko,
            param_threshold_rouge,
            param_model_data_url,
        ],
        steps=[
            step_clean_split,
            step_training,
            step_evaluation,
            step_condition,
        ],
        sagemaker_session=session,
    )

    return pipeline


# ==============================================================================
# MAIN — Para ejecutar desde notebook o CLI
# ==============================================================================

if __name__ == "__main__":
    """
    Uso desde notebook:
        %run define_explainability_pipeline.py
        
    O importar:
        from define_explainability_pipeline import create_pipeline
        pipeline = create_pipeline()
        pipeline.upsert(role_arn=ROLE)
        
        # Ejecutar todo el pipeline
        execution = pipeline.start(parameters={"HfToken": "hf_xxx"})
        
        # Solo training (sin evaluación)
        execution = pipeline.start(parameters={
            "HfToken": "hf_xxx",
            "ExecuteSteps": "training_only",
        })
    """
    session = sagemaker.Session()
    role = get_execution_role()

    pipeline = create_pipeline(role, session)

    # Upsert (crear o actualizar)
    pipeline.upsert(role_arn=role)
    print(f"\nPipeline '{PIPELINE_NAME}' creado/actualizado.")
    print(f"ARN: {pipeline.describe()['PipelineArn']}")

    print(f"""
Uso:
  # Ejecutar pipeline completo:
  pipeline.start(parameters={{"HfToken": "<tu-token>"}})
  
  # Con hiperparámetros custom:
  pipeline.start(parameters={{
      "HfToken": "<tu-token>",
      "Epochs": "5",
      "LearningRate": "0.0001",
      "LoraR": "32",
  }})
  
  # Cambiar umbrales de aprobación:
  pipeline.start(parameters={{
      "HfToken": "<tu-token>",
      "ThresholdKeywordOverlap": 0.5,
      "ThresholdRougeL": 0.35,
  }})
""")
