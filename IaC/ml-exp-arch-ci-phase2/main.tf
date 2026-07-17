# ==============================================================================
# HYMM-REC Explainability CI Phase 2: Fine-Tuning Pipeline (Llama 3.1 8B QLoRA)
# ==============================================================================
# Pipeline de CI para el modelo de explicabilidad:
#   1. Processing Job: Clean Gold Set + Split (train 80% / val 10% / test 10%)
#   2. Training Job: Fine-tuning Llama 3.1 8B con QLoRA (SFTTrainer)
#   3. Processing Job: Evaluation (ROUGE-L, Exact Match, Keyword Overlap)
#   4. Quality Gate: Condition basada en umbrales
#   5. Model Registry: Registrar modelo si metricas superan umbral
#
# Orquestacion: SageMaker Pipeline (definido via JSON generado por SDK)
# ==============================================================================

data "aws_caller_identity" "current" {
  provider = aws.account1
}
data "aws_partition" "current" {
  provider = aws.account1
}
data "aws_region" "current" {
  provider = aws.account1
}

# ==============================================================================
# MODEL REGISTRY — Package Group for Explainability Model
# ==============================================================================

module "aws_ml_gov_model_serving_explainability_package_group_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-serving-packagegroup-sagemaker"

  enable_sagemaker_package_group = true
  project                        = var.project
  package_group_name             = var.model_package_group_name
}


# ==============================================================================
# SAGEMAKER PIPELINE — MLOps Explainability CI
# ==============================================================================
# Pipeline definition JSON generado con:
#   python sm-pipelines/define_explainability_pipeline.py --definition
# El JSON se guarda en sm-dag-pipelines/ y Terraform lo despliega.

module "aws-ml-governance-model-ops-explainability-pipelines-sagemaker-layer-module" {
  providers = {
    aws.main = aws.account1
  }
  source                    = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-ops-pipelines-sagemaker"
  project                   = var.project
  enable_sagemaker_pipeline = true
  pipeline-sm-name          = var.sagemaker_pipeline_name
  source_definition_path    = "${path.root}/sm-dag-pipelines"
  vars_map                  = {}
  create_role               = true
  role_name                 = "${var.project}-${var.sagemaker_pipeline_name}-smp-iar-${terraform.workspace}"
  custom_policy_path        = "${path.root}/extra-policies/sagemaker-pipeline"
  create_terraform_style    = false
  parameters_custom_policy_map = {
    region                  = data.aws_region.current.name
    account_id              = data.aws_caller_identity.current.account_id
    project                 = var.project
    pipeline_role_name      = var.sagemaker_execution_role_name
    s3_sagemaker_assets_arn = "arn:aws:s3:::${var.sagemaker_scripts_bucket}"
    s3_datalake_gold_arn    = "arn:aws:s3:::${var.gold_bucket_name}"
    kms_arn                 = var.storage_kms_key_id
  }
}


# ==============================================================================
# SAGEMAKER PIPELINE SCRIPTS — Upload dev/ scripts to S3
# ==============================================================================
# Terraform uploads processing, training, evaluation, and inference scripts to S3.
# These are also packaged by the SDK when generating the pipeline definition,
# but kept here as backup/reference and for manual execution from notebook.
# ==============================================================================

locals {
  sagemaker_scripts_bucket = var.sagemaker_scripts_bucket
  scripts_s3_prefix        = "hymmrec/explainability/scripts"

  # Collect all .py files from dev/ subdirectories
  processing_scripts = fileset("${path.module}/dev/processing", "*.py")
  training_scripts   = fileset("${path.module}/dev/training", "*.py")
  evaluation_scripts = fileset("${path.module}/dev/evaluation", "*.py")
  inference_scripts  = fileset("${path.module}/dev/inference", "**")
}

# --- Processing scripts (clean + split) ---
resource "aws_s3_object" "processing_scripts" {
  provider = aws.account1
  for_each = local.processing_scripts

  bucket       = local.sagemaker_scripts_bucket
  key          = "${local.scripts_s3_prefix}/processing/${each.value}"
  source       = "${path.module}/dev/processing/${each.value}"
  etag         = filemd5("${path.module}/dev/processing/${each.value}")
  content_type = "text/x-python"

  tags = {
    project = var.project
    phase   = "explainability-processing"
  }
}

# --- Training scripts (QLoRA fine-tuning) ---
resource "aws_s3_object" "training_scripts" {
  provider = aws.account1
  for_each = local.training_scripts

  bucket       = local.sagemaker_scripts_bucket
  key          = "${local.scripts_s3_prefix}/training/${each.value}"
  source       = "${path.module}/dev/training/${each.value}"
  etag         = filemd5("${path.module}/dev/training/${each.value}")
  content_type = "text/x-python"

  tags = {
    project = var.project
    phase   = "explainability-training"
  }
}

# --- Evaluation scripts ---
resource "aws_s3_object" "evaluation_scripts" {
  provider = aws.account1
  for_each = local.evaluation_scripts

  bucket       = local.sagemaker_scripts_bucket
  key          = "${local.scripts_s3_prefix}/evaluation/${each.value}"
  source       = "${path.module}/dev/evaluation/${each.value}"
  etag         = filemd5("${path.module}/dev/evaluation/${each.value}")
  content_type = "text/x-python"

  tags = {
    project = var.project
    phase   = "explainability-evaluation"
  }
}

# --- Inference scripts (for endpoint deploy) ---
resource "aws_s3_object" "inference_scripts" {
  provider = aws.account1
  for_each = local.inference_scripts

  bucket       = local.sagemaker_scripts_bucket
  key          = "${local.scripts_s3_prefix}/inference/${each.value}"
  source       = "${path.module}/dev/inference/${each.value}"
  etag         = filemd5("${path.module}/dev/inference/${each.value}")
  content_type = lookup(
    { "py" = "text/x-python", "txt" = "text/plain" },
    reverse(split(".", each.value))[0],
    "application/octet-stream"
  )

  tags = {
    project = var.project
    phase   = "explainability-inference"
  }
}
