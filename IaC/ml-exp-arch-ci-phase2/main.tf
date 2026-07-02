# ==============================================================================
# HYMM-REC Explainability CI Phase 2: Fine-Tuning Pipeline (Llama 3.1 8B QLoRA)
# ==============================================================================
# Pipeline de CI para el modelo de explicabilidad:
#   1. Processing Job: Clean Gold Set + Split (train 80% / val 10% / test 10%)
#   2. Training Job: Fine-tuning Llama 3.1 8B con QLoRA (SFTTrainer)
#   3. Processing Job: Evaluation (ROUGE-L, Exact Match, Semantic Similarity)
#   4. Model Registry: Registrar modelo si métricas superan umbral
#
# Orquestación: SageMaker Notebook Instance invoca cada paso secuencialmente
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
# SAGEMAKER PIPELINE SCRIPTS — Upload dev/ scripts to S3
# ==============================================================================
# Terraform uploads processing, training, and evaluation scripts to S3
# so the SageMaker Notebook can reference them at runtime.
# Structure in S3:
#   s3://hymmrec-sagemaker-assets/hymmrec/explainability/scripts/
#     ├── processing/   (clean + split script)
#     ├── training/     (QLoRA fine-tuning script)
#     └── evaluation/   (evaluation script)
# ==============================================================================

locals {
  sagemaker_scripts_bucket = var.sagemaker_scripts_bucket
  scripts_s3_prefix        = "hymmrec/explainability/scripts"

  # Collect all .py files from dev/ subdirectories
  processing_scripts = fileset("${path.module}/dev/processing", "*.py")
  training_scripts   = fileset("${path.module}/dev/training", "*.py")
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
