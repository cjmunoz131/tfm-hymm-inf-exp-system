# ==============================================================================
# HYMM-REC Explainability CD Phase 2: Model Deployment
# ==============================================================================
# Despliega el modelo de explicabilidad (Llama 3.1 8B QLoRA) como:
#   - SageMaker Model (from Model Registry / S3 artifact)
#   - SageMaker Endpoint (GPU ml.g5.xlarge para inferencia)
#
# El modelo genera 3 keywords de explicación dado:
#   - Perfil del usuario (géneros favoritos, historial)
#   - Película recomendada (título, géneros, sinopsis)
#
# Flujo:
#   Model Package (Approved) → aws_sagemaker_model → Endpoint Config → Endpoint
#
# Alternativa: Para invocación batch desde el inference pipeline,
# el artefacto se consume directamente desde S3 sin endpoint real-time.
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
# EXPLAINABILITY MODEL — SageMaker Model
# ==============================================================================
# Modelo Llama 3.1 8B fine-tuned con QLoRA (adapter weights + base model reference).
# El container de inferencia carga el base model + aplica el adapter en runtime.
# Input: JSON {"instruction": ..., "input": ...}
# Output: JSON {"keywords": "keyword1, keyword2, keyword3"}

module "aws_ml_compute_model_serving_explainability_model_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-compute-model-serving-deployment-sagemaker"

  enable_sagemaker_model             = true
  project                            = var.project
  sagemaker_model_name               = var.explainability_model_sagemaker_name
  sagemaker_model_execution_role_arn = aws_iam_role.sagemaker_explainability_endpoint_role.arn
  sagemaker_model_enable_network_isolation = false

  sagemaker_model_primary_container = [
    {
      image          = var.huggingface_inference_image
      model_data_url = "s3://${var.sagemaker_assets_bucket}/hymmrec/explainability/model-artifacts/model.tar.gz"
    }
  ]

  sagemaker_model_container  = []
  sagemaker_model_vpc_config = []
}

# ==============================================================================
# EXPLAINABILITY MODEL — Endpoint Configuration + Endpoint
# ==============================================================================
# Endpoint real-time para inferencia de explicabilidad.
# Instancia: ml.g5.xlarge (24GB A10G) — suficiente para modelo 8B cuantizado.
#
# NOTA: Comentado por defecto para evitar costos en desarrollo.
# Descomentar cuando se necesite el endpoint activo.
# Para uso batch, el modelo se invoca directamente desde el inference pipeline
# usando el artefacto en S3 (sin endpoint).

# module "aws_sagemaker_gov_model_serving_explainability_endpoint_layer_module" {
#   providers = {
#     aws.main = aws.account1
#   }
#   source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-ml-governance-model-serving-endpoint-sagemaker"
#
#   enable_sagemaker_endpoint = true
#   project                   = var.project
#   endpoint-name             = var.explainability_endpoint_name
#
#   enable_sagemaker_endpoint_configuration = true
#   endpoint_configuration_name             = var.explainability_endpoint_config_name
#   sagemaker_endpoint_configuration_kms_key_arn = var.storage_kms_key_id
#   sagemaker_endpoint_configuration_production_variants = [
#     {
#       variant_name           = "AllTraffic"
#       model_name             = module.aws_ml_compute_model_serving_explainability_model_layer_module.sagemaker_model_id
#       initial_instance_count = 1
#       instance_type          = var.endpoint_instance_type
#     }
#   ]
#
#   # Autoscaling (scale-to-zero no disponible para GPU endpoints)
#   enable_sagemaker_default_autoscaling = true
#   endpoint_instance_min_capacity       = 1
#   endpoint_instance_max_capacity       = 2
#   scale_in_cooldown                    = 600
#   scale_out_cooldown                   = 120
#   sagemaker_variant_name               = "AllTraffic"
#   invocations_target_value             = 50
# }
