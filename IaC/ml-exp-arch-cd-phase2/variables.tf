variable "project" {
  type        = string
  description = "Deployment project"
  default     = "hymmrec"
}

variable "provisioner" {
  type        = string
  description = "Infraestructure provisioner"
  default     = "Terraform"
}

variable "owner" {
  type        = string
  description = "Project Owner"
  default     = "cjmunoz"
}

variable "org_unit" {
  type        = string
  description = "Organizational unit"
  default     = "products_crew"
}

variable "fin_unit" {
  type        = string
  description = "finance unit"
  default     = "vice_technology"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

########################

variable "storage_kms_key_id" {
  description = "KMS key ARN for encryption at rest"
  type        = string
  default     = "arn:aws:kms:us-east-1:697682206292:key/25b8c612-11f5-4f9e-bfa7-9d9fb69ecc64"
}

variable "sagemaker_assets_bucket" {
  description = "S3 bucket with model artifacts (Platinum)"
  type        = string
  default     = "hymmrec-sagemaker-assets"
}

variable "gold_bucket" {
  description = "S3 bucket Gold zone"
  type        = string
  default     = "hymmrec-dilkehousegold01"
}

# ==============================================================================
# Explainability Model — Endpoint Configuration
# ==============================================================================

variable "explainability_model_sagemaker_name" {
  description = "SageMaker model name for explainability LLM"
  type        = string
  default     = "explainability-llama"
}

variable "explainability_endpoint_name" {
  description = "SageMaker endpoint name for explainability model"
  type        = string
  default     = "explainability-llama"
}

variable "explainability_endpoint_config_name" {
  description = "SageMaker endpoint configuration name"
  type        = string
  default     = "explainability-llama"
}

variable "endpoint_instance_type" {
  description = "Instance type for explainability inference endpoint (requires GPU)"
  type        = string
  default     = "ml.g5.xlarge"
}

variable "huggingface_inference_image" {
  description = "HuggingFace TGI inference container image URI (GPU, us-east-1)"
  type        = string
  default     = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
}

variable "model_package_group_name" {
  description = "Model Registry package group name"
  type        = string
  default     = "hymmrec-explainability-llama"
}
