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

variable "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN (notebook instance role)"
  type        = string
  default     = "arn:aws:iam::697682206292:role/sgmkr-notebook-tfm-hymm-rec-ml-iar-dev"
}

variable "sagemaker_execution_role_name" {
  description = "SageMaker execution role name"
  type        = string
  default     = "sgmkr-notebook-tfm-hymm-rec-ml-iar-dev"
}

variable "gold_bucket_name" {
  description = "Gold datalake bucket (explainability silver set source)"
  type        = string
  default     = "hymmrec-dilkehousegold01"
}

variable "sagemaker_scripts_bucket" {
  description = "S3 bucket for SageMaker scripts and model artefacts (Platinum)"
  type        = string
  default     = "hymmrec-sagemaker-assets"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
  default     = "vpc-06c6dbcb8d69b1cd0"
}

variable "private_subnet_id_list" {
  description = "Private subnet IDs"
  type        = set(string)
  default     = ["subnet-02aded95c1836461e", "subnet-01b39e3bc54ae2bec"]
}

# ==============================================================================
# Explainability Model Configuration
# ==============================================================================

variable "explainability_model_id" {
  description = "HuggingFace model ID for the base LLM"
  type        = string
  default     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

variable "explainability_use_case" {
  description = "Use case path prefix in Gold/Platinum buckets"
  type        = string
  default     = "explainability"
}

variable "processing_instance_type" {
  description = "Instance type for processing jobs (clean + split)"
  type        = string
  default     = "ml.m5.large"
}

variable "training_instance_type" {
  description = "Instance type for QLoRA fine-tuning (requires GPU A10G)"
  type        = string
  default     = "ml.g5.2xlarge"
}

variable "endpoint_instance_type" {
  description = "Instance type for endpoint (evaluation temporal + production)"
  type        = string
  default     = "ml.g5.2xlarge"
}

variable "model_package_group_name" {
  description = "SageMaker Model Registry package group name for explainability"
  type        = string
  default     = "hymmrec-explainability-llama"
}
