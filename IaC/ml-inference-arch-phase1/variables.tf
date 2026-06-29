###############################################################################
# GENERAL PROJECT
###############################################################################
variable "project" {
  type        = string
  description = "Deployment project"
  default     = "hymmrec"
}

variable "provisioner" {
  type        = string
  description = "Infrastructure provisioner"
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
  default     = "tfm"
}

variable "fin_unit" {
  type        = string
  description = "Finance unit"
  default     = "vice_technology"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "domain_id" {
  type    = string
  default = "01"
}

###############################################################################
# NETWORKING (reusa VPC existente del data-arch-phase1)
###############################################################################
variable "vpc_id" {
  type        = string
  description = "VPC ID from data-arch-phase1"
  default     = "vpc-06c6dbcb8d69b1cd0"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs from data-arch-phase1"
  default     = ["subnet-02aded95c1836461e", "subnet-01b39e3bc54ae2bec"]
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["a", "b"]
}

###############################################################################
# SECURITY - KMS
###############################################################################
variable "integration_kms_key_name" {
  description = "KMS key name for integration layer"
  type        = string
  default     = "kms-int-hymm-rec-inf"
}

variable "storage_kms_key_id" {
  description = "KMS key ARN for encryption"
  type        = string
  default     = "arn:aws:kms:us-east-1:697682206292:key/25b8c612-11f5-4f9e-bfa7-9d9fb69ecc64"
}

###############################################################################
# LAMBDA
###############################################################################
variable "lambda_runtime" {
  type    = string
  default = "python3.11"
}

###############################################################################
# EVENT BRIDGE
###############################################################################
variable "event_bridge_scheduler_expression" {
  type    = string
  default = "rate(1 day)"
}

variable "data_orchestrator_trigger_functionality" {
  type    = string
  default = "inf-data-orch-trigger"
}

###############################################################################
# ORCHESTRATION CONTROL LAMBDAS
###############################################################################
variable "register_master_orchestration_execution_functionality" {
  type    = string
  default = "register-inf-orch-execution"
}

variable "update_master_orchestration_status_functionality" {
  type    = string
  default = "update-inf-orch-status"
}

###############################################################################
# DYNAMODB (reusa tabla existente del data-arch-phase1)
###############################################################################
variable "dynamodb_executions_history_table_name" {
  description = "DynamoDB table name for pipeline executions history"
  type        = string
  default     = "pipelines-executions-history"
}

###############################################################################
# S3 BUCKETS (existentes del data-arch-phase1)
###############################################################################
variable "silver_bucket_name" {
  description = "Silver zone bucket name"
  type        = string
  default     = "hymmrec-dilkehousesilver01"
}

variable "gold_bucket_name" {
  description = "Gold zone bucket name"
  type        = string
  default     = "hymmrec-dilkehousegold01"
}

variable "sagemaker_assets_bucket_name" {
  description = "SageMaker assets bucket (models, embeddings, inference I/O)"
  type        = string
  default     = "hymmrec-sagemaker-assets"
}

variable "glue_assets_bucket_name" {
  description = "Glue assets bucket for scripts"
  type        = string
  default     = "hymmrec-glue-assests-bucket"
}

variable "glue_assets_repository_name" {
  description = "Glue assets repository name (prefix in bucket)"
  type        = string
  default     = "glue-repository"
}

variable "glue_network_connection_name" {
  description = "Glue network connection name (created in data-arch-phase1)"
  type        = string
  default     = "glc-hymmrec-dev"
}

variable "glue_security_configuration_name" {
  description = "Glue security configuration name (created in data-arch-phase1)"
  type        = string
  default     = "glc-hymmrec-dev"
}

###############################################################################
# DATA DOMAINS
###############################################################################
variable "ml_use_case" {
  type        = string
  description = "ML use case path in gold layer"
  default     = "ml_feature_store"
}

variable "movie_domain" {
  type        = string
  description = "Movie data domain in silver"
  default     = "obt_movie_affinity"
}

variable "silver_catalog_database_name" {
  type        = string
  description = "Glue Catalog database name for Silver layer (Iceberg tables)"
  default     = "hymmrec_tfm_obt_movie_affinity_silver"
}

variable "gold_catalog_database_name" {
  type        = string
  description = "Glue Catalog database name for Gold layer (ML Feature Store)"
  default     = "hymmrec_tfm_ml_feature_store_gold"
}

variable "recommendations_catalog_database_name" {
  type        = string
  description = "Glue Catalog database name for Gold ML Recommendations (top-K + explainability)"
  default     = "hymmrec_tfm_ml_recommendations_gold"
}

###############################################################################
# SAGEMAKER - Item Tower Model (creado en ml-arch-cd-phase2)
###############################################################################
variable "item_tower_model_name" {
  description = "Nombre del modelo SageMaker del Item Tower (creado en ml-arch-cd-phase2)"
  type        = string
  default     = "hymmrec-item-tower-sm-model-dev"
}

###############################################################################
# OPENSEARCH SERVERLESS (desplegado en proyecto separado ml-inference-vectorsearch-aoss)
###############################################################################
variable "opensearch_endpoint" {
  description = "OpenSearch Serverless collection endpoint (output del proyecto vectorsearch-aoss)"
  type        = string
  default     = "f3cprbwt2yb9t7xvm84a.aoss.us-east-1.on.aws"
}

variable "opensearch_index_name" {
  description = "OpenSearch index name for item embeddings"
  type        = string
  default     = "hymmrec-items-vectors"
}

###############################################################################
# SAGEMAKER ENDPOINTS (creados en ml-arch-cd-phase2)
###############################################################################
variable "user_tower_endpoint_name" {
  description = "SageMaker endpoint name for User Tower"
  type        = string
  default     = "hymmrec-user-tower-sgmkr-ep"
}

variable "full_model_endpoint_name" {
  description = "SageMaker endpoint name for Full Model (Two-Heads)"
  type        = string
  default     = "hymmrec-full-model-sgmkr-ep"
}
