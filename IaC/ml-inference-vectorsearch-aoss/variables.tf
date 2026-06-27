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
# OPENSEARCH SERVERLESS — Collection & Collection Group
###############################################################################
variable "vectorsearch_collection_group_name" {
  description = "Nombre del Collection Group NextGen (se prefija con project)"
  type        = string
  default     = "vectors-cg"
}

variable "vectorsearch_collection_name" {
  description = "Nombre de la Collection VECTORSEARCH (se prefija con project)"
  type        = string
  default     = "items-vectors"
}

variable "collection_group_standby_replicas" {
  description = "Standby replicas para el collection group (ENABLED o DISABLED)"
  type        = string
  default     = "ENABLED"
}

variable "max_indexing_ocu" {
  description = "Máximo de OCUs para indexación"
  type        = number
  default     = 2
}

variable "max_search_ocu" {
  description = "Máximo de OCUs para búsqueda"
  type        = number
  default     = 2
}

###############################################################################
# IAM — Roles externos que necesitan acceso a AOSS
###############################################################################
variable "sagemaker_notebook_role_name" {
  description = "Nombre del IAM role de SageMaker Notebook (creado en data-arch-phase1)"
  type        = string
  default     = "sgmkr-notebook-tfm-hymm-rec-ml-iar-dev"
}

