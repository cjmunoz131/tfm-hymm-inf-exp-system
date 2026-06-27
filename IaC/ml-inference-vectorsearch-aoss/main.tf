data "aws_caller_identity" "current" {
  provider = aws.account1
}
data "aws_partition" "current" {
  provider = aws.account1
}
data "aws_region" "current" {
  provider = aws.account1
}

locals {
  partition  = data.aws_partition.current.partition
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.region

  # ARNs de principals que necesitan acceso a OpenSearch Serverless
  # Glue Job (items_opensearch_indexing) — role creado en ml-inference-arch-phase1
  glue_indexing_role_arn = "arn:aws:iam::${local.account_id}:role/role-gl-${var.project}-items-os-idx-glj-${terraform.workspace}"

  # SageMaker Notebook Instance role — creado en data-arch-phase1
  sagemaker_notebook_role_arn = "arn:aws:iam::${local.account_id}:role/${var.sagemaker_notebook_role_name}"

  # SSO Admin role (para pruebas manuales vía dashboard/API)
  sso_admin_role_arn = "arn:aws:iam::${local.account_id}:role/aws-reserved/sso.amazonaws.com/us-east-2/AWSReservedSSO_AdministratorAccess_cabda561aaa68976"
}

################################################################################
# OpenSearch Serverless — VECTORSEARCH Collection (NextGen, scale-to-zero)
################################################################################

module "aws_database_search_vectordb_os_serverless_layer_module" {
  source = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-database-nosql-docs-searchanalytics-aoss"

  providers = {
    aws.main = aws.account1
    aws.dns  = aws.dns
  }

  create  = true
  region  = local.region
  project = var.project

  # Collection Group (NextGen + scale-to-zero)
  create_collection_group             = true
  collection_group_name               = "${var.project}-${var.vectorsearch_collection_group_name}"
  collection_group_generation         = "NEXTGEN"
  collection_group_standby_replicas   = var.collection_group_standby_replicas
  collection_group_capacity_limits = {
    min_indexing_capacity_in_ocu = 0
    max_indexing_capacity_in_ocu = var.max_indexing_ocu
    min_search_capacity_in_ocu  = 0
    max_search_capacity_in_ocu  = var.max_search_ocu
  }

  # Collection
  name             = "${var.project}-${var.vectorsearch_collection_name}"
  type             = "VECTORSEARCH"
  standby_replicas = var.collection_group_standby_replicas

  # Encryption (AWS managed key — sin costo KMS adicional)
  create_encryption_policy = true

  # Network Policy (público para TFM — simplifica acceso desde Glue y notebook)
  create_network_policy = true
  enable_vpce_access    = false

  # Access Policy — principals que pueden indexar y buscar
  create_access_policy     = true
  access_policy_principals = [
    local.glue_indexing_role_arn,
    local.sagemaker_notebook_role_arn,
    local.sso_admin_role_arn
  ]
}
