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
  region     = data.aws_region.current.name
}

###############################################################################
# NETWORKING (reusa VPC existente del data-arch-phase1 via data sources)
###############################################################################
data "aws_vpc" "existing" {
  provider = aws.account1
  id       = var.vpc_id
}

data "aws_subnets" "private" {
  provider = aws.account1
  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }
  filter {
    name   = "subnet-id"
    values = var.private_subnet_ids
  }
}

###############################################################################
# SECURITY - KMS
###############################################################################
module "aws_security_keys_integration_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source   = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-security-keys-kms"
  key_name = var.integration_kms_key_name
}

###############################################################################
# SECURITY GROUPS
###############################################################################
resource "aws_security_group" "sg_lambda" {
  provider    = aws.account1
  name        = "${var.project}-inf-lambda-sg-${terraform.workspace}"
  description = "Security group for inference pipeline lambdas"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "sg_glue" {
  provider    = aws.account1
  name        = "${var.project}-inf-glue-sg-${terraform.workspace}"
  description = "Security group for inference pipeline Glue jobs"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }
}

###############################################################################
# SSM PARAMETER STORE - CONFIGS
###############################################################################
resource "aws_ssm_parameter" "hymmrec_inference_execution_plan_tmpl" {
  provider = aws.account1
  name     = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}/inference-execution-plan-manifest.json"
  type     = "String"
  value    = file("${path.root}/config/inference-execution-plan-manifest.json")
}

resource "aws_ssm_parameter" "hymmrec_domain_context" {
  provider = aws.account1
  name     = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}/domain-context.json"
  type     = "String"
  value    = file("${path.root}/config/domain-context.json")
}

###############################################################################
# DYNAMODB - Executions History (reusa la tabla existente via data source)
###############################################################################
data "aws_dynamodb_table" "executions_history" {
  provider = aws.account1
  name     = "${var.project}-${var.dynamodb_executions_history_table_name}"
}

###############################################################################
# EVENT BRIDGE SCHEDULER (triggers inference pipeline daily)
###############################################################################
module "aws_integration_event_bus_event_bridge_scheduler_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source              = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-integration-event-bus-event_bridge"
  name                = "${var.project}-inf-ebr-scheduler-${terraform.workspace}"
  description         = "Eventbridge scheduler to trigger the inference pipeline"
  rule_type           = "schedule"
  project             = var.project
  schedule_expression = var.event_bridge_scheduler_expression
  state               = "ENABLED"
  targets = [{
    target_id     = "lambda-${var.project}-inf-${var.data_orchestrator_trigger_functionality}-${terraform.workspace}"
    arn           = module.aws_app_compute_lambda_inference_orchestrator_trigger_layer_module.lambda_arn
    required_role = true
  }]
  statements_policy = [{
    Effect   = "Allow"
    Action   = ["lambda:InvokeFunction"]
    Resource = [module.aws_app_compute_lambda_inference_orchestrator_trigger_layer_module.lambda_arn]
  }]
}

###############################################################################
# LAMBDA: Inference Orchestrator Trigger
###############################################################################
module "aws_app_compute_lambda_inference_orchestrator_trigger_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                    = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-app-compute-lambda"
  subnets_ids               = var.private_subnet_ids
  security_group_ids        = [aws_security_group.sg_lambda.id]
  vpc_attach                = true
  lambda_name               = "${var.project}-inf-${var.data_orchestrator_trigger_functionality}"
  lambda_script             = ""
  lambda_runtime            = var.lambda_runtime
  description               = "Inference pipeline orchestrator trigger for ${var.project}"
  source_code_path          = "./${path.root}/dev/lambdas"
  output_zip_path           = "./${path.root}/dev/artefacts/lambdas"
  create_layers             = false
  lambda_layers_definitions = {}
  lambda_layers             = null
  project                   = var.project
  use_existing_role         = false
  add_custom_policy         = true
  custom_policy_path        = "${path.root}/extra-policies/lambda"
  parameters_custom_policy_map = {
    region                  = local.region
    account_id              = local.account_id
    ssm_parameter_prefix    = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}"
    sfn_master_pipeline_arn = module.aws_integration_workflow_inference_pipeline_step_function_layer_module.state_machine_arn
  }
  environment_variables = {
    EXECUTION_PLAN_MANIFEST  = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}/inference-execution-plan-manifest.json"
    DOMAIN_CONTEXT_PARAMETER = "/${var.project}/${terraform.workspace}/${var.data_orchestrator_trigger_functionality}/domain-context.json"
    STATE_MACHINE_ARN        = module.aws_integration_workflow_inference_pipeline_step_function_layer_module.state_machine_arn
    REGION                   = local.region
    ID_DOMAIN_PARAMETER      = var.domain_id
    INFERENCE_PIPELINE_ID    = "INFERENCE-001"
  }
}

###############################################################################
# LAMBDA: Register Inference Orchestration Execution
###############################################################################
module "aws_app_compute_lambda_register_inference_orchestration_execution_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source             = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-app-compute-lambda"
  subnets_ids        = var.private_subnet_ids
  security_group_ids = [aws_security_group.sg_lambda.id]
  vpc_attach         = true
  lambda_name        = "${var.project}-${var.register_master_orchestration_execution_functionality}"
  lambda_script      = ""
  lambda_runtime     = var.lambda_runtime
  description        = "Register inference orchestration execution for ${var.project}"
  source_code_path   = "./${path.root}/dev/lambdas"
  output_zip_path    = "./${path.root}/dev/artefacts/lambdas"
  lambda_layers      = null
  project            = var.project
  use_existing_role  = false
  add_custom_policy  = true
  custom_policy_path = "${path.root}/extra-policies/lambda"
  parameters_custom_policy_map = {
    region               = local.region
    account_id           = local.account_id
    kms_database_key_arn = var.storage_kms_key_id
    dynamodb_table_arn   = data.aws_dynamodb_table.executions_history.arn
    ssm_parameter_prefix = "/${var.project}/${terraform.workspace}/${var.register_master_orchestration_execution_functionality}"
  }
  environment_variables = {
    DDB_TABLE_NAME = data.aws_dynamodb_table.executions_history.name
  }
}

###############################################################################
# LAMBDA: Update Inference Orchestration Status
###############################################################################
module "aws_app_compute_lambda_update_inference_orchestration_status_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source             = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-app-compute-lambda"
  subnets_ids        = var.private_subnet_ids
  security_group_ids = [aws_security_group.sg_lambda.id]
  vpc_attach         = true
  lambda_name        = "${var.project}-${var.update_master_orchestration_status_functionality}"
  lambda_script      = ""
  lambda_runtime     = var.lambda_runtime
  description        = "Update inference orchestration status for ${var.project}"
  source_code_path   = "./${path.root}/dev/lambdas"
  output_zip_path    = "./${path.root}/dev/artefacts/lambdas"
  lambda_layers      = null
  project            = var.project
  use_existing_role  = false
  add_custom_policy  = true
  custom_policy_path = "${path.root}/extra-policies/lambda"
  parameters_custom_policy_map = {
    region               = local.region
    account_id           = local.account_id
    kms_database_key_arn = var.storage_kms_key_id
    dynamodb_table_arn   = data.aws_dynamodb_table.executions_history.arn
    ssm_parameter_prefix = "/${var.project}/${terraform.workspace}/${var.update_master_orchestration_status_functionality}"
  }
  environment_variables = {
    DDB_TABLE_NAME = data.aws_dynamodb_table.executions_history.name
  }
}

###############################################################################
# GLUE SECURITY CONFIGURATION (reusa la existente de data-arch-phase1)
###############################################################################
data "aws_glue_connection" "existing_network_connection" {
  provider = aws.account1
  id       = var.glue_network_connection_name
}

###############################################################################
# GLUE JOB 1: Items Batch Preparation (Python Shell)
# Lee Gold parquet + embeddings_catalog.pkl → genera JSONL para Batch Transform
###############################################################################
module "aws_data_processing_job_glue_items_batch_preparation_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source         = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-processing-job-glue"
  create         = true
  project        = var.project
  auto_scaling   = false
  job_name       = format("%s-%s-%s", var.project, "items-batch-preparation", "glj-inf-${terraform.workspace}")
  job_connections = [var.glue_network_connection_name]
  glue_version   = "4.0"
  timeout        = 2880
  max_capacity   = "1.0"
  max_retries    = 1
  execution_property = {
    max_concurrent_runs = 1
  }
  security_configuration = var.glue_security_configuration_name
  create_role            = true
  role_name              = format("%s-items-prep-glj-%s", var.project, terraform.workspace)
  bucket_deployment      = var.glue_assets_bucket_name
  repository_name        = var.glue_assets_repository_name
  glue_access_databases_tables = {}
  aws_glue_connection_arn      = data.aws_glue_connection.existing_network_connection.arn
  add_iceberg_config           = false
  script_name = "items_batch_preparation_job"
  job_parameters = {
    "--gold_interactions_path"  = "s3://${var.gold_bucket_name}/data/${var.ml_use_case}/interactions/feature_interactions.parquet"
    "--embeddings_catalog_path" = "s3://${var.sagemaker_assets_bucket_name}/${var.project}/model_artefacts/embeddings/embeddings_catalog.pkl"
    "--output_path"             = "s3://${var.gold_bucket_name}/data/${var.ml_use_case}/inference/batch-transform-input/items_for_batch.jsonl"
    "--aws_region"              = local.region
    "--library-set"             = "analytics"
  }
  keys = [
    var.storage_kms_key_id,
    "arn:aws:kms:${local.region}:${local.account_id}:alias/aws/glue"
  ]
  additional_policies = [
    {
      name   = "AllowS3ReadGoldAndSagemakerAssets"
      sid    = "AllowS3ReadGoldAndSagemakerAssets"
      effect = "Allow"
      actions = [
        "s3:GetObject",
        "s3:ListBucket"
      ]
      resources = [
        "arn:aws:s3:::${var.gold_bucket_name}",
        "arn:aws:s3:::${var.gold_bucket_name}/*",
        "arn:aws:s3:::${var.sagemaker_assets_bucket_name}",
        "arn:aws:s3:::${var.sagemaker_assets_bucket_name}/*"
      ]
    },
    {
      name   = "AllowS3WriteBatchTransformInput"
      sid    = "AllowS3WriteBatchTransformInput"
      effect = "Allow"
      actions = [
        "s3:PutObject",
        "s3:DeleteObject"
      ]
      resources = [
        "arn:aws:s3:::${var.gold_bucket_name}/data/${var.ml_use_case}/inference/*"
      ]
    }
  ]
  command = {
    name           = "pythonshell"
    script_path    = "${var.glue_assets_repository_name}/scripts"
    python_version = "3.9"
  }
  main_source_path = "./${path.root}/dev/glue"
  source_buckets   = [var.gold_bucket_name, var.sagemaker_assets_bucket_name]
  destiny_buckets  = [var.gold_bucket_name]
  sources_types    = ["s3"]
}

###############################################################################
# GLUE JOB 2: Items OpenSearch Indexing (Python Shell)
# Lee Batch Transform output + Silver metadata → Bulk index OpenSearch kNN
###############################################################################
module "aws_data_processing_job_glue_items_opensearch_indexing_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source         = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-data-processing-job-glue"
  create         = true
  project        = var.project
  auto_scaling   = false
  job_name       = format("%s-%s-%s", var.project, "items-opensearch-indexing", "glj-inf-${terraform.workspace}")
  job_connections = [var.glue_network_connection_name]
  glue_version   = "4.0"
  timeout        = 2880
  max_capacity   = "1.0"
  max_retries    = 1
  execution_property = {
    max_concurrent_runs = 1
  }
  security_configuration = var.glue_security_configuration_name
  create_role            = true
  role_name              = format("%s-items-os-idx-glj-%s", var.project, terraform.workspace)
  bucket_deployment      = var.glue_assets_bucket_name
  repository_name        = var.glue_assets_repository_name
  glue_access_databases_tables = {}
  aws_glue_connection_arn      = data.aws_glue_connection.existing_network_connection.arn
  add_iceberg_config           = false
  script_name = "items_opensearch_indexing_job"
  job_parameters = {
    "--batch_transform_output_path" = "s3://${var.gold_bucket_name}/data/${var.ml_use_case}/inference/batch-transform-output/"
    "--silver_movies_path"          = "s3://${var.silver_bucket_name}/data/${var.movie_domain}/cleansed_movies/"
    "--encoders_path"               = "s3://${var.sagemaker_assets_bucket_name}/${var.project}/model_artefacts/encoders/encoders.pkl"
    "--opensearch_endpoint"         = var.opensearch_endpoint
    "--opensearch_index_name"       = var.opensearch_index_name
    "--aws_region"                  = local.region
    "--library-set"                 = "analytics"
  }
  keys = [
    var.storage_kms_key_id,
    "arn:aws:kms:${local.region}:${local.account_id}:alias/aws/glue"
  ]
  additional_policies = [
    {
      name   = "AllowS3ReadBatchOutputAndSilver"
      sid    = "AllowS3ReadBatchOutputAndSilver"
      effect = "Allow"
      actions = [
        "s3:GetObject",
        "s3:ListBucket"
      ]
      resources = [
        "arn:aws:s3:::${var.gold_bucket_name}",
        "arn:aws:s3:::${var.gold_bucket_name}/*",
        "arn:aws:s3:::${var.sagemaker_assets_bucket_name}",
        "arn:aws:s3:::${var.sagemaker_assets_bucket_name}/*",
        "arn:aws:s3:::${var.silver_bucket_name}",
        "arn:aws:s3:::${var.silver_bucket_name}/*"
      ]
    },
    {
      name   = "AllowOpenSearchServerlessAccess"
      sid    = "AllowOpenSearchServerlessAccess"
      effect = "Allow"
      actions = [
        "aoss:APIAccessAll"
      ]
      resources = [
        "arn:aws:aoss:${local.region}:${local.account_id}:collection/*"
      ]
    }
  ]
  command = {
    name           = "pythonshell"
    script_path    = "${var.glue_assets_repository_name}/scripts"
    python_version = "3.9"
  }
  main_source_path = "./${path.root}/dev/glue"
  source_buckets   = [var.gold_bucket_name, var.sagemaker_assets_bucket_name, var.silver_bucket_name]
  destiny_buckets  = []
  sources_types    = ["s3"]
}

###############################################################################
# SAGEMAKER MODEL (Item Tower - creado en ml-arch-cd-phase2, referenciado aquí)
###############################################################################
data "aws_iam_role" "sagemaker_endpoint_role" {
  provider = aws.account1
  name     = "${var.project}-sm-endpoint-iar-${terraform.workspace}"
}


###############################################################################
# STEP FUNCTION: Inference Pipeline Orchestration
###############################################################################
module "aws_integration_workflow_inference_pipeline_step_function_layer_module" {
  providers = {
    aws.main = aws.account1
  }
  source                 = "git@github.com:cjmunoz131/terraform_modules//modules/aws/aws-integration-workflow-process-step-function"
  create                 = true
  sfn_publish            = true
  source_definition_path = "${path.root}/state-machine-asls"
  sfn_state_machine_name = "sfn-${var.project}-inference-data-pipeline-${terraform.workspace}"
  type                   = "STANDARD"
  vars_map = {
    registerInferenceOrchestrationARN = module.aws_app_compute_lambda_register_inference_orchestration_execution_layer_module.lambda_arn
    updateInferenceOrchStatusARN      = module.aws_app_compute_lambda_update_inference_orchestration_status_layer_module.lambda_arn
    items_batch_preparation_glj       = module.aws_data_processing_job_glue_items_batch_preparation_layer_module.job_name
    items_opensearch_indexing_glj     = module.aws_data_processing_job_glue_items_opensearch_indexing_layer_module.job_name
    item_tower_model_name             = var.item_tower_model_name
    batch_transform_input_s3_uri      = "s3://${var.gold_bucket_name}/data/${var.ml_use_case}/inference/batch-transform-input/"
    batch_transform_output_s3_uri     = "s3://${var.gold_bucket_name}/data/${var.ml_use_case}/inference/batch-transform-output/"
  }
  tracing_enabled    = true
  custom_policy_path = "${path.root}/extra-policies/step-function"
  create_role        = true
  create_terraform_style = false
  logging_configuration = {
    level                  = "ALL"
    include_execution_data = true
  }
  parameters_custom_policy_map = {
    registerInferenceOrchestrationARN = module.aws_app_compute_lambda_register_inference_orchestration_execution_layer_module.lambda_arn
    updateInferenceOrchStatusARN      = module.aws_app_compute_lambda_update_inference_orchestration_status_layer_module.lambda_arn
    items_batch_preparation_job_ARN   = module.aws_data_processing_job_glue_items_batch_preparation_layer_module.job_arn
    items_opensearch_indexing_job_ARN = module.aws_data_processing_job_glue_items_opensearch_indexing_layer_module.job_arn
    sagemaker_bt_role_arn             = data.aws_iam_role.sagemaker_endpoint_role.arn
    region                            = local.region
    account_id                        = local.account_id
    sagemaker_assets_bucket           = var.sagemaker_assets_bucket_name
    gold_bucket                       = var.gold_bucket_name
  }
  cloudwatch_log_group_name              = "${var.project}-inference-pipeline-${terraform.workspace}-SMLG"
  cloudwatch_log_group_retention_in_days = 7
  cloudwatch_log_group_kms_key_id        = module.aws_security_keys_integration_layer_module.kms_key_arn
  role_name                              = "${var.project}-inference-pipeline-sfn-iar-${terraform.workspace}"
}
