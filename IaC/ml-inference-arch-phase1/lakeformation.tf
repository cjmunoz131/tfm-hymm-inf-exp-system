# ==============================================================================
# LAKE FORMATION PERMISSIONS — Inference Pipeline
# ==============================================================================
# Permisos para el Glue Job de indexación (items_opensearch_indexing_job)
# que necesita leer de Silver (cleansed_movies) y escribir en Gold (items_consolidated).
# ==============================================================================

# ---------------------------------------------------------------
# GLUE JOB: Items OpenSearch Indexing
# Lee: Silver DB (cleansed_movies)
# Escribe: Gold DB (hymmrec_items_consolidated)
# ---------------------------------------------------------------
resource "aws_lakeformation_permissions" "glue_os_indexing_silver_database" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_items_opensearch_indexing_layer_module.iam_roles

  permissions = ["DESCRIBE"]

  database {
    name = var.silver_catalog_database_name
  }
}

resource "aws_lakeformation_permissions" "glue_os_indexing_silver_tables" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_items_opensearch_indexing_layer_module.iam_roles

  permissions = ["SELECT", "DESCRIBE"]

  table {
    database_name = var.silver_catalog_database_name
    wildcard      = true
  }
}

resource "aws_lakeformation_permissions" "glue_os_indexing_gold_database" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_items_opensearch_indexing_layer_module.iam_roles

  permissions = ["DESCRIBE", "CREATE_TABLE", "ALTER"]

  database {
    name = var.gold_catalog_database_name
  }
}

resource "aws_lakeformation_permissions" "glue_os_indexing_gold_tables" {
  provider   = aws.account1
  principal  = module.aws_data_processing_job_glue_items_opensearch_indexing_layer_module.iam_roles

  permissions = ["ALL"]

  table {
    database_name = var.gold_catalog_database_name
    wildcard      = true
  }
}
