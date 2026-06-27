output "step_function_arn" {
  description = "Inference pipeline Step Function ARN"
  value       = module.aws_integration_workflow_inference_pipeline_step_function_layer_module.state_machine_arn
}

output "item_tower_model_name" {
  description = "SageMaker Item Tower model name for Batch Transform (from ml-arch-cd-phase2)"
  value       = var.item_tower_model_name
}

output "batch_transform_input_path" {
  description = "S3 path for Batch Transform input JSONL"
  value       = "s3://${var.gold_bucket_name}/data/${var.ml_use_case}/inference/batch-transform-input/"
}

output "batch_transform_output_path" {
  description = "S3 path for Batch Transform output"
  value       = "s3://${var.gold_bucket_name}/data/${var.ml_use_case}/inference/batch-transform-output/"
}
