# ==============================================================================
# IAM ROLE — Explainability Endpoint Execution Role
# ==============================================================================
# Role que asume el contenedor del endpoint para:
#   - Descargar model.tar.gz (adapter LoRA) desde S3
#   - Descargar modelo base de HuggingFace Hub (requiere acceso a internet)
#   - Desencriptar con KMS
#   - Pull de imagen ECR (HuggingFace TGI inference)
#   - Escribir logs a CloudWatch

resource "aws_iam_role" "sagemaker_explainability_endpoint_role" {
  provider = aws.account1
  name     = "${var.project}-sm-exp-endpoint-iar-${terraform.workspace}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_explainability_endpoint_policy" {
  provider = aws.account1
  name     = "${var.project}-sm-exp-endpoint-policy-${terraform.workspace}"
  role     = aws_iam_role.sagemaker_explainability_endpoint_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ModelAndDataAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.sagemaker_assets_bucket}",
          "arn:aws:s3:::${var.sagemaker_assets_bucket}/*",
          "arn:aws:s3:::${var.gold_bucket}",
          "arn:aws:s3:::${var.gold_bucket}/*"
        ]
      },
      {
        Sid    = "KMSDecryptEncrypt"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:Encrypt",
          "kms:DescribeKey",
          "kms:GenerateDataKey*",
          "kms:CreateGrant"
        ]
        Resource = [var.storage_kms_key_id]
      },
      {
        Sid    = "ECRPullInferenceImage"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/Endpoints/*",
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/TransformJobs/*",
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/ProcessingJobs/*"
        ]
      },
      {
        Sid    = "CloudWatchMetrics"
        Effect = "Allow"
        Action = ["cloudwatch:PutMetricData"]
        Resource = "*"
      }
    ]
  })
}
