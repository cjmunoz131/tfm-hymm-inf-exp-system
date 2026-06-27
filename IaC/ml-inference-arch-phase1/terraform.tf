provider "aws" {
  alias   = "account1"
  profile = "shared-services" #var.deployment_profile
  region  = "us-east-1"       #var.region
  assume_role {
    role_arn = "arn:aws:iam::${lookup(local.account_mapping, local.env)[0]}:role/terraform-role"
  }
  default_tags {
    tags = {
      Customer    = local.commons.customer
      Environment = terraform.workspace
      Org_unit    = local.commons.org_unit
      Provisioner = local.commons.provisioner
      Solution    = local.commons.project
      fin_unit    = local.commons.fin_unit
    }
  }
}

provider "aws" {
  alias   = "dns"
  profile = "shared-services" #var.deployment_profile
  region  = "us-east-1"       #var.region_shared_services
  assume_role {
    role_arn = "arn:aws:iam::${local.account_mapping.shared_services}:role/terraform-role"
  }
  default_tags {
    tags = {
      Customer    = local.commons.customer
      Environment = "shared"
      Org_unit    = local.commons.org_unit
      Provisioner = local.commons.provisioner
      Solution    = local.commons.project
      fin_unit    = local.commons.fin_unit
    }
  }
}

terraform {
  required_providers {

    random = {
      source  = "hashicorp/random"
      version = ">= 3.4.0"
    }

    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.44"
    }
  }
}

locals {
  env = terraform.workspace
  account_mapping = {
    dev : [""]
    shared_services : ""
  }
  commons = {
    customer    = var.owner
    org_unit    = var.org_unit
    project     = var.project
    fin_unit    = var.fin_unit
    provisioner = var.provisioner
  }
}

