# Computed Local Values for RAG Agent Infrastructure
# These locals provide reusable computed values throughout the configuration

locals {
  # Project identifiers
  project_name = var.project_name
  environment  = var.environment

  # Container names with consistent naming
  postgres_container_name = "${local.project_name}-postgres-${local.environment}"
  redis_container_name    = "${local.project_name}-redis-${local.environment}"
  app_container_name      = "${local.project_name}-app-${local.environment}"
  prometheus_container_name = "${local.project_name}-prometheus-${local.environment}"
  grafana_container_name   = "${local.project_name}-grafana-${local.environment}"

  # Network configuration
  network_name = var.network_name != "" ? var.network_name : "${local.project_name}-network"

  # Database connection strings
  database_url = "postgresql://${var.postgres_user}:${var.postgres_password}@${local.postgres_container_name}:${var.postgres_port}/${var.postgres_db}"
  redis_url    = "redis://${local.redis_container_name}:${var.redis_port}"

  # Docker image names
  postgres_image = "pgvector/pgvector:pg15"
  redis_image    = "redis:${var.redis_version}"
  app_image      = "${local.project_name}:${var.app_image_tag}"
  prometheus_image = "prom/prometheus:latest"
  grafana_image   = "grafana/grafana:latest"

  # Common labels for all resources
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "Terraform"
    Component   = "rag-agent-infrastructure"
  }

  # Health check configurations
  postgres_healthcheck = {
    test     = ["CMD-SHELL", "pg_isready -U ${var.postgres_user} -d ${var.postgres_db}"]
    interval = var.healthcheck_interval
    timeout  = var.healthcheck_timeout
    retries  = var.healthcheck_retries
  }

  redis_healthcheck = {
    test     = ["CMD", "redis-cli", "ping"]
    interval = var.healthcheck_interval
    timeout  = var.healthcheck_timeout
    retries  = var.healthcheck_retries
  }

  app_healthcheck = {
    test     = ["CMD", "curl", "-f", "http://localhost:${var.app_port}/health"]
    interval = var.healthcheck_interval
    timeout  = var.healthcheck_timeout
    retries  = var.healthcheck_retries
  }

  prometheus_healthcheck = {
    test     = ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
    interval = var.healthcheck_interval
    timeout  = var.healthcheck_timeout
    retries  = var.healthcheck_retries
  }

  grafana_healthcheck = {
    test     = ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3000/api/health"]
    interval = var.healthcheck_interval
    timeout  = var.healthcheck_timeout
    retries  = var.healthcheck_retries
  }

  # Volume configurations
  postgres_volumes = [
    {
      host_path      = abspath("${path.root}/../data/postgres")
      container_path = "/var/lib/postgresql/data"
    }
  ]

  redis_volumes = [
    {
      host_path      = abspath("${path.root}/../data/redis")
      container_path = "/data"
    }
  ]

  app_volumes = [
    {
      host_path      = abspath("${path.root}/../data")
      container_path = "/app/data"
    }
  ]

  prometheus_volumes = [
    {
      host_path      = abspath("${path.root}/../data/prometheus")
      container_path = "/prometheus"
    },
    {
      host_path      = abspath("${path.root}/../monitoring/prometheus.yml")
      container_path = "/etc/prometheus/prometheus.yml"
    }
  ]

  grafana_volumes = [
    {
      host_path      = abspath("${path.root}/../data/grafana")
      container_path = "/var/lib/grafana"
    }
  ]
}