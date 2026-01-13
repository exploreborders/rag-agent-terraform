# Terraform Outputs for RAG Agent Infrastructure
# These outputs provide connection details and status information

# Network Information
output "network_name" {
  description = "Docker network name for the RAG agent services"
  value       = docker_network.rag_network.name
}

output "network_id" {
  description = "Docker network ID"
  value       = docker_network.rag_network.id
}

# PostgreSQL Service
output "postgres_container_name" {
  description = "PostgreSQL container name"
  value       = docker_container.postgres.name
}

output "postgres_host" {
  description = "PostgreSQL host (container name)"
  value       = local.postgres_container_name
}

output "postgres_port" {
  description = "PostgreSQL port on host"
  value       = var.postgres_port
}

output "postgres_connection_string" {
  description = "PostgreSQL connection string"
  value       = local.database_url
  sensitive   = true
}

output "postgres_status" {
  description = "PostgreSQL container health status"
  value       = docker_container.postgres.healthcheck[0].test
}

# Redis Service
output "redis_container_name" {
  description = "Redis container name"
  value       = docker_container.redis.name
}

output "redis_host" {
  description = "Redis host (container name)"
  value       = local.redis_container_name
}

output "redis_port" {
  description = "Redis port on host"
  value       = var.redis_port
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = local.redis_url
}

output "redis_status" {
  description = "Redis container health status"
  value       = docker_container.redis.healthcheck[0].test
}

# Application Service
output "app_container_name" {
  description = "Application container name"
  value       = docker_container.app.name
}

output "app_port" {
  description = "Application port on host"
  value       = var.app_port
}

output "app_url" {
  description = "Application URL"
  value       = "http://localhost:${var.app_port}"
}

output "app_health_url" {
  description = "Application health check URL"
  value       = "http://localhost:${var.app_port}/health"
}

output "app_docs_url" {
  description = "Application API documentation URL"
  value       = "http://localhost:${var.app_port}/docs"
}

output "app_status" {
  description = "Application container health status"
  value       = docker_container.app.healthcheck[0].test
}

# Monitoring Services
output "prometheus_container_name" {
  description = "Prometheus container name"
  value       = docker_container.prometheus.name
}

output "prometheus_port" {
  description = "Prometheus port on host"
  value       = var.prometheus_port
}

output "prometheus_url" {
  description = "Prometheus URL"
  value       = "http://localhost:${var.prometheus_port}"
}

output "prometheus_status" {
  description = "Prometheus container health status"
  value       = docker_container.prometheus.healthcheck[0].test
}

output "grafana_container_name" {
  description = "Grafana container name"
  value       = docker_container.grafana.name
}

output "grafana_port" {
  description = "Grafana port on host"
  value       = var.grafana_port
}

output "grafana_url" {
  description = "Grafana URL"
  value       = "http://localhost:${var.grafana_port}"
}

output "grafana_status" {
  description = "Grafana container health status"
  value       = docker_container.grafana.healthcheck[0].test
}

# Infrastructure Summary
output "infrastructure_status" {
  description = "Summary of infrastructure deployment status"
  value = {
    project_name = local.project_name
    environment  = local.environment
    network      = docker_network.rag_network.name
    services = {
      postgres = {
        name     = docker_container.postgres.name
        status   = "running"
        port     = var.postgres_port
        health   = docker_container.postgres.healthcheck[0].test
      }
      redis = {
        name     = docker_container.redis.name
        status   = "running"
        port     = var.redis_port
        health   = docker_container.redis.healthcheck[0].test
      }
      app = {
        name     = docker_container.app.name
        status   = "running"
        port     = var.app_port
        health   = docker_container.app.healthcheck[0].test
        urls     = {
          api    = "http://localhost:${var.app_port}"
          docs   = "http://localhost:${var.app_port}/docs"
          health = "http://localhost:${var.app_port}/health"
        }
      }
      prometheus = {
        name     = docker_container.prometheus.name
        status   = "running"
        port     = var.prometheus_port
        health   = docker_container.prometheus.healthcheck[0].test
        urls     = {
          ui = "http://localhost:${var.prometheus_port}"
        }
      }
      grafana = {
        name     = docker_container.grafana.name
        status   = "running"
        port     = var.grafana_port
        health   = docker_container.grafana.healthcheck[0].test
        urls     = {
          ui = "http://localhost:${var.grafana_port}"
        }
      }
    }
  }
}