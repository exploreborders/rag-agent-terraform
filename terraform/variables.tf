# Terraform Variables for RAG Agent Infrastructure
# These variables configure the local Docker-based infrastructure

# Docker Configuration
variable "docker_host" {
  description = "Docker daemon host (leave empty for local Docker socket)"
  type        = string
  default     = null
}

variable "docker_registry_username" {
  description = "Docker registry username (optional)"
  type        = string
  default     = null
}

variable "docker_registry_password" {
  description = "Docker registry password (optional)"
  type        = string
  sensitive   = true
  default     = null
}

# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "rag-agent"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod"
  }
}

# PostgreSQL Configuration
variable "postgres_version" {
  description = "PostgreSQL version to use"
  type        = string
  default     = "15"
}

variable "postgres_port" {
  description = "PostgreSQL port on host"
  type        = number
  default     = 5432
}

variable "postgres_user" {
  description = "PostgreSQL username"
  type        = string
  default     = "rag_user"
}

variable "postgres_password" {
  description = "PostgreSQL password"
  type        = string
  default     = "rag_password"
  sensitive   = true
}

variable "postgres_db" {
  description = "PostgreSQL database name"
  type        = string
  default     = "rag_db"
}

# Redis Configuration
variable "redis_version" {
  description = "Redis version to use"
  type        = string
  default     = "7"
}

variable "redis_port" {
  description = "Redis port on host"
  type        = number
  default     = 6379
}

# Application Configuration
variable "app_port" {
  description = "Application port on host"
  type        = number
  default     = 8000
}

variable "app_image_tag" {
  description = "Application Docker image tag"
  type        = string
  default     = "latest"
}

# Monitoring Configuration
variable "prometheus_port" {
  description = "Prometheus port on host"
  type        = number
  default     = 9090
}

variable "grafana_port" {
  description = "Grafana port on host"
  type        = number
  default     = 3000
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  default     = "admin"
  sensitive   = true
}

variable "postgres_exporter_port" {
  description = "PostgreSQL exporter port on host"
  type        = number
  default     = 9187
}

variable "redis_exporter_port" {
  description = "Redis exporter port on host"
  type        = number
  default     = 9121
}

variable "node_exporter_port" {
  description = "Node exporter port on host"
  type        = number
  default     = 9100
}

# MCP Coordinator Configuration
variable "mcp_coordinator_port" {
  description = "MCP Coordinator port on host"
  type        = number
  default     = 8001
}

variable "mcp_coordinator_memory_limit" {
  description = "Memory limit for MCP Coordinator container in MB"
  type        = number
  default     = 536870912 # 512MB in bytes
}

variable "mcp_coordinator_memory_swap_limit" {
  description = "Memory swap limit for MCP Coordinator container in MB"
  type        = number
  default     = 1073741824 # 1GB in bytes
}

# Network Configuration
variable "network_name" {
  description = "Docker network name"
  type        = string
  default     = "rag-agent-network"
}

variable "network_driver" {
  description = "Docker network driver"
  type        = string
  default     = "bridge"
}

# Resource Limits
variable "postgres_memory_limit" {
  description = "PostgreSQL container memory limit"
  type        = string
  default     = "512m"
}

variable "postgres_memory_swap_limit" {
  description = "PostgreSQL container memory swap limit"
  type        = string
  default     = "1g"
}

variable "redis_memory_limit" {
  description = "Redis container memory limit"
  type        = string
  default     = "256m"
}

variable "app_memory_limit" {
  description = "Application container memory limit"
  type        = string
  default     = "1g"
}

# Health Check Configuration
variable "healthcheck_interval" {
  description = "Health check interval in seconds"
  type        = string
  default     = "30s"
}

variable "healthcheck_timeout" {
  description = "Health check timeout in seconds"
  type        = string
  default     = "10s"
}

variable "healthcheck_retries" {
  description = "Number of health check retries"
  type        = number
  default     = 3
}

variable "postgres_startup_timeout" {
  description = "PostgreSQL startup timeout in seconds"
  type        = number
  default     = 300
}

variable "redis_startup_timeout" {
  description = "Redis startup timeout in seconds"
  type        = number
  default     = 120
}