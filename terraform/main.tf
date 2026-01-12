# Main Terraform Configuration for RAG Agent Infrastructure
# This file defines the Docker-based infrastructure for the RAG system

# Docker Network
resource "docker_network" "rag_network" {
  name   = local.network_name
  driver = var.network_driver

  labels {
    label = "project"
    value = local.project_name
  }

  labels {
    label = "environment"
    value = local.environment
  }
}

# PostgreSQL with pgvector
resource "docker_image" "postgres" {
  name = local.postgres_image
}

resource "docker_container" "postgres" {
  name  = local.postgres_container_name
  image = docker_image.postgres.image_id

  # Environment variables
  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=${var.postgres_db}",
    "PGDATA=/var/lib/postgresql/data/pgdata"
  ]

  # Ports
  ports {
    internal = 5432
    external = var.postgres_port
  }

  # Volumes for data persistence
  dynamic "volumes" {
    for_each = local.postgres_volumes
    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  # Memory limits (optional - commented out for basic functionality)
  # memory = var.postgres_memory_limit
  # memory_swap = var.postgres_memory_swap_limit

  # Health check
  healthcheck {
    test         = local.postgres_healthcheck.test
    interval     = local.postgres_healthcheck.interval
    timeout      = local.postgres_healthcheck.timeout
    retries      = local.postgres_healthcheck.retries
    start_period = "30s"
  }

  # Networking
  networks_advanced {
    name = docker_network.rag_network.name
  }

  # Restart policy
  restart = "unless-stopped"

  # Labels
  dynamic "labels" {
    for_each = local.common_tags
    content {
      label = lower(replace(labels.key, "_", "-"))
      value = labels.value
    }
  }

  # Wait for container to be healthy
  depends_on = [docker_network.rag_network]
}

# Redis for caching and memory
resource "docker_image" "redis" {
  name = local.redis_image
}

resource "docker_container" "redis" {
  name  = local.redis_container_name
  image = docker_image.redis.image_id

  # Command to start Redis with append-only file
  command = ["redis-server", "--appendonly", "yes"]

  # Ports
  ports {
    internal = 6379
    external = var.redis_port
  }

  # Volumes for data persistence
  dynamic "volumes" {
    for_each = local.redis_volumes
    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  # Memory limits (optional - commented out for basic functionality)
  # memory = var.redis_memory_limit

  # Health check
  healthcheck {
    test         = local.redis_healthcheck.test
    interval     = local.redis_healthcheck.interval
    timeout      = local.redis_healthcheck.timeout
    retries      = local.redis_healthcheck.retries
    start_period = "10s"
  }

  # Networking
  networks_advanced {
    name = docker_network.rag_network.name
  }

  # Restart policy
  restart = "unless-stopped"

  # Labels
  dynamic "labels" {
    for_each = local.common_tags
    content {
      label = lower(replace(labels.key, "_", "-"))
      value = labels.value
    }
  }

  # Wait for network to be ready
  depends_on = [docker_network.rag_network]
}

# Application Container
resource "docker_image" "app" {
  name = local.app_image

  # Build context from parent directory
  build {
    context    = "${path.root}/.."
    dockerfile = "docker/app/Dockerfile"
    tag        = [local.app_image]
  }
}

resource "docker_container" "app" {
  name  = local.app_container_name
  image = docker_image.app.image_id

  # Environment variables
  env = [
    "ENVIRONMENT=${local.environment}",
    "DATABASE_URL=${local.database_url}",
    "REDIS_URL=${local.redis_url}",
    "POSTGRES_HOST=${local.postgres_container_name}",
    "REDIS_HOST=${local.redis_container_name}",
    "OLLAMA_BASE_URL=http://host.docker.internal:11434"
  ]

  # Ports
  ports {
    internal = 8000
    external = var.app_port
  }

  # Volumes for data access
  dynamic "volumes" {
    for_each = local.app_volumes
    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  # Memory limits (optional - commented out for basic functionality)
  # memory = var.app_memory_limit

  # Health check
  healthcheck {
    test         = local.app_healthcheck.test
    interval     = local.app_healthcheck.interval
    timeout      = local.app_healthcheck.timeout
    retries      = local.app_healthcheck.retries
    start_period = "60s"
  }

  # Networking
  networks_advanced {
    name = docker_network.rag_network.name
  }

  # Restart policy
  restart = "unless-stopped"

  # Labels
  dynamic "labels" {
    for_each = local.common_tags
    content {
      label = lower(replace(labels.key, "_", "-"))
      value = labels.value
    }
  }

  # Dependencies - wait for database and cache to be ready
  depends_on = [
    docker_container.postgres,
    docker_container.redis,
    docker_network.rag_network
  ]
}