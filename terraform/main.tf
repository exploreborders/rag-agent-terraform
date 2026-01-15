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

  # Command to start Redis with configuration file
  command = ["redis-server", "/etc/redis/redis.conf"]

  # Ports
  ports {
    internal = 6379
    external = var.redis_port
  }
  # Additional port for message queue operations
  ports {
    internal = 6380
    external = 6380
  }

  # Volumes for data persistence and configuration
  dynamic "volumes" {
    for_each = local.redis_volumes
    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  # Redis configuration file
  volumes {
    host_path      = abspath("${path.root}/../redis/redis.conf")
    container_path = "/etc/redis/redis.conf"
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

# Application Container - expects image to be built separately (fast!)
resource "docker_image" "app" {
  name = local.app_image
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
    "OLLAMA_BASE_URL=http://host.docker.internal:11434",
    "MCP_COORDINATOR_URL=http://${local.mcp_coordinator_container_name}:${var.mcp_coordinator_port}",
    "LANGGRAPH_CHECKPOINT_URL=${local.database_url}",
    "AGENT_MESSAGE_CHANNELS=${jsonencode(local.agent_message_channels)}",
    "MULTI_AGENT_ENABLED=true",
    "REDIS_MESSAGE_QUEUE_ENABLED=true"
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

  # Dependencies - wait for database, cache, and MCP coordinator to be ready
  depends_on = [
    docker_container.postgres,
    docker_container.redis,
    docker_container.mcp_coordinator,
    docker_network.rag_network
  ]
}

# MCP Coordinator Container - uses local image (build separately)
resource "docker_image" "mcp_coordinator" {
  name = local.mcp_coordinator_image
}

# MCP Coordinator Container
resource "docker_container" "mcp_coordinator" {
  name  = local.mcp_coordinator_container_name
  image = docker_image.mcp_coordinator.image_id

  # Environment variables
  env = [
    "REDIS_URL=${local.redis_url}",
    "DOCKER_HOST=unix:///var/run/docker.sock",
    "MCP_COORDINATOR_PORT=${var.mcp_coordinator_port}",
    "AGENT_MESSAGE_CHANNELS=${jsonencode(local.agent_message_channels)}"
  ]

  # Ports
  ports {
    internal = var.mcp_coordinator_port
    external = var.mcp_coordinator_port
  }

  # Docker socket for MCP Tool Container Management
  volumes {
    host_path      = "/var/run/docker.sock"
    container_path = "/var/run/docker.sock"
  }

  # Data volumes
  dynamic "volumes" {
    for_each = local.mcp_volumes
    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  # Memory limits
  memory      = var.mcp_coordinator_memory_limit
  memory_swap = var.mcp_coordinator_memory_swap_limit

  # Health check
  healthcheck {
    test         = local.mcp_coordinator_healthcheck.test
    interval     = local.mcp_coordinator_healthcheck.interval
    timeout      = local.mcp_coordinator_healthcheck.timeout
    retries      = local.mcp_coordinator_healthcheck.retries
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

  # Dependencies - wait for Redis and network
  depends_on = [
    docker_container.redis,
    docker_network.rag_network
  ]
}


# Prometheus for metrics collection
resource "docker_image" "prometheus" {
  name = local.prometheus_image
}

resource "docker_container" "prometheus" {
  name  = local.prometheus_container_name
  image = docker_image.prometheus.image_id

  # Ports
  ports {
    internal = 9090
    external = var.prometheus_port
  }

  # Volumes for configuration and data
  dynamic "volumes" {
    for_each = local.prometheus_volumes
    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  # Health check
  healthcheck {
    test         = local.prometheus_healthcheck.test
    interval     = local.prometheus_healthcheck.interval
    timeout      = local.prometheus_healthcheck.timeout
    retries      = local.prometheus_healthcheck.retries
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

  # Wait for network to be ready
  depends_on = [docker_network.rag_network]
}

# Grafana for metrics visualization
resource "docker_image" "grafana" {
  name = local.grafana_image
}

resource "docker_container" "grafana" {
  name  = local.grafana_container_name
  image = docker_image.grafana.image_id

  # Environment variables
  env = [
    "GF_SECURITY_ADMIN_PASSWORD=${var.grafana_admin_password}",
    "GF_USERS_ALLOW_SIGN_UP=false"
  ]

  # Ports
  ports {
    internal = 3000
    external = var.grafana_port
  }

  # Volumes for data persistence
  dynamic "volumes" {
    for_each = local.grafana_volumes
    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  # Health check
  healthcheck {
    test         = local.grafana_healthcheck.test
    interval     = local.grafana_healthcheck.interval
    timeout      = local.grafana_healthcheck.timeout
    retries      = local.grafana_healthcheck.retries
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

  # Wait for network to be ready
  depends_on = [docker_network.rag_network]
}

# PostgreSQL Exporter
resource "docker_image" "postgres_exporter" {
  name = local.postgres_exporter_image
}

resource "docker_container" "postgres_exporter" {
  name  = local.postgres_exporter_container_name
  image = docker_image.postgres_exporter.image_id

  # Environment variables for PostgreSQL connection
  env = [
    "DATA_SOURCE_NAME=postgresql://${var.postgres_user}:${var.postgres_password}@${local.postgres_container_name}:${var.postgres_port}/${var.postgres_db}?sslmode=disable"
  ]

  # Ports
  ports {
    internal = 9187
    external = var.postgres_exporter_port
  }

  # Health check
  healthcheck {
    test         = local.postgres_exporter_healthcheck.test
    interval     = local.postgres_exporter_healthcheck.interval
    timeout      = local.postgres_exporter_healthcheck.timeout
    retries      = local.postgres_exporter_healthcheck.retries
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

  # Dependencies
  depends_on = [docker_container.postgres]
}

# Redis Exporter
resource "docker_image" "redis_exporter" {
  name = local.redis_exporter_image
}

resource "docker_container" "redis_exporter" {
  name  = local.redis_exporter_container_name
  image = docker_image.redis_exporter.image_id

  # Command for Redis exporter
  command = [
    "-redis.addr",
    "redis://${local.redis_container_name}:${var.redis_port}"
  ]

  # Ports
  ports {
    internal = 9121
    external = var.redis_exporter_port
  }

  # Health check
  healthcheck {
    test         = local.redis_exporter_healthcheck.test
    interval     = local.redis_exporter_healthcheck.interval
    timeout      = local.redis_exporter_healthcheck.timeout
    retries      = local.redis_exporter_healthcheck.retries
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

  # Dependencies
  depends_on = [docker_container.redis]
}

# Node Exporter
resource "docker_image" "node_exporter" {
  name = local.node_exporter_image
}

resource "docker_container" "node_exporter" {
  name  = local.node_exporter_container_name
  image = docker_image.node_exporter.image_id

  # Command to run node exporter
  command = [
    "--path.rootfs=/host",
    "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)",
    "--collector.netclass",
    "--collector.netdev",
    "--collector.cpu",
    "--collector.meminfo",
    "--collector.loadavg",
    "--collector.filesystem",
    "--collector.diskstats"
  ]

  # Volumes for host access
  volumes {
    host_path      = "/"
    container_path = "/host"
    read_only      = true
  }

  # Ports
  ports {
    internal = 9100
    external = var.node_exporter_port
  }

  # Health check
  healthcheck {
    test         = local.node_exporter_healthcheck.test
    interval     = local.node_exporter_healthcheck.interval
    timeout      = local.node_exporter_healthcheck.timeout
    retries      = local.node_exporter_healthcheck.retries
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

  # Dependencies
  depends_on = [docker_network.rag_network]
}