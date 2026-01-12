# Docker Provider Configuration
# This provider manages Docker containers locally for development

# Configure Docker provider to use local Docker daemon
provider "docker" {
  host = var.docker_host

  # Optional: Configure registry authentication if needed
  # registry_auth {
  #   address  = "registry-1234567890.amazonaws.com"
  #   username = var.docker_registry_username
  #   password = var.docker_registry_password
  # }
}