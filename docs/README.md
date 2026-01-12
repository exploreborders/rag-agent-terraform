# RAG Agent Terraform - Setup Guide

This guide provides comprehensive instructions for setting up and deploying the RAG Agent Terraform system.

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+**: [Download from python.org](https://python.org)
- **Docker 24.0+**: [Install Docker](https://docs.docker.com/get-docker/)
- **Terraform 1.5.0+**: [Install Terraform](https://developer.hashicorp.com/terraform/downloads)
- **Ollama**: [Install Ollama](https://ollama.ai/download)

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd rag-agent-terraform

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt

# Copy environment configuration
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` with your settings:

```bash
# Application settings
ENVIRONMENT=development
DEBUG=true

# Database settings (will be set by Docker)
POSTGRES_HOST=rag-agent-postgres-dev
REDIS_HOST=rag-agent-redis-dev

# Ollama settings (local instance)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2:latest
OLLAMA_EMBED_MODEL=embeddinggemma:latest
OLLAMA_VISION_MODEL=devstral-small-2:latest
```

### 3. Set Up Ollama Models

```bash
# Pull required models
ollama pull llama3.2:latest
ollama pull embeddinggemma:latest

# Optional: Pull vision model for image processing
ollama pull devstral-small-2:latest

# Verify models are available
ollama list
```

### 4. Deploy Infrastructure

```bash
# Initialize Terraform
cd terraform
terraform init

# Plan the deployment
terraform plan

# Apply the configuration
terraform apply
```

### 5. Start the Application

```bash
# Start the application
make dev

# Or run directly
cd src && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/health

# Access API documentation
open http://localhost:8000/docs
```

## 📋 Detailed Setup

### Local Development Setup

For development with hot reloading and debugging:

```bash
# Use the development Makefile target
make workflow-dev

# This will:
# 1. Set up the virtual environment
# 2. Install dependencies
# 3. Initialize Terraform
# 4. Deploy infrastructure
# 5. Start the development server
```

### Docker Compose Alternative

For containerized development:

```bash
# Use Docker Compose instead of Terraform
docker-compose up -d --build

# View logs
docker-compose logs -f app
```

### Production Deployment

For production deployment:

```bash
# Set production environment
export ENVIRONMENT=production

# Deploy with Terraform
cd terraform
terraform apply -var="environment=production"

# Build and deploy application
make deploy
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `development` | No |
| `DEBUG` | Enable debug mode | `false` | No |
| `SECRET_KEY` | Application secret key | Auto-generated | No |
| `API_HOST` | API bind address | `0.0.0.0` | No |
| `API_PORT` | API port | `8000` | No |
| `DATABASE_URL` | PostgreSQL connection URL | Auto-assembled | No |
| `REDIS_URL` | Redis connection URL | Auto-assembled | No |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` | Yes |
| `OLLAMA_MODEL` | Primary LLM model | `llama3.2:latest` | Yes |
| `OLLAMA_EMBED_MODEL` | Embedding model | `embeddinggemma:latest` | Yes |
| `OLLAMA_VISION_MODEL` | Vision model | `devstral-small-2:latest` | No |
| `MAX_UPLOAD_SIZE` | Max file upload size (bytes) | `52428800` | No |

### Database Configuration

The system uses PostgreSQL with the pgvector extension:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables (handled automatically)
CREATE TABLE documents (...);
CREATE TABLE document_chunks (...);
```

### Redis Configuration

Redis is used for:
- Conversation memory
- Query result caching
- Document processing cache

Default configuration:
- Port: `6379`
- Append-only file: Enabled
- Memory management: LRU eviction

## 🐳 Docker Configuration

### Application Container

The application runs in a multi-stage Docker container:

```dockerfile
# Build stage: Install dependencies
FROM python:3.11-slim as builder
# ... dependency installation

# Production stage: Runtime environment
FROM python:3.11-slim as production
# ... application setup
```

### Container Resources

Default resource limits:
- **Memory**: 1GB (application), 512MB (PostgreSQL), 256MB (Redis)
- **Health Checks**: 30s interval, 10s timeout, 3 retries

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed
```bash
# Check if Ollama is running
ollama serve

# Check available models
ollama list

# Test API connectivity
curl http://localhost:11434/api/tags
```

#### 2. Database Connection Error
```bash
# Check PostgreSQL container
docker ps | grep postgres

# View container logs
docker logs rag-agent-postgres-dev

# Test database connectivity
docker exec -it rag-agent-postgres-dev psql -U rag_user -d rag_db
```

#### 3. Redis Connection Failed
```bash
# Check Redis container
docker ps | grep redis

# Test Redis connectivity
docker exec -it rag-agent-redis-dev redis-cli ping
```

#### 4. Application Won't Start
```bash
# Check application logs
docker logs rag-agent-app-dev

# Verify environment variables
docker exec -it rag-agent-app-dev env | grep -E "(DATABASE|REDIS|OLLAMA)"

# Test health endpoint
curl http://localhost:8000/health
```

### Health Checks

Monitor system health:

```bash
# Overall health
curl http://localhost:8000/health

# System statistics
curl http://localhost:8000/stats

# Container health
docker ps
docker stats
```

### Logs and Debugging

```bash
# Application logs
docker logs -f rag-agent-app-dev

# Infrastructure logs
docker-compose logs -f

# Terraform state
cd terraform && terraform show

# Ollama logs
ollama logs
```

## 🔄 Updates and Maintenance

### Updating Models

```bash
# Pull latest model versions
ollama pull llama3.2:latest
ollama pull embeddinggemma:latest

# Restart application
docker restart rag-agent-app-dev
```

### Database Migrations

The system handles schema initialization automatically. For manual migrations:

```bash
# Access database
docker exec -it rag-agent-postgres-dev psql -U rag_user -d rag_db

# Run migration scripts
# (Schema changes are handled by the application)
```

### Backup and Recovery

```bash
# Database backup
docker exec rag-agent-postgres-dev pg_dump -U rag_user rag_db > backup.sql

# Redis backup
docker exec rag-agent-redis-dev redis-cli save

# Volume backups (if using named volumes)
docker run --rm -v rag-agent_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .
```

## 📞 Support

If you encounter issues:

1. Check the [troubleshooting guide](./troubleshooting.md)
2. Review the [API documentation](./api.md)
3. Check GitHub Issues for similar problems
4. Create a new issue with detailed information

## 🔗 Next Steps

Once setup is complete:

1. **Upload Documents**: Start with the [usage guide](./rag-system.md)
2. **API Integration**: Check the [API documentation](./api.md)
3. **Customization**: Modify configurations for your use case
4. **Monitoring**: Set up logging and monitoring as needed

Happy RAG building! 🚀