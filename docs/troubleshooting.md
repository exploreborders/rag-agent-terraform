# RAG Agent Terraform - Troubleshooting Guide

This guide helps diagnose and resolve common issues with the RAG Agent Terraform system.

## 🔍 Quick Diagnosis

### System Health Check

```bash
# Check overall system health
curl http://localhost:8000/health

# Check service containers
docker ps

# Check container logs
docker logs rag-agent-app-dev
docker logs rag-agent-postgres-dev
docker logs rag-agent-redis-dev
```

### Common Symptoms and Solutions

## 🚨 Critical Issues

### 1. System Won't Start

**Symptoms:**
- Application container fails to start
- Health check returns 503 errors
- Terraform apply fails

**Solutions:**

```bash
# Check Docker daemon
docker info

# Verify Ollama is running
ollama serve

# Check Ollama models
ollama list

# Validate Terraform configuration
cd terraform && terraform validate

# Rebuild containers
docker-compose down
docker-compose up --build
```

### 2. Database Connection Failed

**Symptoms:**
- `vector_store_healthy: false` in health check
- Queries fail with connection errors
- Application logs show PostgreSQL errors

**Solutions:**

```bash
# Check PostgreSQL container
docker ps | grep postgres

# View PostgreSQL logs
docker logs rag-agent-postgres-dev

# Test database connection
docker exec -it rag-agent-postgres-dev psql -U rag_user -d rag_db -c "SELECT 1;"

# Check connection string
docker exec -it rag-agent-app-dev env | grep DATABASE_URL

# Restart database
docker restart rag-agent-postgres-dev
```

### 3. Redis Connection Failed

**Symptoms:**
- `memory_healthy: false` in health check
- Caching errors in logs
- Session management fails

**Solutions:**

```bash
# Check Redis container
docker ps | grep redis

# View Redis logs
docker logs rag-agent-redis-dev

# Test Redis connection
docker exec -it rag-agent-redis-dev redis-cli ping

# Check Redis configuration
docker exec -it rag-agent-redis-dev redis-cli config get appendonly

# Restart Redis
docker restart rag-agent-redis-dev
```

### 4. Ollama Connection Failed

**Symptoms:**
- `ollama_available: false` in health check
- Embedding generation fails
- Model loading errors

**Solutions:**

```bash
# Check Ollama service
ollama serve

# Verify models are installed
ollama list

# Check Ollama API
curl http://localhost:11434/api/tags

# Test model loading
ollama run llama3.2:latest "Hello"

# Check network connectivity (for Docker)
curl http://host.docker.internal:11434/api/tags
```

## 📁 Document Processing Issues

### 1. File Upload Fails

**Error:** "Unsupported file type"

**Solutions:**
```bash
# Check supported types
curl http://localhost:8000/health  # Shows supported formats

# Verify file type
file your_document.pdf

# Check file size (max 50MB)
ls -lh your_document.pdf
```

**Error:** "File too large"

**Solutions:**
```bash
# Check configured limit
docker exec -it rag-agent-app-dev env | grep MAX_UPLOAD_SIZE

# Split large documents
# For PDFs: Use PDF splitting tools
# For text: Split into smaller files
```

### 2. PDF Processing Fails

**Error:** "Document processing failed"

**Symptoms:**
- PDF upload succeeds but processing fails
- Empty chunks generated
- OCR errors

**Solutions:**

```bash
# Check PyMuPDF installation
docker exec -it rag-agent-app-dev python -c "import fitz; print('PyMuPDF OK')"

# Test PDF manually
docker exec -it rag-agent-app-dev python -c "
import fitz
doc = fitz.open('/path/to/test.pdf')
print(f'Pages: {len(doc)}')
print(f'Text: {doc[0].get_text()[:200]}')
doc.close()
"

# Check for corrupted PDF
pdfinfo your_document.pdf
```

### 3. Image OCR Fails

**Error:** OCR processing failed

**Symptoms:**
- Image uploads succeed but no text extracted
- Empty chunks from images

**Solutions:**

```bash
# Check Tesseract installation
docker exec -it rag-agent-app-dev tesseract --version

# Test OCR manually
docker exec -it rag-agent-app-dev tesseract /path/to/image.png stdout

# Check image format
docker exec -it rag-agent-app-dev python -c "
from PIL import Image
img = Image.open('/path/to/image.png')
print(f'Format: {img.format}, Mode: {img.mode}, Size: {img.size}')
"
```

### 4. Text Processing Issues

**Error:** Encoding errors

**Solutions:**

```bash
# Check file encoding
file your_document.txt

# Convert encoding if needed
iconv -f latin1 -t utf8 your_document.txt > converted.txt

# Test encoding detection
python -c "
import chardet
with open('your_document.txt', 'rb') as f:
    raw = f.read()
    result = chardet.detect(raw)
    print(f'Detected encoding: {result}')
"
```

## ❓ Query Processing Issues

### 1. Queries Return No Results

**Symptoms:**
- Query succeeds but returns empty answer
- No sources found

**Solutions:**

```bash
# Check if documents are indexed
curl http://localhost:8000/stats

# List documents
curl http://localhost:8000/documents

# Test embedding generation
docker exec -it rag-agent-app-dev python -c "
from app.ollama_client import OllamaClient
import asyncio

async def test():
    client = OllamaClient()
    result = await client.embed_batch(['test text'])
    print(f'Embedding dimension: {len(result[0])}')

asyncio.run(test())
"
```

### 2. Slow Query Responses

**Symptoms:**
- Queries take >5 seconds
- High processing time in responses

**Solutions:**

```bash
# Check system resources
docker stats

# Clear caches
curl -X POST http://localhost:8000/cache/clear

# Check vector search performance
docker exec -it rag-agent-postgres-dev psql -U rag_user -d rag_db -c "
EXPLAIN ANALYZE
SELECT id, embedding <=> '[0.1,0.2,...]' as distance
FROM document_chunks
ORDER BY embedding <=> '[0.1,0.2,...]'
LIMIT 5;
"
```

### 3. Low Similarity Scores

**Symptoms:**
- All similarity scores < 0.5
- Irrelevant results

**Solutions:**

```bash
# Adjust similarity threshold
# In .env: SIMILARITY_THRESHOLD=0.6

# Check embedding quality
# Test with known similar texts
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "top_k": 10}'
```

## 🔧 Infrastructure Issues

### 1. Terraform Apply Fails

**Common Errors:**

**Error:** "docker_image.app will be created"
```bash
# This is normal - Terraform is planning to build the image
# The error occurs if Dockerfile is missing
ls -la docker/app/Dockerfile
```

**Error:** Duplicate provider configuration
```bash
# Remove duplicate provider blocks in terraform/*.tf files
cd terraform
grep -n "required_providers" *.tf
```

**Error:** Connection to Docker daemon failed
```bash
# Check Docker is running
docker info

# Check Docker socket permissions
ls -la /var/run/docker.sock

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
```

### 2. Container Networking Issues

**Symptoms:**
- Containers can't communicate
- Service discovery fails

**Solutions:**

```bash
# Check Docker network
docker network ls

# Inspect network
docker network inspect rag-agent-network

# Test inter-container connectivity
docker exec -it rag-agent-app-dev ping rag-agent-postgres-dev

# Check environment variables
docker exec -it rag-agent-app-dev env | grep -E "(DATABASE|REDIS)_"
```

### 3. Resource Constraints

**Symptoms:**
- Out of memory errors
- Container restarts
- Slow performance

**Solutions:**

```bash
# Check resource usage
docker stats

# Adjust memory limits in terraform/variables.tf
# postgres_memory_limit = "1g"
# redis_memory_limit = "512m"
# app_memory_limit = "2g"

# Reapply Terraform
cd terraform && terraform apply
```

## 📊 Performance Issues

### 1. High Memory Usage

```bash
# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check Ollama memory
ollama ps

# Clear caches
curl -X POST http://localhost:8000/cache/clear

# Restart services
docker-compose restart
```

### 2. Slow Document Processing

```bash
# Check chunking performance
time curl -X POST http://localhost:8000/documents/upload \
  -F "file=@large_document.pdf"

# Optimize chunking parameters in .env
# CHUNK_SIZE=500  # Smaller chunks for faster processing
# MAX_WORKERS=8   # More parallel workers

# Monitor processing
docker logs -f rag-agent-app-dev
```

### 3. Database Performance

```bash
# Check database performance
docker exec -it rag-agent-postgres-dev psql -U rag_user -d rag_db -c "
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
FROM pg_stat_user_tables;
"

# Analyze query performance
docker exec -it rag-agent-postgres-dev psql -U rag_user -d rag_db -c "
EXPLAIN ANALYZE SELECT COUNT(*) FROM document_chunks;
"

# Reindex if needed
docker exec -it rag-agent-postgres-dev psql -U rag_user -d rag_db -c "
REINDEX INDEX CONCURRENTLY idx_document_chunks_embedding;
"
```

## 🔄 Recovery Procedures

### 1. Reset Database

```bash
# Stop application
docker stop rag-agent-app-dev

# Reset database
docker exec -it rag-agent-postgres-dev psql -U rag_user -d rag_db -c "
DROP TABLE IF EXISTS document_chunks;
DROP TABLE IF EXISTS documents;
"

# Restart application (will recreate schema)
docker start rag-agent-app-dev

# Re-upload documents
curl -X POST http://localhost:8000/documents/upload -F "file=@document.pdf"
```

### 2. Clear All Data

```bash
# Stop services
docker-compose down

# Remove volumes (WARNING: destroys all data)
docker volume rm rag-agent_postgres_data rag-agent_redis_data

# Restart fresh
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### 3. Full System Reset

```bash
# Destroy infrastructure
cd terraform && terraform destroy -auto-approve

# Clean up containers and volumes
docker-compose down -v --remove-orphans
docker system prune -f

# Rebuild from scratch
cd terraform && terraform init && terraform apply
```

## 📞 Getting Help

### Diagnostic Information

When reporting issues, include:

```bash
# System information
uname -a
docker --version
docker-compose --version

# Service status
docker ps -a
curl http://localhost:8000/health
curl http://localhost:8000/stats

# Recent logs
docker logs --tail 100 rag-agent-app-dev
docker logs --tail 50 rag-agent-postgres-dev
docker logs --tail 50 rag-agent-redis-dev

# Configuration
cat .env | grep -v PASSWORD
cd terraform && terraform version
```

### Support Channels

1. **GitHub Issues**: Create detailed bug reports
2. **Documentation**: Check this troubleshooting guide
3. **Logs**: Enable debug logging for more information
4. **Community**: Check existing issues for similar problems

### Debug Mode

Enable debug logging:

```bash
# In .env
DEBUG=true
LOG_LEVEL=DEBUG

# Restart application
docker restart rag-agent-app-dev

# View debug logs
docker logs -f rag-agent-app-dev
```

This troubleshooting guide covers the most common issues. For complex problems, please provide detailed diagnostic information when seeking help.