# AGENTS.md - Agentic Coding Guidelines for RAG Agent Terraform Project

## 🚀 Project Overview

This is a Terraform-managed local infrastructure for an agentic RAG (Retrieval-Augmented Generation) system combining LangChain orchestration with LlamaIndex document indexing. The system processes PDF, text, and image documents using local Ollama AI models, PostgreSQL with pgvector for embeddings, and Redis for caching.

The project consists of:
- **Backend**: FastAPI application with document processing and RAG capabilities
- **Frontend**: React/TypeScript application (in `/frontend` directory)
- **Infrastructure**: Terraform-managed Docker containers and networking

## 🛠️ Development Commands

### Environment Setup
```bash
# Complete development environment setup
make workflow-dev

# Or manually:
make setup        # Set up Python virtual environment
make install      # Install dependencies
make infra-init   # Initialize Terraform
make deploy       # Deploy infrastructure
make dev          # Start development server
```

### Infrastructure Management
```bash
# Initialize and apply Terraform infrastructure
make infra-init
make infra-apply

# Or manually:
cd terraform && terraform init && terraform apply

# Destroy infrastructure
make destroy
# or
cd terraform && terraform destroy

# Validate Terraform configuration
make infra-validate
# or
cd terraform && terraform validate

# Check infrastructure health
docker ps
```

### Application Development
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
make install
# or
pip install -e .[dev]

# Run FastAPI application with hot reload
make dev
# or
cd src && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
open http://localhost:8000/docs

# Run frontend development server (if available)
cd frontend && npm run dev
```

### Frontend Development
```bash
# Install frontend dependencies
cd frontend && npm install

# Start development server
cd frontend && npm start

# Run frontend tests
cd frontend && npm test

# Build for production
cd frontend && npm run build

# Run tests in CI mode with coverage
cd frontend && npm run test:ci
```

### Testing Commands
```bash
# Run all tests
make test
# or
cd src && python -m pytest tests/ -v --cov=app --cov-report=html

# Run unit tests only (excludes integration tests)
make test-unit
# or
cd src && python -m pytest tests/ -v -m "not integration"

# Run integration tests only
make test-integration
# or
cd src && python -m pytest tests/ -v -m "integration"

# Run specific test file
cd src && python -m pytest tests/test_api.py -v

# Run single test function
cd src && python -m pytest tests/test_api.py::TestAPIIntegration::test_health_endpoint -v

# Run tests with coverage report
make test-cov
# or
cd src && python -m pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

# Run tests in parallel (requires pytest-xdist)
cd src && python -m pytest tests/ -n auto
```

### Code Quality & Linting
```bash
# Format Python code with black (88 character line length)
cd src && black .

# Sort imports with isort
cd src && isort .

# Lint Python code with flake8 (max 100 chars, E501 ignored)
cd src && flake8 .

# Type checking with mypy
cd src && mypy .

# Run all code quality checks
make lint
# or
cd src && black . && isort . && flake8 . && mypy .
```

#### Code Formatting Configuration
- **Black**: 88 character line length, Python 3.11+ syntax
- **Flake8**: 100 character line length (E501 error ignored)
- **isort**: Black-compatible import sorting
- **MyPy**: Strict type checking enabled

### Document Processing & Data Management
```bash
# Ingest sample documents
cd src && python scripts/ingest_documents.py --input data/documents/ --recursive

# Setup vector database
cd src && python scripts/setup_vector_db.py

# Evaluate RAG performance
cd src && python scripts/evaluate_rag.py --test-set data/test_queries.json

# Seed database with sample data
./scripts/seed_db.sh
```

## 📝 Code Style Guidelines

### Python Code Style

#### Import Organization
```python
# Standard library imports (alphabetically sorted)
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party imports (alphabetically sorted)
import httpx
from fastapi import FastAPI, HTTPException
from langchain.chains import RetrievalQA
from llama_index import VectorStoreIndex
import psycopg2
import redis
import structlog

# Local imports (organized by module hierarchy)
from app.config import Settings
from app.models import Document, Query
from app.vector_store import VectorStore
```

#### Naming Conventions
- **Classes**: `PascalCase` (e.g., `RAGAgent`, `DocumentLoader`)
- **Functions/Methods**: `snake_case` (e.g., `process_document`, `get_embedding`)
- **Variables**: `snake_case` (e.g., `document_text`, `vector_embeddings`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_CHUNK_SIZE`, `DEFAULT_MODEL`)
- **Private members**: `_snake_case` (e.g., `_ollama_client`, `_vector_store`)

#### Type Hints
```python
from typing import Dict, List, Optional, Union, Any
import numpy as np

def process_document(
    document_path: Path,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """Process a document and return chunks with metadata."""

def get_embedding(text: str) -> Optional[np.ndarray]:
    """Generate embeddings for text using Ollama."""

async def query_rag(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
    """Query the RAG system with optional filtering."""
```

#### Error Handling
```python
import logging
from fastapi import HTTPException
from app.exceptions import DocumentProcessingError, VectorStoreError

logger = logging.getLogger(__name__)

def load_document(file_path: Path) -> Document:
    """Load and validate document from file path."""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Document processing logic
        content = file_path.read_text(encoding='utf-8')

        # Validate content
        if not content.strip():
            raise ValueError("Document is empty")

        return Document(content=content, path=file_path)

    except FileNotFoundError as e:
        logger.error(f"Document loading failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error for {file_path}: {e}")
        raise HTTPException(status_code=400, detail="Invalid file encoding")
    except Exception as e:
        logger.error(f"Unexpected error loading document: {e}")
        raise DocumentProcessingError(f"Failed to load document: {str(e)}")
```

#### Async/Await Patterns
```python
import asyncio
from typing import List, Dict, Any

class VectorStore:
    async def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts concurrently."""
        tasks = [self._embed_single(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def _embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        # Implementation with proper async handling
        async with self.ollama_client:
            return await self.ollama_client.embed(text)
```

#### Logging with Structlog
```python
import logging
import structlog

# Configure structured logging in main.py
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
shared_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
]
structlog.configure(
    processors=shared_processors,
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage in functions
logger.info(
    "Document uploaded and processed",
    filename=file.filename,
    file_size=file_size,
    chunks_count=result.chunks_count,
    processing_time=processing_time,
)

logger.error(
    "Document processing failed",
    filename=file.filename,
    error=str(e),
    exc_info=True
)
```

### Terraform Code Style

#### Resource Naming
```hcl
# Use descriptive, hierarchical naming
resource "docker_container" "rag_app" {
  name  = "${local.project_name}-app-${local.environment}"
  image = docker_image.app.latest

  # Consistent naming pattern: {project}-{component}-{environment}
}

resource "docker_network" "rag_network" {
  name = "${local.project_name}-network"
}
```

#### Variable Organization
```hcl
# variables.tf - Input variables
variable "environment" {
  description = "Deployment environment (dev/staging/prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod"
  }
}

variable "ollama_host" {
  description = "Ollama server host"
  type        = string
  default     = "localhost"
}

# locals.tf - Computed values
locals {
  project_name = "rag-agent"
  common_tags = {
    Project     = local.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
```

### Docker Best Practices

#### Multi-stage Builds
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing Guidelines

### Unit Tests Structure
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models import HealthStatus


class TestAPIIntegration:
    """Integration tests for API endpoints using pytest markers."""

    @pytest.fixture
    def client(self):
        """Test client for FastAPI application."""
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_query_endpoint_success(self, client):
        """Test successful query processing."""
        query_data = {"query": "What is machine learning?", "top_k": 3}

        from app.models import QueryResponse, QuerySource

        mock_response = QueryResponse(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI...",
            sources=[
                QuerySource(
                    document_id="doc1",
                    filename="ml_guide.pdf",
                    content_type="application/pdf",
                    chunk_text="Machine learning content...",
                    similarity_score=0.85,
                )
            ],
            confidence_score=0.85,
            processing_time=1.2,
            total_sources=1,
        )

        with patch("app.main.rag_agent") as mock_agent:
            async def mock_query(*args, **kwargs):
                return mock_response

            mock_agent.query = mock_query

            response = client.post("/query", json=query_data)

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "What is machine learning?"
            assert "answer" in data
            assert len(data["sources"]) == 1
            assert data["sources"][0]["similarity_score"] == 0.85
```

### Integration Tests
```python
import pytest
import httpx
from fastapi.testclient import TestClient

class TestAPIIntegration:
    def test_document_upload_flow(self, client, sample_pdf):
        """Test complete document upload and processing flow."""
        # Upload document
        with open(sample_pdf, "rb") as f:
            response = client.post("/documents/upload", files={"file": f})

        assert response.status_code == 200
        doc_id = response.json()["id"]

        # Query about uploaded document
        query_response = client.post("/query", json={
            "query": "What is the main topic?",
            "document_ids": [doc_id]
        })

        assert query_response.status_code == 200
        assert "answer" in query_response.json()
```

### Test Configuration and Markers
```python
# pytest.ini_options in pyproject.toml
[tool.pytest.ini_options]
minversion = "7.4"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
markers = [
    "unit: marks tests as unit tests",
    "integration: marks tests as integration tests",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "asyncio: marks tests as async",
]
```

### Test Coverage Requirements
- **Unit Tests**: 90%+ coverage for business logic
- **Integration Tests**: All API endpoints and workflows
- **Infrastructure Tests**: Terraform validation and container health
- **Performance Tests**: Response times and resource usage

## 🔒 Security Guidelines

### Environment Variables
```python
# config.py
from pydantic import BaseSettings, validator
import os

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://rag_user:rag_pass@localhost:5432/rag_db"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_embed_model: str = "embeddinggemma:latest"

    # Security
    secret_key: str
    api_key_salt: str = "your-secret-salt-here"

    # File handling
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = [".pdf", ".txt", ".jpg", ".png"]

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator('secret_key', pre=True, always=True)
    def generate_secret_key(cls, v):
        return v or os.urandom(32).hex()
```

### Input Validation
```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import re

class DocumentQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: Optional[List[str]] = []
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = {}

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        # Prevent injection attacks
        if re.search(r'[<>]', v):
            raise ValueError('Query contains invalid characters')
        return v.strip()

class DocumentUpload(BaseModel):
    filename: str
    content_type: str
    size: int

    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = [
            'application/pdf',
            'text/plain',
            'image/jpeg',
            'image/png'
        ]
        if v not in allowed_types:
            raise ValueError(f'Content type {v} not allowed')
        return v
```

## 📚 Documentation Standards

### API Documentation
```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any

router = APIRouter()

@router.post(
    "/documents/upload",
    summary="Upload and process a document",
    description="""
    Upload a document (PDF, text, or image) for processing and indexing.

    The document will be:
    1. Validated for type and size
    2. Processed for text extraction
    3. Split into chunks with overlap
    4. Embedded using Ollama
    5. Stored in vector database

    Supported formats: PDF, TXT, JPG, PNG
    Maximum size: 50MB
    """,
    response_model=DocumentResponse,
    responses={
        200: {"description": "Document processed successfully"},
        400: {"description": "Invalid file format or size"},
        500: {"description": "Processing failed"}
    }
)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    overlap: int = 200
) -> DocumentResponse:
    """Upload and process a document for the RAG system."""
    # Implementation...
```

### Code Documentation
```python
class RAGAgent:
    """
    Hybrid RAG agent combining LangChain orchestration with LlamaIndex indexing.

    This agent provides conversational AI capabilities with document-grounded
    responses using local Ollama models for both generation and embeddings.

    Attributes:
        ollama_client: Client for interacting with Ollama API
        vector_store: PostgreSQL vector store for document embeddings
        memory: Redis-backed conversation memory
        langchain_agent: LangChain agent for tool orchestration
        llama_index: LlamaIndex for document indexing and retrieval
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        vector_store: VectorStore,
        memory: RedisMemory
    ):
        """
        Initialize the RAG agent with required components.

        Args:
            ollama_client: Configured Ollama client instance
            vector_store: Vector store for document embeddings
            memory: Memory system for conversation context
        """
        self.ollama_client = ollama_client
        self.vector_store = vector_store
        self.memory = memory
        self._setup_agent()

    async def query(
        self,
        query: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a query using the hybrid RAG system.

        Retrieves relevant documents, generates a response using the language
        model, and maintains conversation context.

        Args:
            query: User's question or request
            context: Optional additional context documents
            **kwargs: Additional parameters for fine-tuning

        Returns:
            Dictionary containing:
            - answer: Generated response
            - sources: List of source documents
            - confidence: Response confidence score
            - metadata: Additional processing information

        Raises:
            RAGError: If query processing fails
        """
        # Implementation...
```

## 🚀 Deployment Guidelines

### Local Development Setup
```bash
# Clone repository
git clone <repository-url>
cd rag-agent-terraform

# Set up Python environment
python -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt

# Start infrastructure
cd terraform
terraform init
terraform apply

# Verify services are running
docker ps

# Start application
cd ../src
python -m uvicorn app.main:app --reload
```

### Production Deployment
```bash
# Use production configuration
export ENVIRONMENT=production
export SECRET_KEY=$(openssl rand -hex 32)

# Build and deploy containers
make deploy

# Run health checks
curl http://localhost:8000/health

# Monitor logs
docker-compose logs -f app
```

## 🔧 Development Workflow

### Feature Development
1. Create feature branch: `git checkout -b feature/document-chunking`
2. Implement changes with tests
3. Run full test suite: `make test`
4. Format code: `make lint`
5. Commit with clear message: `git commit -m "feat: implement intelligent document chunking"`
6. Create pull request with description

### Code Review Checklist
- [ ] Tests pass and coverage maintained
- [ ] Code follows style guidelines
- [ ] Type hints added for new functions
- [ ] Documentation updated
- [ ] Security considerations addressed
- [ ] Performance implications reviewed
- [ ] Error handling implemented

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## 📊 Monitoring & Logging

### Structured Logging
```python
import logging
import json
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_query(self, query: str, response: Dict[str, Any], duration: float):
        """Log RAG query with structured data."""
        self.logger.info(
            "RAG query processed",
            extra={
                "event": "rag_query",
                "query_length": len(query),
                "response_length": len(response.get("answer", "")),
                "sources_count": len(response.get("sources", [])),
                "duration_ms": duration,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log errors with context."""
        self.logger.error(
            f"Error occurred: {str(error)}",
            extra={
                "event": "error",
                "error_type": type(error).__name__,
                "context": context,
                "timestamp": datetime.utcnow().isoformat()
            },
            exc_info=True
        )
```

### Health Checks
```python
from fastapi import APIRouter, Depends
from app.dependencies import get_vector_store, get_ollama_client

router = APIRouter()

@router.get("/health")
async def health_check(
    vector_store = Depends(get_vector_store),
    ollama_client = Depends(get_ollama_client)
):
    """Comprehensive health check for all system components."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    # Check vector store
    try:
        await vector_store.health_check()
        health_status["services"]["vector_store"] = "healthy"
    except Exception as e:
        health_status["services"]["vector_store"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check Ollama
    try:
        await ollama_client.health_check()
        health_status["services"]["ollama"] = "healthy"
    except Exception as e:
        health_status["services"]["ollama"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check Redis
    try:
        await redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    return health_status
```

## 🎯 Performance Guidelines

### Optimization Strategies
- **Batch Processing**: Process multiple documents/embeddings concurrently
- **Caching**: Cache frequent queries and embeddings in Redis
- **Chunking**: Intelligent text chunking with semantic boundaries
- **Async Operations**: Use async/await for I/O operations
- **Resource Limits**: Implement rate limiting and resource quotas

### Benchmarking
```python
import time
import asyncio
from typing import List, Dict, Any

class RAGBenchmark:
    def __init__(self, rag_agent: RAGAgent):
        self.agent = rag_agent

    async def benchmark_query(
        self,
        queries: List[str],
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark RAG query performance."""
        results = []

        for query in queries:
            query_times = []

            for _ in range(iterations):
                start_time = time.time()
                response = await self.agent.query(query)
                end_time = time.time()

                query_times.append(end_time - start_time)
                results.append({
                    "query": query,
                    "response_length": len(response["answer"]),
                    "sources_count": len(response["sources"]),
                    "duration": end_time - start_time
                })

            # Calculate statistics
            avg_time = sum(query_times) / len(query_times)
            p95_time = sorted(query_times)[int(len(query_times) * 0.95)]

            print(f"Query: {query}")
            print(f"Average time: {avg_time:.3f}s")
            print(f"P95 time: {p95_time:.3f}s")
            print("---")

        return {
            "results": results,
            "summary": {
                "total_queries": len(results),
                "average_response_time": sum(r["duration"] for r in results) / len(results),
                "p95_response_time": sorted([r["duration"] for r in results])[int(len(results) * 0.95)]
            }
        }
```

---

This AGENTS.md file provides comprehensive guidelines for developing and maintaining the RAG Agent Terraform project. Follow these guidelines to ensure consistent, maintainable, and high-quality code across all components of the system.</content>
<parameter name="filePath">AGENTS.md