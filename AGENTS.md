# AGENTS.md - Agentic Coding Guidelines for RAG Agent Terraform Project

## 🚀 Project Overview

**Multi-agent RAG system** with LangGraph orchestration, FastAPI backend, PostgreSQL with pgvector, and local Ollama models.

## 🛠️ Development Commands

### Quick Start
```bash
make setup    # Setup Python environment
make build    # Build Docker images
make up       # Deploy services with Terraform
make dev      # Start development server
```

### Testing & Quality
```bash
make test     # Run all tests with coverage
make lint     # Format code (black, isort, flake8, mypy)

# Single test execution
cd src && python -m pytest tests/test_api.py::TestAPIIntegration::test_health_endpoint -v
cd src && python -m pytest tests/test_vector_store.py::TestVectorStore::test_similarity_search -v
cd src && python -m pytest tests/test_multi_agent.py -k "test_agent_routing" -v
```

### Code Quality
- **Black**: 88 character line length, Python 3.11+
- **Flake8**: 100 chars (E501 ignored), strict mode
- **isort**: Black-compatible, 88 chars, multi-line output 3
- **MyPy**: Strict type checking (disallow_untyped_defs, no_implicit_optional)

## 📝 Code Style Guidelines

### Import Organization
```python
# Standard library (alphabetical)
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party (alphabetical)
import httpx
from fastapi import FastAPI, HTTPException
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph
import psycopg2
import redis
import structlog

# Local (module hierarchy)
from app.config import settings
from app.models import Document, Query
from app.vector_store import VectorStore
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `DockerMultiAgentRAGState`)
- **Functions/Methods**: `snake_case` (e.g., `process_document`, `get_embedding`)
- **Variables**: `snake_case` (e.g., `agent_tasks`, `workflow_results`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_UPLOAD_SIZE`)
- **Private members**: `_snake_case` (e.g., `_ensure_connection`)

### Type Hints
```python
def process_document(document_path: Path, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Process a document and return chunks with metadata."""

async def similarity_search(query_vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Perform vector similarity search with optional filters."""
```

### Error Handling
```python
import logging
from fastapi import HTTPException
from app.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

def load_document(file_path: Path) -> Document:
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        content = file_path.read_text(encoding='utf-8')
        if not content.strip():
            raise ValueError("Document is empty")
        return Document(content=content, path=file_path)
    except FileNotFoundError as e:
        logger.error(f"Document loading failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error loading document: {e}")
        raise DocumentProcessingError(f"Failed to load document: {str(e)}")
```

### Custom Exceptions
```python
class RAGException(Exception):
    """Base exception for RAG system errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}

class DocumentProcessingError(RAGException):
    """Raised when document processing fails."""
    pass
```

### Async/Await Patterns
```python
class VectorStore:
    async def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts concurrently."""
        tasks = [self._embed_single(text) for text in texts]
        return await asyncio.gather(*tasks)

@router.post("/agents/query")
async def multi_agent_query(request: QueryRequest):
    """Multi-Agent query with LangGraph."""
    try:
        initial_state = create_initial_state(query=request.query, user_id="api_user", user_level="standard")
        config = {"configurable": {"thread_id": f"query_{datetime.utcnow().isoformat()}_{uuid.uuid4().hex[:8]}", "thread_ts": "latest"}}
        result = await multi_agent_graph.ainvoke(initial_state, config=config)
        return AgentQueryResponse(
            query=request.query,
            answer=result.get("final_answer", "No response generated"),
            sources=result.get("sources", []),
            confidence_score=result.get("confidence_score", 0.0),
            processing_time=result.get("processing_time", 0.0),
            total_sources=len(result.get("sources", [])),
        )
    except Exception as e:
        logger.error(f"Multi-agent query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-agent query failed: {str(e)}")
```

### Dependency Injection Patterns
```python
from fastapi import Depends

async def get_vector_store() -> VectorStore:
    return VectorStore(settings.database_url)

async def get_ollama_client() -> OllamaClient:
    return OllamaClient(base_url=settings.ollama_base_url)

@router.post("/query")
async def query_rag(
    request: QueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> QueryResponse:
    """Query the RAG system with dependency injection."""
    # Implementation uses injected dependencies
```

## 🧪 Testing Guidelines

### Unit Tests Structure
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app

class TestAPIIntegration:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_health_endpoint_success(self, client):
        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.vector_store.health_check = AsyncMock()
            mock_agent.ollama_client.health_check = AsyncMock()
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
```

### Test Configuration
- pytest with asyncio support
- Markers: `unit`, `integration`, `slow`, `database`, `redis`, `asyncio`
- Coverage reporting enabled

## 🔒 Security Guidelines

### Environment Variables
```python
from pydantic import BaseSettings, validator
import os

class Settings(BaseSettings):
    database_url: str = "postgresql://rag_user:rag_pass@localhost:5432/rag_db"
    redis_url: str = "redis://localhost:6379"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_embed_model: str = "embeddinggemma:latest"
    secret_key: str
    api_key_salt: str = "your-secret-salt-here"
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
        if re.search(r'[<>]', v):
            raise ValueError('Query contains invalid characters')
        return v.strip()
```

## 📚 Documentation Standards

### API Documentation
```python
@router.post(
    "/documents/upload",
    summary="Upload and process a document",
    description="Upload document for processing and indexing. Supported: PDF, TXT, JPG, PNG. Max 50MB.",
    response_model=DocumentResponse,
    responses={
        200: {"description": "Document processed successfully"},
        400: {"description": "Invalid file format or size"},
        500: {"description": "Processing failed"}
    }
)
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    """Upload and process a document for the RAG system."""
```

### Code Documentation
```python
class RAGAgent:
    """Hybrid RAG agent with LangChain orchestration and LlamaIndex indexing."""

    def __init__(self, ollama_client: OllamaClient, vector_store: VectorStore, memory: RedisMemory):
        """Initialize RAG agent with required components."""
        self.ollama_client = ollama_client
        self.vector_store = vector_store
        self.memory = memory
        self._setup_agent()

    async def query(self, query: str, context: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Process query using hybrid RAG system."""
```

## 🔧 Development Workflow

### Feature Development
1. Create feature branch: `git checkout -b feature/document-chunking`
2. Implement changes with tests
3. Run tests: `make test`
4. Format code: `make lint`
5. Commit: `git commit -m "feat: implement intelligent document chunking"`

### Commit Message Format
```
type(scope): description
[optional body]
[optional footer]
```
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

This AGENTS.md provides guidelines for developing the RAG Agent Terraform project.