"""
Shared test fixtures and utilities for the RAG Agent test suite.
Provides common mocks, test data, and utilities used across all test modules.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from app.models import (
    OllamaEmbedResponse,
    OllamaGenerateResponse,
    QueryResponse,
)


# Mock Data Fixtures
@pytest.fixture
def sample_document_content():
    """Sample document content for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn from data without being explicitly programmed. The field has grown
    rapidly in recent years with applications in image recognition, natural language
    processing, and predictive analytics.

    Deep learning, a subset of machine learning, uses neural networks with multiple
    layers to model complex patterns in data. These networks can automatically learn
    hierarchical features from raw input data.
    """


@pytest.fixture
def sample_document_metadata():
    """Sample document metadata for testing."""
    return {
        "filename": "ml_guide.txt",
        "content_type": "text/plain",
        "size": 1024,
        "word_count": 150,
        "page_count": 1,
        "upload_time": "2026-01-13T10:00:00",
        "checksum": "abc123",
    }


@pytest.fixture
def sample_embedding():
    """Sample vector embedding for testing (768 dimensions like nomic-embed-text)."""
    np.random.seed(42)  # For reproducible tests
    return np.random.normal(0, 1, 768).tolist()


@pytest.fixture
def sample_chunks():
    """Sample document chunks with embeddings."""
    return [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Machine learning is a subset of artificial intelligence.",
            "chunk_index": 0,
            "total_chunks": 3,
            "embedding": [0.1] * 768,
            "metadata": {"chunk_size": 50, "word_count": 8},
        },
        {
            "id": "chunk-2",
            "document_id": "doc-1",
            "content": "The field has grown rapidly in recent years.",
            "chunk_index": 1,
            "total_chunks": 3,
            "embedding": [0.2] * 768,
            "metadata": {"chunk_size": 45, "word_count": 7},
        },
    ]


@pytest.fixture
def sample_query_response():
    """Sample query response for testing."""
    return QueryResponse(
        query="What is machine learning?",
        answer="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        sources=[
            {
                "document_id": "doc-1",
                "filename": "ml_guide.txt",
                "content_type": "text/plain",
                "chunk_text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "similarity_score": 0.85,
                "metadata": {"chunk_index": 0},
            }
        ],
        confidence_score=0.85,
        processing_time=1.2,
        total_sources=1,
    )


# Mock Service Fixtures
@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    client = AsyncMock()

    # Mock embed_batch response
    embed_response = OllamaEmbedResponse(
        model="nomic-embed-text",
        embedding=[0.1, 0.2, 0.3] * 256,  # 768 dimensions
        total_duration=1000000,
        load_duration=500000,
        prompt_eval_count=10,
    )
    client.embed_batch.return_value = [embed_response]

    # Mock generate response
    generate_response = OllamaGenerateResponse(
        model="llama3.2:latest",
        created_at="2026-01-13T18:00:00Z",
        response="Machine learning is a powerful technology.",
        done=True,
        total_duration=2000000,
        load_duration=500000,
        prompt_eval_count=15,
        eval_count=25,
        eval_duration=1500000,
    )
    client.generate.return_value = generate_response

    # Mock health check
    client.health_check.return_value = True

    return client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = AsyncMock()

    # Mock context manager
    store.__aenter__ = AsyncMock(return_value=store)
    store.__aexit__ = AsyncMock(return_value=None)

    # Mock methods
    store.store_chunks.return_value = 2
    store.similarity_search.return_value = [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "Sample content",
            "chunk_index": 0,
            "total_chunks": 1,
            "metadata": {},
            "filename": "test.txt",
            "content_type": "text/plain",
            "similarity_score": 0.85,
        }
    ]
    store.list_documents.return_value = [
        {
            "id": "doc-1",
            "filename": "test.txt",
            "content_type": "text/plain",
            "size": 1024,
            "uploaded_at": "2026-01-13T10:00:00",
            "status": "completed",
            "chunks_count": 2,
        }
    ]
    store.get_document.return_value = {
        "id": "doc-1",
        "filename": "test.txt",
        "content_type": "text/plain",
        "size": 1024,
        "uploaded_at": "2026-01-13T10:00:00",
        "status": "completed",
        "chunks_count": 2,
    }
    store.delete_document.return_value = True
    store.health_check.return_value = True

    return store


@pytest.fixture
def mock_redis_memory():
    """Mock Redis memory for testing."""
    memory = AsyncMock()

    # Mock caching methods
    memory.get_cached_embedding.return_value = None
    memory.cache_embedding.return_value = None
    memory.get_cached_query_result.return_value = None
    memory.cache_query_result.return_value = None

    # Mock conversation methods
    memory.get_conversation.return_value = None
    memory.store_conversation.return_value = None
    memory.update_conversation.return_value = None

    # Mock stats
    memory.get_stats.return_value = {
        "total_keys": 10,
        "used_memory": 1024,
        "connections": 1,
    }

    memory.health_check.return_value = True

    return memory


@pytest.fixture
def mock_rag_agent(mock_vector_store, mock_redis_memory, mock_ollama_client):
    """Mock RAG agent with all dependencies."""
    with (
        patch("app.rag_agent.VectorStore", return_value=mock_vector_store),
        patch("app.rag_agent.AgentMemory", return_value=mock_redis_memory),
        patch("app.rag_agent.OllamaClient", return_value=mock_ollama_client),
    ):
        from app.rag_agent import RAGAgent

        agent = RAGAgent()
        return agent


# Test Utilities
@pytest.fixture
def temp_test_file(tmp_path, sample_document_content):
    """Create a temporary test file."""
    test_file = tmp_path / "test_document.txt"
    test_file.write_text(sample_document_content)
    return test_file


@pytest.fixture
def temp_text_content():
    """Content for temporary text file."""
    return """Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.

Key concepts include:
- Supervised learning
- Unsupervised learning
- Reinforcement learning
- Neural networks
- Deep learning

Applications span image recognition, natural language processing, recommendation systems, and autonomous vehicles."""


@pytest.fixture
def temp_text_file(tmp_path, temp_text_content):
    """Create a temporary text file for testing."""
    test_file = tmp_path / "test_ml_fundamentals.txt"
    test_file.write_text(temp_text_content)
    return str(test_file)


@pytest.fixture
def mock_file_upload():
    """Mock file upload for testing."""
    file_mock = Mock()
    file_mock.filename = "test_document.txt"
    file_mock.file = Mock()
    file_mock.file.read.return_value = b"This is test content for upload."
    return file_mock


# Async Test Utilities
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest Configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "asyncio: marks tests as async")


# Custom Assertions
def assert_dict_contains_subset(subset: Dict[str, Any], superset: Dict[str, Any]):
    """Assert that all key-value pairs in subset are present in superset."""
    for key, value in subset.items():
        assert key in superset, f"Key '{key}' not found in superset"
        assert superset[key] == value, (
            f"Value mismatch for key '{key}': expected {value}, got {superset[key]}"
        )


def assert_embedding_valid(embedding: List[float], expected_dim: int = 768):
    """Assert that an embedding is valid."""
    assert isinstance(embedding, list), "Embedding must be a list"
    assert len(embedding) == expected_dim, (
        f"Embedding must have {expected_dim} dimensions"
    )
    assert all(isinstance(x, (int, float)) for x in embedding), (
        "All embedding values must be numeric"
    )


# Test Data Generators
def generate_test_embedding(seed: int = 42, dim: int = 768) -> List[float]:
    """Generate a deterministic test embedding."""
    np.random.seed(seed)
    return np.random.normal(0, 1, dim).tolist()


def generate_test_chunks(document_id: str, num_chunks: int = 3) -> List[Dict[str, Any]]:
    """Generate test document chunks."""
    chunks = []
    for i in range(num_chunks):
        chunks.append(
            {
                "id": f"chunk-{i + 1}",
                "document_id": document_id,
                "content": f"Test content chunk {i + 1}",
                "chunk_index": i,
                "total_chunks": num_chunks,
                "embedding": generate_test_embedding(seed=i, dim=768),
                "metadata": {"chunk_size": 20, "word_count": 4},
            }
        )
    return chunks


def generate_test_documents(count: int = 3) -> List[Dict[str, Any]]:
    """Generate test documents."""
    documents = []
    for i in range(count):
        documents.append(
            {
                "id": f"doc-{i + 1}",
                "filename": f"test_doc_{i + 1}.txt",
                "content_type": "text/plain",
                "size": 1024 + i * 100,
                "uploaded_at": f"2026-01-13T10:0{i}:00",
                "status": "completed",
                "chunks_count": 2 + i,
            }
        )
    return documents
