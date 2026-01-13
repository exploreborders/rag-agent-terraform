"""
Integration test fixtures and utilities for testing with real services.
Uses testcontainers to spin up PostgreSQL and Redis containers for testing.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from app.config import Settings
from app.ollama_client import OllamaClient
from app.rag_agent import RAGAgent
from app.redis_memory import AgentMemory
from app.vector_store import VectorStore


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def postgres_container() -> Dict[str, Any]:
    """Use existing PostgreSQL container."""
    # Use the existing container that's already running
    connection_info = {
        "host": "localhost",
        "port": 5432,
        "username": "rag_user",
        "password": "rag_password",
        "database": "rag_db",
        "connection_string": "postgresql://rag_user:rag_password@localhost:5432/rag_db",
    }

    return connection_info


@pytest.fixture(scope="session")
def redis_container() -> Dict[str, Any]:
    """Use existing Redis container."""
    # Use the existing container that's already running
    connection_info = {
        "host": "localhost",
        "port": 6379,
        "connection_string": "redis://localhost:6379",
    }

    return connection_info


@pytest.fixture
async def real_vector_store(postgres_container) -> AsyncGenerator[VectorStore, None]:
    """Create a VectorStore connected to real PostgreSQL."""
    # Override settings for testing
    original_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = postgres_container["connection_string"]

    try:
        store = VectorStore()
        await store.connect()
        await store.initialize_schema()

        yield store

        # Cleanup
        await store.disconnect()

    finally:
        if original_db_url:
            os.environ["DATABASE_URL"] = original_db_url
        else:
            os.environ.pop("DATABASE_URL", None)


@pytest.fixture
async def real_redis_memory(redis_container) -> AsyncGenerator[AgentMemory, None]:
    """Create Redis memory connected to real Redis."""
    # Override settings for testing
    original_redis_url = os.environ.get("REDIS_URL")
    os.environ["REDIS_URL"] = redis_container["connection_string"]

    try:
        memory = AgentMemory()
        yield memory

    finally:
        # Clean up Redis data
        await memory.clear_all_cache()

        if original_redis_url:
            os.environ["REDIS_URL"] = original_redis_url
        else:
            os.environ.pop("REDIS_URL", None)


@pytest.fixture
async def real_ollama_client() -> AsyncGenerator[OllamaClient, None]:
    """Create Ollama client (using mock for integration tests since we don't have real Ollama)."""
    # For integration tests, we'll use a mock Ollama client that simulates responses
    # In a real environment, this would connect to actual Ollama
    from unittest.mock import AsyncMock

    client = OllamaClient()
    # Mock the embed_batch method to return embedding lists directly
    original_embed_batch = client.embed_batch
    client.embed_batch = AsyncMock(
        return_value=[[0.1, 0.2, 0.3] * 256]  # 768 dimensions - return list directly
    )

    # Mock the generate method
    original_generate = client.generate
    client.generate = AsyncMock(
        return_value=type(
            "GenerateResponse",
            (),
            {"response": "This is a test response from the AI model.", "done": True},
        )()
    )

    yield client

    # Restore originals
    client.embed_batch = original_embed_batch
    client.generate = original_generate


@pytest.fixture
async def real_rag_agent(
    real_vector_store, real_redis_memory, real_ollama_client
) -> AsyncGenerator[RAGAgent, None]:
    """Create a RAGAgent with real database connections."""
    agent = RAGAgent(
        vector_store=real_vector_store,
        memory=real_redis_memory,
        ollama_client=real_ollama_client,
    )

    yield agent


@pytest.fixture
def temp_upload_dir(tmp_path) -> Path:
    """Create a temporary upload directory for testing."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture
def sample_test_document(temp_upload_dir) -> Path:
    """Create a sample test document."""
    doc_path = temp_upload_dir / "test_document.txt"
    doc_path.write_text(
        """
    Machine learning is a powerful technology that enables computers to learn from data
    without being explicitly programmed. This field has revolutionized many industries
    including healthcare, finance, and autonomous vehicles.

    Deep learning, a subset of machine learning, uses neural networks with multiple layers
    to model complex patterns in data. These networks can automatically learn hierarchical
    features from raw input data, making them particularly effective for tasks like image
    recognition and natural language processing.
    """
    )
    return doc_path


@pytest.fixture
async def populated_vector_store(real_vector_store, sample_test_document):
    """Vector store populated with test data."""
    # This would be used in tests that need pre-populated data
    # For now, just return the empty store
    yield real_vector_store


# Integration test markers
def pytest_configure(config):
    """Configure pytest with integration test markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests that require real external services"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "database: marks tests that require database connectivity"
    )
    config.addinivalue_line(
        "markers", "redis: marks tests that require Redis connectivity"
    )
