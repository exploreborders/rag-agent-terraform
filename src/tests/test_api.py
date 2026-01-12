"""Integration tests for FastAPI application."""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from app.main import app
from app.models import HealthStatus


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Test client for FastAPI application."""
        return TestClient(app)

    @pytest.fixture
    async def async_client(self):
        """Async test client for FastAPI application."""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        """Test health check endpoint."""
        # Mock the RAG agent health check
        with patch("app.main.rag_agent") as mock_agent:
            mock_health = HealthStatus(
                status="healthy",
                timestamp="2024-01-01T00:00:00Z",
                services={
                    "ollama": "healthy",
                    "vector_store": "healthy",
                    "redis": "healthy",
                },
            )
            mock_agent.health_check.return_value = mock_health

            response = await async_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "services" in data
            assert data["services"]["ollama"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_endpoint_degraded(self, async_client):
        """Test health check with degraded services."""
        with patch("app.main.rag_agent") as mock_agent:
            mock_health = HealthStatus(
                status="degraded",
                timestamp="2024-01-01T00:00:00Z",
                services={
                    "ollama": "unhealthy",
                    "vector_store": "healthy",
                    "redis": "healthy",
                },
            )
            mock_agent.health_check.return_value = mock_health

            response = await async_client.get("/health")

            assert response.status_code == 503  # Service Unavailable
            data = response.json()
            assert data["status"] == "degraded"
            assert data["services"]["ollama"] == "unhealthy"

    def test_health_endpoint_no_agent(self, client):
        """Test health check when agent is not initialized."""
        # Temporarily set rag_agent to None
        import app.main

        original_agent = app.main.rag_agent
        app.main.rag_agent = None

        try:
            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert "not initialized" in data["detail"].lower()
        finally:
            app.main.rag_agent = original_agent

    @pytest.mark.asyncio
    async def test_query_endpoint_success(self, async_client):
        """Test successful query processing."""
        query_data = {"query": "What is machine learning?", "top_k": 3}

        mock_response = {
            "query": "What is machine learning?",
            "answer": "Machine learning is a subset of AI...",
            "sources": [
                {
                    "document_id": "doc1",
                    "filename": "ml_guide.pdf",
                    "content_type": "application/pdf",
                    "chunk_text": "Machine learning content...",
                    "similarity_score": 0.85,
                }
            ],
            "confidence_score": 0.85,
            "processing_time": 1.2,
            "total_sources": 1,
        }

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.query.return_value = mock_response

            response = await async_client.post("/query", json=query_data)

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "What is machine learning?"
            assert "answer" in data
            assert len(data["sources"]) == 1
            assert data["sources"][0]["similarity_score"] == 0.85

    @pytest.mark.asyncio
    async def test_query_endpoint_empty_query(self, async_client):
        """Test query endpoint with empty query."""
        query_data = {"query": ""}

        response = await async_client.post("/query", json=query_data)

        assert response.status_code == 400
        data = response.json()
        assert "cannot be empty" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_query_endpoint_with_session(self, async_client):
        """Test query with conversation session."""
        query_data = {"query": "Tell me more about this", "session_id": "session_123"}

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.query.return_value = {
                "query": "Tell me more about this",
                "answer": "Based on our conversation...",
                "sources": [],
                "session_id": "session_123",
            }

            response = await async_client.post("/query", json=query_data)

            assert response.status_code == 200
            mock_agent.query.assert_called_once()
            call_args = mock_agent.query.call_args[1]
            assert call_args["session_id"] == "session_123"

    @pytest.mark.asyncio
    async def test_list_documents(self, async_client):
        """Test listing documents."""
        mock_documents = [
            {
                "id": "doc1",
                "filename": "test.pdf",
                "content_type": "application/pdf",
                "size": 1024000,
                "upload_time": "2024-01-01T00:00:00Z",
            },
            {
                "id": "doc2",
                "filename": "guide.txt",
                "content_type": "text/plain",
                "size": 512000,
                "upload_time": "2024-01-02T00:00:00Z",
            },
        ]

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.list_documents.return_value = mock_documents

            response = await async_client.get("/documents")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["filename"] == "test.pdf"
            assert data[1]["filename"] == "guide.txt"

    @pytest.mark.asyncio
    async def test_list_documents_with_pagination(self, async_client):
        """Test document listing with pagination."""
        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.list_documents.return_value = []

            response = await async_client.get("/documents?limit=5&offset=10")

            assert response.status_code == 200
            mock_agent.list_documents.assert_called_once_with(limit=5, offset=10)

    @pytest.mark.asyncio
    async def test_get_document_success(self, async_client):
        """Test getting a specific document."""
        document_id = "test_doc_123"
        mock_document = {
            "id": document_id,
            "filename": "test.pdf",
            "content_type": "application/pdf",
            "size": 1024000,
            "upload_time": "2024-01-01T00:00:00Z",
        }

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.get_document.return_value = mock_document

            response = await async_client.get(f"/documents/{document_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == document_id
            assert data["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, async_client):
        """Test getting a non-existent document."""
        document_id = "nonexistent_doc"

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.get_document.return_value = None

            response = await async_client.get(f"/documents/{document_id}")

            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_delete_document_success(self, async_client):
        """Test successful document deletion."""
        document_id = "test_doc_123"

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.delete_document.return_value = True

            response = await async_client.delete(f"/documents/{document_id}")

            assert response.status_code == 200
            data = response.json()
            assert "deleted successfully" in data["message"]
            assert data["document_id"] == document_id

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, async_client):
        """Test deleting a non-existent document."""
        document_id = "nonexistent_doc"

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.delete_document.return_value = False

            response = await async_client.delete(f"/documents/{document_id}")

            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_stats(self, async_client):
        """Test getting system statistics."""
        mock_stats = {
            "total_documents": 5,
            "total_chunks": 1250,
            "cache_stats": {"hits": 100, "misses": 25},
            "ollama_available": True,
            "vector_store_healthy": True,
            "memory_healthy": True,
        }

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.get_stats.return_value = mock_stats

            response = await async_client.get("/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["total_documents"] == 5
            assert data["total_chunks"] == 1250
            assert data["ollama_available"] is True

    @pytest.mark.asyncio
    async def test_clear_cache(self, async_client):
        """Test clearing system caches."""
        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.clear_cache.return_value = True

            response = await async_client.post("/cache/clear")

            assert response.status_code == 200
            data = response.json()
            assert "cleared successfully" in data["message"]

    @pytest.mark.asyncio
    async def test_clear_cache_failure(self, async_client):
        """Test cache clearing failure."""
        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.clear_cache.return_value = False

            response = await async_client.post("/cache/clear")

            assert response.status_code == 500
            data = response.json()
            assert "failed" in data["detail"].lower()

    def test_cors_headers(self, client):
        """Test CORS headers are set correctly."""
        response = client.options("/health")
        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

    @pytest.mark.asyncio
    async def test_error_handling_rag_error(self, async_client):
        """Test RAG agent error handling."""
        from app.rag_agent import RAGAgentError

        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.query.side_effect = RAGAgentError("Query processing failed")

            query_data = {"query": "test query"}
            response = await async_client.post("/query", json=query_data)

            assert response.status_code == 422
            data = response.json()
            assert "RAG Agent Error" in data["error"]
            assert "Query processing failed" in data["message"]

    @pytest.mark.asyncio
    async def test_error_handling_unexpected_error(self, async_client):
        """Test unexpected error handling."""
        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.query.side_effect = Exception("Unexpected error")

            query_data = {"query": "test query"}
            response = await async_client.post("/query", json=query_data)

            assert response.status_code == 500
            data = response.json()
            assert "Internal Server Error" in data["error"]
