"""Integration tests for FastAPI application."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import HealthStatus


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Test client for FastAPI application."""
        return TestClient(app)

    def test_health_endpoint(self, client):
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

            # Mock async method to return a coroutine
            async def mock_health_check():
                return mock_health

            mock_agent.health_check = mock_health_check

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "services" in data
            assert data["services"]["ollama"] == "healthy"

    def test_health_endpoint_degraded(self, client):
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

            # Mock async method to return a coroutine
            async def mock_health_check():
                return mock_health

            mock_agent.health_check = mock_health_check

            response = client.get("/health")

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

    async def test_query_endpoint_empty_query(self, client):
        """Test query endpoint with empty query."""
        # Mock agent to avoid 503
        with patch("app.main.rag_agent") as mock_agent:
            query_data = {"query": ""}

            response = client.post("/query", json=query_data)

            assert response.status_code == 422  # Pydantic validation error
            data = response.json()
            assert "query" in str(data).lower()  # Should mention query field

    async def test_query_endpoint_with_session(self, client):
        """Test query with conversation session."""
        query_data = {"query": "Tell me more about this"}

        with patch("app.main.rag_agent") as mock_agent:
            from unittest.mock import AsyncMock

            from app.models import QueryResponse

            mock_agent.query = AsyncMock(
                return_value=QueryResponse(
                    query="Tell me more about this",
                    answer="Based on our conversation...",
                    sources=[],
                    session_id="session_123",
                )
            )

            response = client.post("/query?session_id=session_123", json=query_data)

            assert response.status_code == 200
            mock_agent.query.assert_called_once()
            # Verify session_id was passed (check call arguments)
            call_args, call_kwargs = mock_agent.query.call_args
            assert call_kwargs.get("session_id") == "session_123"

    async def test_list_documents(self, client):
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

            async def mock_list_documents(*args, **kwargs):
                return mock_documents

            mock_agent.list_documents = mock_list_documents

            response = client.get("/documents")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["filename"] == "test.pdf"
            assert data[1]["filename"] == "guide.txt"

    async def test_list_documents_with_pagination(self, client):
        """Test document listing with pagination."""
        with patch("app.main.rag_agent") as mock_agent:
            from unittest.mock import AsyncMock

            mock_agent.list_documents = AsyncMock(return_value=[])

            response = client.get("/documents?limit=5&offset=10")

            assert response.status_code == 200
            mock_agent.list_documents.assert_called_once_with(limit=5, offset=10)

    async def test_get_document_success(self, client):
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

            async def mock_get_document(*args, **kwargs):
                return mock_document

            mock_agent.get_document = mock_get_document

            response = client.get(f"/documents/{document_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == document_id
            assert data["filename"] == "test.pdf"

    async def test_get_document_not_found(self, client):
        """Test getting a non-existent document."""
        document_id = "nonexistent_doc"

        with patch("app.main.rag_agent") as mock_agent:

            async def mock_get_document(*args, **kwargs):
                return None

            mock_agent.get_document = mock_get_document

            response = client.get(f"/documents/{document_id}")

            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    async def test_delete_document_success(self, client):
        """Test successful document deletion."""
        document_id = "test_doc_123"

        with patch("app.main.rag_agent") as mock_agent:

            async def mock_delete_document(*args, **kwargs):
                return True

            mock_agent.delete_document = mock_delete_document

            response = client.delete(f"/documents/{document_id}")

            assert response.status_code == 200
            data = response.json()
            assert "deleted successfully" in data["message"]
            assert data["document_id"] == document_id

    async def test_delete_document_not_found(self, client):
        """Test deleting a non-existent document."""
        document_id = "nonexistent_doc"

        with patch("app.main.rag_agent") as mock_agent:

            async def mock_delete_document(*args, **kwargs):
                return False

            mock_agent.delete_document = mock_delete_document

            response = client.delete(f"/documents/{document_id}")

            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    async def test_get_stats(self, client):
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

            async def mock_get_stats(*args, **kwargs):
                return mock_stats

            mock_agent.get_stats = mock_get_stats

            response = client.get("/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["total_documents"] == 5
            assert data["total_chunks"] == 1250
            assert data["ollama_available"] is True

    async def test_clear_cache(self, client):
        """Test clearing system caches."""
        with patch("app.main.rag_agent") as mock_agent:

            async def mock_clear_cache(*args, **kwargs):
                return True

            mock_agent.clear_cache = mock_clear_cache

            response = client.post("/cache/clear")

            assert response.status_code == 200
            data = response.json()
            assert "cleared successfully" in data["message"]

    async def test_clear_cache_failure(self, client):
        """Test cache clearing failure."""
        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.clear_cache.return_value = False

            response = client.post("/cache/clear")

            assert response.status_code == 500
            data = response.json()
            assert "failed" in data["detail"].lower()

    def test_cors_headers(self, client):
        """Test CORS headers are set correctly."""
        # Test CORS headers on a regular GET request
        with patch("app.main.rag_agent") as mock_agent:
            mock_health = HealthStatus(
                status="healthy",
                timestamp="2024-01-01T00:00:00Z",
                services={"test": "healthy"},
            )

            async def mock_health_check():
                return mock_health

            mock_agent.health_check = mock_health_check

            response = client.get(
                "/health", headers={"Origin": "http://localhost:3000"}
            )
            assert response.status_code == 200
            # CORS headers should be present
            assert response.headers.get("access-control-allow-origin") == "*"
            assert response.headers.get("access-control-allow-credentials") == "true"

    async def test_error_handling_rag_error(self, client):
        """Test RAG agent error handling."""
        from app.rag_agent import RAGAgentError

        with patch("app.main.rag_agent") as mock_agent:

            async def mock_query(*args, **kwargs):
                raise RAGAgentError("Query processing failed")

            mock_agent.query = mock_query

            query_data = {"query": "test query"}
            response = client.post("/query", json=query_data)

            assert response.status_code == 422
            data = response.json()
            assert "Query processing failed" in data["detail"]

    async def test_error_handling_unexpected_error(self, client):
        """Test unexpected error handling."""
        with patch("app.main.rag_agent") as mock_agent:

            async def mock_query(*args, **kwargs):
                raise Exception("Unexpected error")

            mock_agent.query = mock_query

            query_data = {"query": "test query"}
            response = client.post("/query", json=query_data)

            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]
