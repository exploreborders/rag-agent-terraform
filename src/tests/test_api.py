"""Integration tests for FastAPI application."""

from unittest.mock import patch, AsyncMock

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

    def test_health_endpoint_healthy(self, client):
        """Test health check endpoint when system is healthy."""
        # Mock the RAG agent to simulate healthy state
        with patch("app.main.rag_agent") as mock_agent:
            # Mock the required methods
            mock_agent.vector_store.health_check = AsyncMock()
            mock_agent.ollama_client.health_check = AsyncMock()

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "version" in data
            assert "services" in data

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
        with patch("app.main.rag_agent") as mock_agent:  # noqa: F841
            query_data = {"query": ""}

            response = client.post("/query", json=query_data)

            assert response.status_code == 422  # Pydantic validation error
            data = response.json()
            assert "query" in str(data).lower()  # Should mention query field

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

    def test_cors_headers(self, client):
        """Test CORS headers are set correctly."""
        # Mock the health check dependencies
        with patch("app.main.rag_agent") as mock_agent:
            mock_agent.vector_store.health_check = AsyncMock()
            mock_agent.ollama_client.health_check = AsyncMock()

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

            assert response.status_code == 500  # Now returns 500 for agent errors
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
            assert "Query processing failed" in data["detail"]
