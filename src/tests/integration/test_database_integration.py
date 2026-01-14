"""
Simple integration tests for database operations with real PostgreSQL.
Tests basic connectivity and document operations.
"""

import pytest


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Basic integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_vector_store_connection(self, real_vector_store):
        """Test that vector store can connect to real PostgreSQL."""
        assert real_vector_store._pool is not None

        # Test health check
        health = await real_vector_store.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_document_operations(self, real_vector_store):
        """Test basic document storage and retrieval."""
        # Store a document
        doc_id = await real_vector_store.store_document(
            "test-doc-1",
            {
                "filename": "test.txt",
                "content_type": "text/plain",
                "size": 1024,
                "upload_time": "2024-01-01T00:00:00",
            },
        )
        assert doc_id == "test-doc-1"

        # Retrieve the document
        retrieved = await real_vector_store.get_document("test-doc-1")
        assert retrieved is not None
        assert retrieved["id"] == "test-doc-1"
        assert retrieved["filename"] == "test.txt"

    @pytest.mark.asyncio
    async def test_document_listing(self, real_vector_store):
        """Test document listing functionality."""
        # List documents (may be empty or have test data)
        documents = await real_vector_store.list_documents(limit=10, offset=0)
        assert isinstance(documents, list)

        # If we have documents, check their structure
        for doc in documents:
            assert "id" in doc
            assert "filename" in doc
