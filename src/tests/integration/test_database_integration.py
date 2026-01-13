"""
Integration tests for database operations with real PostgreSQL.
Tests vector storage, retrieval, and similarity search with actual database.
"""

from typing import Any, Dict, List

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.slow
class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_vector_store_connection(self, real_vector_store):
        """Test that vector store can connect to real PostgreSQL."""
        assert real_vector_store._pool is not None

        # Test health check
        health = await real_vector_store.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_document_storage_and_retrieval(self, real_vector_store):
        """Test storing and retrieving documents."""
        # Store a document
        metadata = {
            "filename": "test.txt",
            "content_type": "text/plain",
            "size": 1024,
            "word_count": 100,
            "page_count": 1,
            "upload_time": "2026-01-13T10:00:00",
        }

        doc_id = await real_vector_store.store_document("test-doc-1", metadata)
        assert doc_id == "test-doc-1"

        # Retrieve the document
        retrieved = await real_vector_store.get_document("test-doc-1")
        assert retrieved is not None
        assert retrieved["id"] == "test-doc-1"
        assert retrieved["filename"] == "test.txt"
        assert retrieved["content_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_chunk_storage_and_counting(self, real_vector_store):
        """Test storing chunks and counting them."""
        chunks = [
            {
                "id": "chunk-1",
                "document_id": "test-doc-1",
                "content": "This is the first chunk of test content.",
                "chunk_index": 0,
                "total_chunks": 3,
                "embedding": [0.1] * 768,
                "metadata": {"chunk_size": 50},
            },
            {
                "id": "chunk-2",
                "document_id": "test-doc-1",
                "content": "This is the second chunk of test content.",
                "chunk_index": 1,
                "total_chunks": 3,
                "embedding": np.array([0.2] * 768, dtype=np.float32),
                "metadata": {"chunk_size": 55},
            },
            {
                "id": "chunk-3",
                "document_id": "test-doc-1",
                "content": "This is the third chunk of test content.",
                "chunk_index": 2,
                "total_chunks": 3,
                "embedding": np.array([0.3] * 768, dtype=np.float32),
                "metadata": {"chunk_size": 45},
            },
        ]

        # Store chunks
        stored_count = await real_vector_store.store_chunks(chunks)
        assert stored_count == 3

        # Check total chunk count
        total_chunks = await real_vector_store.get_chunk_count()
        assert total_chunks == 3

        # Check chunk count for specific document
        doc_chunks = await real_vector_store.get_chunk_count("test-doc-1")
        assert doc_chunks == 3

    @pytest.mark.asyncio
    async def test_similarity_search_basic(self, real_vector_store):
        """Test basic similarity search functionality."""
        # First store the document
        await real_vector_store.store_document(
            "search-doc-1",
            {
                "filename": "search_test.txt",
                "content_type": "text/plain",
                "size": 100,
                "upload_time": "2024-01-01T00:00:00",
                "checksum": "search-checksum",
                "word_count": 10,
            },
        )

        # Then store some test chunks
        chunks = [
            {
                "id": "search-chunk-1",
                "document_id": "search-doc-1",
                "content": "Machine learning is a method of data analysis.",
                "chunk_index": 0,
                "total_chunks": 1,
                "embedding": np.array(
                    [0.1, 0.2, 0.3] * 256, dtype=np.float32
                ),  # 768 dimensions
                "metadata": {},
            }
        ]

        await real_vector_store.store_chunks(chunks)

        # Perform similarity search
        query_vector = [0.1, 0.2, 0.3] * 256  # Similar to stored chunk
        results = await real_vector_store.similarity_search(query_vector, top_k=5)

        assert len(results) >= 1
        assert results[0]["id"] == "search-chunk-1"
        assert "similarity_score" in results[0]
        assert isinstance(results[0]["similarity_score"], float)

    @pytest.mark.asyncio
    async def test_similarity_search_with_threshold(self, real_vector_store):
        """Test similarity search with relevance threshold."""
        # Store chunks with different similarity levels
        chunks = [
            {
                "id": "thresh-chunk-1",
                "document_id": "thresh-doc-1",
                "content": "High similarity content",
                "chunk_index": 0,
                "total_chunks": 1,
                "embedding": np.array([0.9] * 768, dtype=np.float32),  # Very similar
                "metadata": {},
            },
            {
                "id": "thresh-chunk-2",
                "document_id": "thresh-doc-2",
                "content": "Low similarity content",
                "chunk_index": 0,
                "total_chunks": 1,
                "embedding": np.array(
                    [0.1] * 768, dtype=np.float32
                ),  # Not very similar
                "metadata": {},
            },
        ]

        await real_vector_store.store_chunks(chunks)

        # Search with high threshold
        query_vector = [0.9] * 768
        results = await real_vector_store.similarity_search(
            query_vector, top_k=10, threshold=0.8
        )

        # Should only return high similarity results
        assert len(results) >= 1
        for result in results:
            assert result["similarity_score"] >= 0.8

    @pytest.mark.asyncio
    async def test_document_listing(self, real_vector_store):
        """Test document listing functionality."""
        # Store multiple documents
        docs_data = [
            (
                "list-doc-1",
                {
                    "filename": "doc1.txt",
                    "content_type": "text/plain",
                    "size": 1000,
                    "upload_time": "2026-01-13T10:00:00",
                },
            ),
            (
                "list-doc-2",
                {
                    "filename": "doc2.txt",
                    "content_type": "text/plain",
                    "size": 2000,
                    "upload_time": "2026-01-13T11:00:00",
                },
            ),
            (
                "list-doc-3",
                {
                    "filename": "doc3.txt",
                    "content_type": "text/plain",
                    "size": 1500,
                    "upload_time": "2026-01-13T12:00:00",
                },
            ),
        ]

        for doc_id, metadata in docs_data:
            await real_vector_store.store_document(doc_id, metadata)

        # List all documents
        documents = await real_vector_store.list_documents(limit=10, offset=0)
        assert len(documents) >= 3

        # Check that documents are returned with correct fields
        for doc in documents:
            assert "id" in doc
            assert "filename" in doc
            assert "uploaded_at" in doc
            assert "chunks_count" in doc

    @pytest.mark.asyncio
    async def test_document_deletion(self, real_vector_store):
        """Test document deletion with cascade to chunks."""
        # Store document and chunks
        await real_vector_store.store_document(
            "delete-doc-1",
            {
                "filename": "delete.txt",
                "content_type": "text/plain",
                "size": 500,
                "upload_time": "2026-01-13T10:00:00",
            },
        )

        chunks = [
            {
                "id": "delete-chunk-1",
                "document_id": "delete-doc-1",
                "content": "Content to be deleted",
                "chunk_index": 0,
                "total_chunks": 1,
                "embedding": np.array([0.5] * 768, dtype=np.float32),
                "metadata": {},
            }
        ]
        await real_vector_store.store_chunks(chunks)

        # Verify they exist
        doc = await real_vector_store.get_document("delete-doc-1")
        assert doc is not None

        chunk_count = await real_vector_store.get_chunk_count("delete-doc-1")
        assert chunk_count == 1

        # Delete document
        result = await real_vector_store.delete_document("delete-doc-1")
        assert result is True

        # Verify they're gone
        doc = await real_vector_store.get_document("delete-doc-1")
        assert doc is None

        chunk_count = await real_vector_store.get_chunk_count("delete-doc-1")
        assert chunk_count == 0

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, real_vector_store):
        """Test concurrent database operations."""
        import asyncio

        async def store_and_search(doc_id: int):
            # Store document
            await real_vector_store.store_document(
                f"concurrent-{doc_id}",
                {
                    "filename": f"concurrent{doc_id}.txt",
                    "content_type": "text/plain",
                    "size": 100 * doc_id,
                    "upload_time": "2026-01-13T10:00:00",
                },
            )

            # Store chunk
            chunk = {
                "id": f"concurrent-chunk-{doc_id}",
                "document_id": f"concurrent-{doc_id}",
                "content": f"Concurrent content {doc_id}",
                "chunk_index": 0,
                "total_chunks": 1,
                "embedding": np.array([float(doc_id) / 10.0] * 768, dtype=np.float32),
                "metadata": {},
            }
            await real_vector_store.store_chunks([chunk])

            # Search
            query_vector = [float(doc_id) / 10.0] * 768
            results = await real_vector_store.similarity_search(query_vector, top_k=1)
            return len(results) > 0

        # Run concurrent operations
        tasks = [store_and_search(i) for i in range(1, 6)]
        results = await asyncio.gather(*tasks)

        # All operations should succeed
        assert all(results)

    @pytest.mark.asyncio
    async def test_database_transaction_integrity(self, real_vector_store):
        """Test that database operations maintain integrity."""
        # Store initial data
        await real_vector_store.store_document(
            "integrity-doc-1",
            {
                "filename": "integrity.txt",
                "content_type": "text/plain",
                "size": 1000,
                "upload_time": "2026-01-13T10:00:00",
            },
        )

        chunks = [
            {
                "id": "integrity-chunk-1",
                "document_id": "integrity-doc-1",
                "content": "Integrity test content",
                "chunk_index": 0,
                "total_chunks": 1,
                "embedding": np.array([0.7] * 768, dtype=np.float32),
                "metadata": {"test": "integrity"},
            }
        ]

        # Store chunks
        stored_count = await real_vector_store.store_chunks(chunks)
        assert stored_count == 1

        # Verify data integrity
        doc = await real_vector_store.get_document("integrity-doc-1")
        assert doc is not None
        assert doc["filename"] == "integrity.txt"

        chunk_count = await real_vector_store.get_chunk_count("integrity-doc-1")
        assert chunk_count == 1

        # Verify search works
        results = await real_vector_store.similarity_search([0.7] * 768, top_k=1)
        assert len(results) >= 1
        assert results[0]["document_id"] == "integrity-doc-1"
