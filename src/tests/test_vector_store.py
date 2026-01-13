"""
Unit tests for VectorStore class.
Covers vector storage, similarity search, document management, and database operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.vector_store import VectorStore


# Mock database connection for all tests
@pytest.fixture(autouse=True)
def mock_database():
    """Mock all database interactions."""
    with (
        patch("asyncpg.create_pool"),
        patch("app.vector_store.register_vector"),
    ):
        yield


class TestVectorStore:
    """Test cases for VectorStore functionality."""

    @pytest.fixture
    async def vector_store(self):
        """Create a VectorStore instance with mocked connection."""
        store = VectorStore()
        mock_conn = AsyncMock()

        # Set up the pool to return mock connection
        with patch.object(store, "_pool") as mock_pool:
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_pool.acquire = MagicMock(return_value=mock_context)

            store._pool = mock_pool
            yield store

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test VectorStore initialization."""
        store = VectorStore()
        assert store.connection_string is not None  # Uses settings.database_url
        assert "postgresql://" in store.connection_string
        assert store._pool is None

    @pytest.mark.asyncio
    async def test_context_manager(self, vector_store):
        """Test VectorStore context manager."""
        with (
            patch.object(
                vector_store, "connect", new_callable=AsyncMock
            ) as mock_connect,
            patch.object(
                vector_store, "disconnect", new_callable=AsyncMock
            ) as mock_disconnect,
        ):
            async with vector_store:
                mock_connect.assert_called_once()

            mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect(self, vector_store):
        """Test database connection."""
        with (
            patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_pool,
            patch("app.vector_store.register_vector", new_callable=AsyncMock),
        ):
            mock_pool_instance = MagicMock()
            mock_conn = AsyncMock()
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_pool_instance.acquire = MagicMock(return_value=mock_context)
            mock_pool.return_value = mock_pool_instance

            await vector_store.connect()

            assert vector_store._pool is not None
            mock_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, vector_store):
        """Test database disconnection."""
        mock_pool = AsyncMock()
        vector_store._pool = mock_pool

        await vector_store.disconnect()

        assert vector_store._pool is None
        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_connection_when_connected(self, vector_store):
        """Test _ensure_connection when already connected."""
        # _pool is already set in the fixture, so it should not call connect
        with patch.object(
            vector_store, "connect", new_callable=AsyncMock
        ) as mock_connect:
            await vector_store._ensure_connection()
            # Should not create new connection since _pool exists
            mock_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_connection_when_not_connected(self, vector_store):
        """Test _ensure_connection when not connected."""
        # Temporarily disable the autouse _ensure_connection patch
        with (
            patch.object(
                VectorStore,
                "_ensure_connection",
                side_effect=VectorStore._ensure_connection,
            ) as original_ensure,
            patch.object(
                vector_store, "connect", new_callable=AsyncMock
            ) as mock_connect,
        ):
            vector_store._pool = None  # Simulate not connected
            await vector_store._ensure_connection()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_schema(self, vector_store):
        """Test database schema initialization."""
        # The mock connection should have execute method
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute = AsyncMock()

        await vector_store.initialize_schema()

        # Should execute schema creation SQL
        assert mock_conn.execute.call_count >= 2  # Multiple DDL statements

    @pytest.mark.asyncio
    async def test_store_document(self, vector_store):
        """Test document metadata storage."""
        with patch.object(
            vector_store, "store_document", return_value="doc-1"
        ) as mock_store:
            metadata = {
                "filename": "test.txt",
                "content_type": "text/plain",
                "size": 1024,
                "word_count": 100,
                "page_count": 1,
                "upload_time": "2026-01-13T10:00:00",
            }

            result = await vector_store.store_document("doc-1", metadata)

            assert result == "doc-1"
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_chunks_success(self, vector_store, sample_chunks):
        """Test successful chunk storage with embeddings."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.executemany = AsyncMock()

        result = await vector_store.store_chunks(sample_chunks)

        assert result == 2
        mock_conn.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_chunks_with_numpy_embedding(self, vector_store):
        """Test chunk storage with numpy array embeddings."""
        chunks = [
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "content": "Test content",
                "chunk_index": 0,
                "total_chunks": 1,
                "embedding": np.array([0.1, 0.2, 0.3] * 256),  # 768 dims
                "metadata": {},
            }
        ]
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.executemany = AsyncMock()

        result = await vector_store.store_chunks(chunks)

        assert result == 1
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_similarity_search_success(self, vector_store):
        """Test successful vector similarity search."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Sample content",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "metadata": "{}",
                    "filename": "test.txt",
                    "content_type": "text/plain",
                    "similarity_score": 0.85,
                }
            ]
        )

        query_vector = [0.1] * 768
        results = await vector_store.similarity_search(query_vector, top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"
        assert results[0]["similarity_score"] == 0.85
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_similarity_search_with_filters(self, vector_store):
        """Test similarity search with document filters."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch = AsyncMock(return_value=[])

        query_vector = [0.1] * 768
        document_ids = ["doc-1", "doc-2"]

        await vector_store.similarity_search(
            query_vector, top_k=3, filters={"document_ids": document_ids}
        )

        # Should include document filter in query
        call_args = (
            mock_conn
        ) = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.call_args
        query = call_args[0][0]
        assert "document_id = ANY($3)" in query

    @pytest.mark.asyncio
    async def test_similarity_search_with_threshold(self, vector_store):
        """Test similarity search with relevance threshold."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch = AsyncMock(return_value=[])

        query_vector = [0.1] * 768

        await vector_store.similarity_search(query_vector, top_k=5, threshold=0.7)

        call_args = (
            mock_conn
        ) = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.call_args
        query = call_args[0][0]
        assert "1 - (dc.embedding <=> $1) > $3" in query

    @pytest.mark.asyncio
    async def test_get_document_success(self, vector_store):
        """Test successful document retrieval."""
        mock_doc = {
            "id": "doc-1",
            "filename": "test.txt",
            "content_type": "text/plain",
            "size": 1024,
            "upload_time": MagicMock(),
            "page_count": 1,
            "word_count": 100,
            "checksum": "abc123",
            "metadata": "{}",
        }
        mock_doc["upload_time"].isoformat.return_value = "2026-01-13T10:00:00"

        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow = AsyncMock(return_value=mock_doc)

        result = await vector_store.get_document("doc-1")

        assert result is not None
        assert result["id"] == "doc-1"
        assert result["filename"] == "test.txt"
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, vector_store):
        """Test document retrieval when document doesn't exist."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow = AsyncMock(return_value=None)

        result = await vector_store.get_document("nonexistent")

        assert result is None
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_documents_success(self, vector_store):
        """Test successful document listing."""
        mock_rows = [
            {
                "id": "doc-1",
                "filename": "test1.txt",
                "content_type": "text/plain",
                "size": 1024,
                "upload_time": MagicMock(),
                "page_count": 1,
                "word_count": 100,
                "checksum": "abc123",
                "metadata": "{}",
                "chunks_count": 5,
            }
        ]
        mock_rows[0]["upload_time"].isoformat.return_value = "2026-01-13T10:00:00"

        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        result = await vector_store.list_documents(limit=10, offset=0)

        assert len(result) == 1
        assert result[0]["id"] == "doc-1"
        assert result[0]["chunks_count"] == 5
        assert result[0]["uploaded_at"] == "2026-01-13T10:00:00"
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_success(self, vector_store):
        """Test successful document deletion."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute = AsyncMock()
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.return_value = "DELETE 1"

        result = await vector_store.delete_document("doc-1")

        assert result is True
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        assert mock_conn.execute.call_count == 2  # DELETE documents + DELETE chunks

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, vector_store):
        """Test document deletion when document doesn't exist."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute = AsyncMock()
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.execute.return_value = "DELETE 0"

        result = await vector_store.delete_document("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_chunk_count_all(self, vector_store):
        """Test getting total chunk count."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value=150)

        result = await vector_store.get_chunk_count()

        assert result == 150
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_chunk_count_for_document(self, vector_store):
        """Test getting chunk count for specific document."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value=25)

        result = await vector_store.get_chunk_count("doc-1")

        assert result == 25
        call_args = (
            mock_conn
        ) = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.call_args
        assert "document_id = $1" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_health_check_success(self, vector_store):
        """Test successful health check."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(return_value=1)

        result = await vector_store.health_check()

        assert result is True
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, vector_store):
        """Test health check failure."""
        mock_conn = vector_store._pool.acquire.return_value.__aenter__.return_value
        mock_conn.fetchval = AsyncMock(side_effect=Exception("Connection failed"))

        result = await vector_store.health_check()

        assert result is False
