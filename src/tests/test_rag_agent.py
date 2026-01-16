"""
Unit tests for RAGAgent class.
Covers core business logic, document processing, querying, and system management.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.rag_agent import RAGAgent, RAGAgentError


class TestRAGAgent:
    """Test cases for RAGAgent functionality."""

    @pytest.fixture
    async def rag_agent(self, mock_vector_store, mock_redis_memory, mock_ollama_client):
        """Create a RAGAgent instance with mocked dependencies."""
        with (
            patch("app.rag_agent.VectorStore", return_value=mock_vector_store),
            patch("app.rag_agent.AgentMemory", return_value=mock_redis_memory),
            patch("app.rag_agent.OllamaClient", return_value=mock_ollama_client),
        ):
            agent = RAGAgent()
            yield agent

    @pytest.mark.asyncio
    async def test_initialization(
        self, mock_vector_store, mock_redis_memory, mock_ollama_client
    ):
        """Test RAGAgent initialization with dependencies."""
        with (
            patch("app.rag_agent.VectorStore", return_value=mock_vector_store),
            patch("app.rag_agent.AgentMemory", return_value=mock_redis_memory),
            patch("app.rag_agent.OllamaClient", return_value=mock_ollama_client),
        ):
            agent = RAGAgent()

            assert agent.vector_store == mock_vector_store
            assert agent.memory == mock_redis_memory
            assert agent.ollama_client == mock_ollama_client
            assert not agent._initialized

    @pytest.mark.asyncio
    async def test_context_manager(self, rag_agent):
        """Test RAGAgent context manager functionality."""
        with patch.object(rag_agent, "cleanup") as mock_cleanup:
            async with rag_agent:
                assert rag_agent._initialized

            # Cleanup should be called
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize(self, rag_agent):
        """Test agent initialization."""
        await rag_agent.initialize()

        assert rag_agent._initialized
        rag_agent.vector_store.initialize_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, rag_agent):
        """Test that initialize is idempotent."""
        await rag_agent.initialize()
        await rag_agent.initialize()  # Should not fail

        assert rag_agent._initialized
        # Should only be called once due to initialized check
        assert rag_agent.vector_store.initialize_schema.call_count == 1

    @pytest.mark.asyncio
    async def test_cleanup(self, rag_agent):
        """Test agent cleanup."""
        rag_agent._initialized = True
        await rag_agent.cleanup()

        assert not rag_agent._initialized
        # Note: cleanup doesn't call close on components, just sets _initialized

    @pytest.mark.asyncio
    async def test_ensure_initialized_when_not_initialized(self, rag_agent):
        """Test _ensure_initialized when agent is not initialized."""
        assert not rag_agent._initialized

        await rag_agent._ensure_initialized()

        assert rag_agent._initialized
        rag_agent.vector_store.initialize_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_initialized_when_already_initialized(self, rag_agent):
        """Test _ensure_initialized when agent is already initialized."""
        rag_agent._initialized = True

        await rag_agent._ensure_initialized()

        # Should not call initialize again
        rag_agent.vector_store.initialize_schema.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_query_hash(self, rag_agent):
        """Test query hash generation."""
        query = "What is machine learning?"
        hash1 = rag_agent._generate_query_hash(query)
        hash2 = rag_agent._generate_query_hash(query)

        # Same query should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex length

        # Different query should produce different hash
        hash3 = rag_agent._generate_query_hash("What is AI?")
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_process_document_success(self, rag_agent, temp_test_file):
        """Test successful document processing."""
        # Setup mocks
        with patch.object(
            rag_agent.document_loader, "process_document"
        ) as mock_process:
            mock_process.return_value = {
                "document_id": "doc-1",
                "chunks": [
                    {"content": "Chunk 1", "chunk_index": 0, "total_chunks": 2},
                    {"content": "Chunk 2", "chunk_index": 1, "total_chunks": 2},
                ],
                "metadata": {
                    "filename": "test.txt",
                    "content_type": "text/plain",
                    "size": 100,
                    "word_count": 20,
                    "page_count": 1,
                    "upload_time": "2026-01-13T10:00:00",
                },
            }
            rag_agent.ollama_client.embed_batch.return_value = [
                MagicMock(embedding=[0.1] * 768),
                MagicMock(embedding=[0.2] * 768),
            ]
            rag_agent.vector_store.store_document.return_value = "doc-1"
            rag_agent.vector_store.store_chunks.return_value = 2

            result = await rag_agent.process_document(
                file_path=str(temp_test_file), content_type="text/plain"
            )

            assert result.status == "processed"
            assert result.chunks_count == 2
            assert result.embeddings_count == 2

            # Verify interactions
            mock_process.assert_called_once()
            # embed_batch is called once for each chunk (2 chunks = 2 calls)
            assert rag_agent.ollama_client.embed_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_process_document_with_caching(self, rag_agent, temp_test_file):
        """Test document processing with embedding caching."""
        # Setup mocks
        with patch.object(
            rag_agent.document_loader, "process_document"
        ) as mock_process:
            mock_process.return_value = {
                "document_id": "doc-1",
                "chunks": [
                    {"content": "Chunk 1", "chunk_index": 0, "total_chunks": 1},
                ],
                "metadata": {
                    "filename": "test.txt",
                    "content_type": "text/plain",
                    "size": 100,
                    "word_count": 20,
                    "page_count": 1,
                    "upload_time": "2026-01-13T10:00:00",
                },
            }
            # Setup cached embeddings
            cached_embedding = [0.1] * 768
            rag_agent.memory.get_cached_embedding.return_value = cached_embedding
            rag_agent.vector_store.store_document.return_value = "doc-1"
            rag_agent.vector_store.store_chunks.return_value = 1

            result = await rag_agent.process_document(
                file_path=str(temp_test_file), content_type="text/plain"
            )

            # Should use cached embeddings
            rag_agent.memory.get_cached_embedding.assert_called()
            # Should not call Ollama for embeddings
            rag_agent.ollama_client.embed_batch.assert_not_called()
            rag_agent.vector_store.store_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_file_not_found(self, rag_agent):
        """Test processing when document file doesn't exist."""
        with pytest.raises(RAGAgentError, match="Document processing failed"):
            await rag_agent.process_document(
                file_path="nonexistent.txt", content_type="text/plain"
            )

    @pytest.mark.asyncio
    async def test_process_document_ollama_error(self, rag_agent, temp_test_file):
        """Test document processing when Ollama fails."""
        # Setup mocks
        with patch.object(
            rag_agent.document_loader, "process_document"
        ) as mock_process:
            mock_process.return_value = {
                "document_id": "doc-1",
                "chunks": [
                    {"content": "Chunk 1", "chunk_index": 0, "total_chunks": 1},
                ],
                "metadata": {
                    "filename": "test.txt",
                    "content_type": "text/plain",
                    "size": 100,
                    "word_count": 20,
                    "page_count": 1,
                    "upload_time": "2026-01-13T10:00:00",
                },
            }
            rag_agent.ollama_client.embed_batch.side_effect = Exception(
                "Ollama connection failed"
            )

            with pytest.raises(RAGAgentError, match="Document processing failed"):
                await rag_agent.process_document(
                    file_path=str(temp_test_file), content_type="text/plain"
                )

    @pytest.mark.asyncio
    async def test_query_success(self, rag_agent):
        """Test successful query execution."""
        rag_agent.vector_store.similarity_search.return_value = [
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
        rag_agent.ollama_client.generate.return_value = MagicMock(
            response="Machine learning is a subset of AI."
        )

        result = await rag_agent.query("What is machine learning?")

        assert result.query == "What is machine learning?"
        assert "Machine learning is a subset of AI" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0].similarity_score == 0.85

    @pytest.mark.asyncio
    async def test_query_with_cached_result(self, rag_agent, sample_query_response):
        """Test query with cached result."""
        rag_agent.memory.get_cached_query_result.return_value = (
            sample_query_response.model_dump()
        )

        result = await rag_agent.query("What is machine learning?")

        assert result.query == "What is machine learning?"
        # Should not call vector search or Ollama
        rag_agent.vector_store.similarity_search.assert_not_called()
        rag_agent.ollama_client.generate.assert_not_called()
        # Note: cache_query_result is called after successful query, not when using cache

    @pytest.mark.asyncio
    async def test_query_with_document_filter(self, rag_agent):
        """Test query with document ID filtering."""
        document_ids = ["doc-1", "doc-2"]
        rag_agent.vector_store.similarity_search.return_value = [
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "content": "Content from doc-1",
                "chunk_index": 0,
                "total_chunks": 1,
                "metadata": {},
                "filename": "doc1.txt",
                "content_type": "text/plain",
                "similarity_score": 0.9,
            },
            {
                "id": "chunk-2",
                "document_id": "doc-2",
                "content": "Content from doc-2",
                "chunk_index": 0,
                "total_chunks": 1,
                "metadata": {},
                "filename": "doc2.txt",
                "content_type": "text/plain",
                "similarity_score": 0.8,
            },
            {
                "id": "chunk-3",
                "document_id": "doc-3",
                "content": "Content from doc-3",
                "chunk_index": 0,
                "total_chunks": 1,
                "metadata": {},
                "filename": "doc3.txt",
                "content_type": "text/plain",
                "similarity_score": 0.7,
            },
        ]

        result = await rag_agent.query("Test query", document_ids=document_ids)

        # Should only return results from specified documents
        assert len(result.sources) == 2
        assert all(source.document_id in document_ids for source in result.sources)

    @pytest.mark.asyncio
    async def test_query_with_session(self, rag_agent):
        """Test query with conversation session."""
        session_id = "session-123"

        await rag_agent.query("Test query", session_id=session_id)

        # Should get and update conversation
        rag_agent.memory.get_conversation.assert_called_with(session_id)
        rag_agent.memory.update_conversation.assert_called_with(
            session_id, rag_agent.memory.update_conversation.call_args[0][1]
        )

    @pytest.mark.asyncio
    async def test_query_empty_query(self, rag_agent):
        """Test query with empty query string."""
        # RAG agent doesn't validate empty queries - that's done at API level
        # This should not raise an exception
        rag_agent.vector_store.similarity_search.return_value = []
        rag_agent.ollama_client.generate.return_value = MagicMock(
            response="No relevant information found."
        )

        result = await rag_agent.query("")

        assert result.query == ""
        assert len(result.sources) == 0

    @pytest.mark.asyncio
    async def test_query_similarity_search_failure(self, rag_agent):
        """Test query when similarity search fails."""
        rag_agent.vector_store.similarity_search.side_effect = Exception(
            "Search failed"
        )

        with pytest.raises(RAGAgentError, match="Query processing failed"):
            await rag_agent.query("Test query")

    @pytest.mark.asyncio
    async def test_list_documents(self, rag_agent):
        """Test document listing."""
        mock_docs = [{"id": "doc-1", "filename": "test.txt"}]
        rag_agent.vector_store.list_documents.return_value = mock_docs

        result = await rag_agent.list_documents(limit=10, offset=0)

        assert result == mock_docs
        rag_agent.vector_store.list_documents.assert_called_with(limit=10, offset=0)

    @pytest.mark.asyncio
    async def test_get_document(self, rag_agent):
        """Test single document retrieval."""
        mock_doc = {"id": "doc-1", "filename": "test.txt"}
        rag_agent.vector_store.get_document.return_value = mock_doc

        result = await rag_agent.get_document("doc-1")

        assert result == mock_doc
        rag_agent.vector_store.get_document.assert_called_with("doc-1")

    @pytest.mark.asyncio
    async def test_delete_document(self, rag_agent):
        """Test document deletion."""
        rag_agent.vector_store.delete_document.return_value = True

        result = await rag_agent.delete_document("doc-1")

        assert result is True
        rag_agent.vector_store.delete_document.assert_called_with("doc-1")

    @pytest.mark.asyncio
    async def test_get_stats(self, rag_agent):
        """Test statistics retrieval."""
        rag_agent.vector_store.get_chunk_count.return_value = 150
        rag_agent.vector_store.list_documents.return_value = [
            {"id": f"doc-{i}"} for i in range(5)
        ]
        rag_agent.memory.get_stats.return_value = {"cache_hits": 10, "cache_misses": 5}

        result = await rag_agent.get_stats()

        assert result["total_documents"] == 5
        assert result["total_chunks"] == 150
        assert result["total_queries"] == 42  # Placeholder value
        assert result["average_response_time"] == 1.2  # Placeholder value

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, rag_agent):
        """Test health check when all services are healthy."""
        rag_agent.vector_store.health_check.return_value = True
        rag_agent.memory.health_check.return_value = True
        rag_agent.ollama_client.health_check.return_value = True

        result = await rag_agent.health_check()

        assert result.status == "healthy"
        assert result.services["vector_store"] == "healthy"
        assert result.services["redis"] == "healthy"
        assert result.services["ollama"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_partial_failure(self, rag_agent):
        """Test health check when some services fail."""
        rag_agent.vector_store.health_check.return_value = True
        rag_agent.memory.health_check.return_value = False
        rag_agent.ollama_client.health_check.return_value = True

        result = await rag_agent.health_check()

        assert result.status == "degraded"
        assert result.services["vector_store"] == "healthy"
        assert result.services["redis"] == "unhealthy"
        assert result.services["ollama"] == "healthy"

    @pytest.mark.asyncio
    async def test_clear_cache(self, rag_agent):
        """Test cache clearing."""
        result = await rag_agent.clear_cache()

        assert result is True
        rag_agent.memory.clear_all_cache.assert_called_once()
        # Note: clear_cache only clears memory cache, not vector store cache

    @pytest.mark.asyncio
    async def test_clear_cache_failure(self, rag_agent):
        """Test cache clearing when it fails."""
        rag_agent.memory.clear_all_cache.side_effect = Exception("Cache clear failed")

        result = await rag_agent.clear_cache()

        assert result is False
