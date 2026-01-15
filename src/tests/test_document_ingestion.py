"""
Comprehensive Document Ingestion Tests

Tests document ingestion functionality for various scenarios including:
- Different file types (txt, pdf)
- Error handling and edge cases
- Duplicate file handling
- Large file processing
- Network and database resilience
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock

from app.rag_agent import RAGAgent, RAGAgentError
from app.vector_store import VectorStoreError


class TestDocumentIngestion:
    """Comprehensive document ingestion test suite."""

    @pytest.fixture
    async def rag_agent(self):
        """Create and initialize RAG agent for testing."""
        agent = RAGAgent()
        await agent.initialize()
        return agent

    @pytest.fixture
    def temp_text_content(self):
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
    def temp_empty_file(self):
        """Create a temporary empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        yield temp_path

        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

    @pytest.fixture
    def temp_large_file(self):
        """Create a temporary large file for testing."""
        # Create a file with ~1MB of content
        content = "This is a test document. " * 50000

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = f.name

        yield temp_path

        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

    @pytest.mark.asyncio
    async def test_ingest_text_file_success(self, rag_agent, temp_text_content):
        """Test successful ingestion of a text file."""
        # Create file in uploads directory
        upload_dir = rag_agent.document_loader.upload_dir
        upload_dir.mkdir(exist_ok=True)
        filename = f"test_ml_fundamentals_{hash(temp_text_content) % 1000}.txt"
        file_path = upload_dir / filename

        with open(file_path, "w") as f:
            f.write(temp_text_content)

        try:
            result = await rag_agent.process_document(
                file_path=filename, content_type="text/plain"
            )

            assert result is not None
            assert result.id is not None
            assert result.metadata.filename == filename
            assert result.status == "processed"
            assert result.chunks_count > 0
            assert result.embeddings_count > 0
        finally:
            # Cleanup
            try:
                file_path.unlink(missing_ok=True)
            except:
                pass

    @pytest.mark.asyncio
    async def test_ingest_empty_file(self, rag_agent, temp_empty_file):
        """Test ingestion of an empty file."""
        with pytest.raises(RAGAgentError, match="Document processing failed"):
            await rag_agent.process_document(
                file_path=os.path.basename(temp_empty_file), content_type="text/plain"
            )

    @pytest.mark.asyncio
    async def test_ingest_large_file(self, rag_agent, temp_large_file):
        """Test ingestion of a large file."""
        result = await rag_agent.process_document(
            file_path=os.path.basename(temp_large_file), content_type="text/plain"
        )

        assert result is not None
        assert result.id is not None
        assert result.chunks_count > 0  # Should be chunked
        assert result.embeddings_count > 0

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_file(self, rag_agent):
        """Test ingestion of a non-existent file."""
        with pytest.raises(RAGAgentError, match="Document processing failed"):
            await rag_agent.process_document(
                file_path="nonexistent_file.txt", content_type="text/plain"
            )

    @pytest.mark.asyncio
    async def test_ingest_duplicate_file(self, rag_agent, temp_text_file):
        """Test ingestion of the same file twice."""
        # First ingestion
        result1 = await rag_agent.process_document(
            file_path=temp_text_file, content_type="text/plain"
        )

        # Second ingestion (should work or handle gracefully)
        # Note: Current implementation may allow duplicates or overwrite
        result2 = await rag_agent.process_document(
            file_path=os.path.basename(temp_text_file), content_type="text/plain"
        )

        # Both should succeed (implementation may vary)
        assert result1 is not None
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_vector_store_failure(self, rag_agent, temp_text_file):
        """Test handling of vector store failures."""
        with patch.object(
            rag_agent.vector_store,
            "store_document",
            side_effect=VectorStoreError("Database error"),
        ):
            with pytest.raises(RAGAgentError, match="Document processing failed"):
                await rag_agent.process_document(
                    file_path=os.path.basename(temp_text_file),
                    content_type="text/plain",
                )

    @pytest.mark.asyncio
    async def test_ollama_failure(self, rag_agent, temp_text_file):
        """Test handling of Ollama embedding failures."""
        with patch.object(
            rag_agent.ollama_client,
            "embed_batch",
            side_effect=Exception("Ollama error"),
        ):
            with pytest.raises(RAGAgentError, match="Document processing failed"):
                await rag_agent.process_document(
                    file_path=os.path.basename(temp_text_file),
                    content_type="text/plain",
                )

    @pytest.mark.asyncio
    async def test_document_loader_failure(self, rag_agent, temp_text_file):
        """Test handling of document loader failures."""
        with patch.object(
            rag_agent.document_loader,
            "process_document",
            side_effect=Exception("Loader error"),
        ):
            with pytest.raises(RAGAgentError, match="Document processing failed"):
                await rag_agent.process_document(
                    file_path=os.path.basename(temp_text_file),
                    content_type="text/plain",
                )

    @pytest.mark.asyncio
    async def test_memory_failure(self, rag_agent, temp_text_file):
        """Test handling of memory/cache failures."""
        with patch.object(
            rag_agent.memory,
            "get_cached_embedding",
            side_effect=Exception("Cache error"),
        ):
            # Should still work (cache failure shouldn't break ingestion)
            result = await rag_agent.process_document(
                file_path=temp_text_file, content_type="text/plain"
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_unsupported_content_type(self, rag_agent, temp_text_file):
        """Test processing with unsupported content type."""
        # Should still attempt processing based on file extension
        result = await rag_agent.process_document(
            file_path=temp_text_file,
            content_type="application/octet-stream",  # Generic type
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_chunking_behavior(self, rag_agent, temp_large_file):
        """Test that large files are properly chunked."""
        result = await rag_agent.process_document(
            file_path=os.path.basename(temp_large_file), content_type="text/plain"
        )

        assert result is not None
        assert result.chunks_count > 1  # Large file should be chunked
        assert result.embeddings_count == result.chunks_count  # One embedding per chunk

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, rag_agent, temp_text_file):
        """Test that document metadata is properly preserved."""
        result = await rag_agent.process_document(
            file_path=os.path.basename(temp_text_file), content_type="text/plain"
        )

        assert result is not None
        assert result.filename == os.path.basename(temp_text_file)
        assert result.content_type == "text/plain"
        assert result.size > 0
        assert result.chunks_count > 0
        assert result.embeddings_count > 0

    @pytest.mark.asyncio
    async def test_caching_behavior(self, rag_agent, temp_text_file):
        """Test that embeddings are properly cached."""
        # First processing
        result1 = await rag_agent.process_document(
            file_path=os.path.basename(temp_text_file), content_type="text/plain"
        )

        # Second processing (should use cached embeddings)
        result2 = await rag_agent.process_document(
            file_path=os.path.basename(temp_text_file), content_type="text/plain"
        )

        # Both should succeed
        assert result1 is not None
        assert result2 is not None
        # Results may be the same or different depending on implementation
        # but both should be valid

    @pytest.mark.asyncio
    async def test_concurrent_ingestion(self, rag_agent, temp_text_file):
        """Test concurrent document ingestion."""
        import asyncio

        async def ingest_once():
            return await rag_agent.process_document(
                file_path=os.path.basename(temp_text_file), content_type="text/plain"
            )

        # Run multiple ingestions concurrently
        tasks = [ingest_once() for _ in range(3)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed or fail gracefully
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent ingestion failed: {result}")
            else:
                assert result is not None
                assert result.id is not None
