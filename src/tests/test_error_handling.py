"""
Error Handling and Resilience Tests

Tests the system's ability to handle various error conditions gracefully:
- Network timeouts and connection failures
- Missing or unavailable services
- Corrupt or invalid data
- Resource exhaustion
- Configuration errors
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.document_loader import UnsupportedFileTypeError
from app.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModelError,
)
from app.rag_agent import RAGAgent, RAGAgentError
from app.vector_store import VectorStoreError


class TestErrorHandling:
    """Test error handling and system resilience."""

    @pytest.fixture
    async def rag_agent(self):
        """Create and initialize RAG agent for testing."""
        agent = RAGAgent()
        await agent.initialize()
        return agent

    @pytest.fixture
    def temp_corrupt_file(self, tmp_path):
        """Create a corrupt file for testing."""
        corrupt_file = tmp_path / "corrupt.txt"
        # Write some binary data that will cause parsing issues
        corrupt_file.write_bytes(b"\x00\x01\x02\x03invalid utf-8 content\x80\x81\x82")
        return str(corrupt_file)

    def test_ollama_connection_timeout(self):
        """Test handling of Ollama service timeouts."""
        client = OllamaClient()

        with patch.object(client, "_client") as mock_client:
            # Mock httpx client to raise timeout
            mock_client.request.side_effect = httpx.TimeoutException(
                "Request timed out"
            )

            with pytest.raises(OllamaConnectionError, match="Request timeout"):
                asyncio.run(client._make_request("GET", "/api/tags"))

    def test_ollama_connection_refused(self):
        """Test handling of Ollama service connection refused."""
        client = OllamaClient()

        with patch.object(client, "_client") as mock_client:
            # Mock httpx client to raise connection error
            mock_client.request.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(
                OllamaConnectionError, match="Failed to connect to Ollama"
            ):
                asyncio.run(client._make_request("GET", "/api/tags"))

    @pytest.mark.asyncio
    async def test_ollama_model_not_found(self):
        """Test handling of requests for non-existent models."""
        client = OllamaClient()

        with patch.object(client, "_client") as mock_client:
            # Create a real httpx.HTTPStatusError
            import httpx

            mock_response = httpx.Response(404, json={"error": "model not found"})
            http_error = httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "http://localhost:11434/api/generate"),
                response=mock_response,
            )
            mock_client.request.side_effect = http_error

            with pytest.raises(OllamaModelError, match="Model or endpoint not found"):
                await client._make_request("GET", "/api/generate")

    @pytest.mark.asyncio
    async def test_ollama_server_error(self):
        """Test handling of Ollama server errors."""
        client = OllamaClient()

        with patch.object(client, "_client") as mock_client:
            # Create a real httpx.HTTPStatusError
            import httpx

            mock_response = httpx.Response(500, json={"error": "Internal server error"})
            http_error = httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("GET", "http://localhost:11434/api/tags"),
                response=mock_response,
            )
            mock_client.request.side_effect = http_error

            with pytest.raises(OllamaClientError, match="HTTP 500"):
                await client._make_request("GET", "/api/tags")

    def test_ollama_unexpected_error(self):
        """Test handling of unexpected errors in Ollama client."""
        client = OllamaClient()

        with patch.object(client, "_client") as mock_client:
            # Mock unexpected exception
            mock_client.request.side_effect = ValueError("Unexpected error")

            with pytest.raises(OllamaClientError, match="Unexpected error"):
                asyncio.run(client._make_request("GET", "/api/tags"))

    @pytest.mark.asyncio
    async def test_corrupt_file_processing(self, rag_agent, temp_corrupt_file):
        """Test handling of corrupt or unreadable files."""
        # Copy corrupt file to upload directory
        upload_dir = rag_agent.document_loader.upload_dir
        upload_dir.mkdir(exist_ok=True)
        filename = "corrupt_test.txt"
        upload_path = upload_dir / filename

        import shutil

        shutil.copy2(temp_corrupt_file, upload_path)

        try:
            # Should handle corrupt file gracefully
            result = await rag_agent.process_document(
                file_path=filename, content_type="text/plain"
            )

            # Should still return a result (might be partial or empty)
            assert result is not None
            assert hasattr(result, "status")
        except RAGAgentError:
            # Acceptable to fail with clear error
            pass
        finally:
            # Cleanup
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_vector_store_connection_failure(self, rag_agent):
        """Test handling when vector store becomes unavailable."""
        # Create a test file
        upload_dir = rag_agent.document_loader.upload_dir
        upload_dir.mkdir(exist_ok=True)
        filename = "vector_store_test.txt"
        upload_path = upload_dir / filename

        with open(upload_path, "w") as f:
            f.write("Test content for vector store failure")

        try:
            with patch.object(
                rag_agent.vector_store,
                "store_document",
                side_effect=VectorStoreError("Connection lost"),
            ):
                with pytest.raises(RAGAgentError, match="Document processing failed"):
                    await rag_agent.process_document(
                        file_path=filename, content_type="text/plain"
                    )
        finally:
            # Cleanup
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_memory_cache_failure(self, rag_agent):
        """Test handling when memory/cache operations fail."""
        # Create a test file
        upload_dir = rag_agent.document_loader.upload_dir
        upload_dir.mkdir(exist_ok=True)
        filename = "memory_test.txt"
        upload_path = upload_dir / filename

        with open(upload_path, "w") as f:
            f.write("Test content for memory failure")

        try:
            with patch.object(
                rag_agent.memory,
                "get_cached_embedding",
                side_effect=Exception("Cache error"),
            ):
                # Currently cache failures break processing - expect RAGAgentError
                with pytest.raises(RAGAgentError, match="Document processing failed"):
                    await rag_agent.process_document(
                        file_path=filename, content_type="text/plain"
                    )
        finally:
            # Cleanup
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass

    def test_unsupported_file_type_error(self):
        """Test UnsupportedFileTypeError is properly raised."""
        from app.document_loader import DocumentLoader

        loader = DocumentLoader()
        # Create a test file
        test_path = Path("/tmp/test.xml")
        test_path.write_text("<xml>test</xml>")

        try:
            with pytest.raises(
                UnsupportedFileTypeError, match="Unsupported content type"
            ):
                asyncio.run(loader.process_document(test_path, "application/xml"))
        finally:
            # Cleanup
            test_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_empty_file_processing(self, rag_agent):
        """Test processing of empty files."""
        # Create empty file
        upload_dir = rag_agent.document_loader.upload_dir
        upload_dir.mkdir(exist_ok=True)
        filename = "empty_test.txt"
        upload_path = upload_dir / filename

        with open(upload_path, "w") as f:
            f.write("")  # Empty file

        try:
            with pytest.raises(RAGAgentError, match="Document processing failed"):
                await rag_agent.process_document(
                    file_path=filename, content_type="text/plain"
                )
        finally:
            # Cleanup
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass

    def test_file_not_found_error(self):
        """Test handling of file not found errors."""
        from app.document_loader import DocumentLoader

        loader = DocumentLoader()
        nonexistent_path = Path("/tmp/nonexistent_file.txt")

        with pytest.raises(FileNotFoundError, match="Document not found"):
            asyncio.run(loader.process_document(nonexistent_path, "text/plain"))

    @pytest.mark.asyncio
    async def test_ollama_embed_batch_failure(self, rag_agent):
        """Test handling of Ollama embedding batch failures."""
        # Create a test file
        upload_dir = rag_agent.document_loader.upload_dir
        upload_dir.mkdir(exist_ok=True)
        filename = "ollama_embed_test.txt"
        upload_path = upload_dir / filename

        with open(upload_path, "w") as f:
            f.write("Test content for embedding failure")

        try:
            with patch.object(
                rag_agent.ollama_client,
                "embed_batch",
                side_effect=Exception("Embedding failed"),
            ):
                with pytest.raises(RAGAgentError, match="Document processing failed"):
                    await rag_agent.process_document(
                        file_path=filename, content_type="text/plain"
                    )
        finally:
            # Cleanup
            try:
                upload_path.unlink(missing_ok=True)
            except OSError:
                pass
