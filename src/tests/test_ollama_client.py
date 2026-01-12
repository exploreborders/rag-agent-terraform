"""Unit tests for Ollama client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from app.ollama_client import OllamaClient, OllamaClientError, OllamaModelError
from app.models import OllamaEmbedRequest, OllamaGenerateRequest


class TestOllamaClient:
    """Test cases for Ollama client."""

    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client for testing."""
        client = AsyncMock()
        client.request = AsyncMock()
        client.stream = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        return client

    @pytest.fixture
    def ollama_client(self, mock_http_client):
        """Ollama client instance with mocked HTTP client."""
        with patch("httpx.AsyncClient", return_value=mock_http_client):
            client = OllamaClient(base_url="http://test:11434")
            client._client = mock_http_client
            return client

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        client = OllamaClient(base_url="http://test:11434")
        assert client.base_url == "http://test:11434"
        assert client.timeout == 120.0

    @pytest.mark.asyncio
    async def test_list_models_success(self, ollama_client, mock_http_client):
        """Test successful model listing."""
        mock_response = {
            "models": [
                {
                    "name": "llama3.2:latest",
                    "size": "3824064992",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "digest": "abc123",
                }
            ]
        }
        mock_http_client.request.return_value = MagicMock()
        mock_http_client.request.return_value.json.return_value = mock_response

        models = await ollama_client.list_models()
        assert len(models) == 1
        assert models[0].name == "llama3.2:latest"

    @pytest.mark.asyncio
    async def test_check_model_available(self, ollama_client, mock_http_client):
        """Test checking if model is available."""
        mock_response = {"models": [{"name": "llama3.2:latest"}]}
        mock_http_client.request.return_value = MagicMock()
        mock_http_client.request.return_value.json.return_value = mock_response

        available = await ollama_client.check_model("llama3.2:latest")
        assert available is True

        available = await ollama_client.check_model("nonexistent-model")
        assert available is False

    @pytest.mark.asyncio
    async def test_embed_success(self, ollama_client, mock_http_client):
        """Test successful embedding generation."""
        mock_response = {"embedding": [0.1, 0.2, 0.3], "model": "embeddinggemma:latest"}
        mock_http_client.request.return_value = MagicMock()
        mock_http_client.request.return_value.json.return_value = mock_response

        request = OllamaEmbedRequest(model="embeddinggemma:latest", prompt="test text")
        result = await ollama_client.embed(request)

        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.model == "embeddinggemma:latest"

    @pytest.mark.asyncio
    async def test_embed_batch(self, ollama_client, mock_http_client):
        """Test batch embedding generation."""
        mock_responses = [
            {"embedding": [0.1, 0.2], "model": "embeddinggemma:latest"},
            {"embedding": [0.3, 0.4], "model": "embeddinggemma:latest"},
        ]

        mock_http_client.request.side_effect = [
            MagicMock(json=MagicMock(return_value=resp)) for resp in mock_responses
        ]

        embeddings = await ollama_client.embed_batch(["text1", "text2"])
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_generate_success(self, ollama_client, mock_http_client):
        """Test successful text generation."""
        mock_response = {
            "model": "llama3.2:latest",
            "created_at": "2024-01-01T00:00:00Z",
            "response": "Generated text",
            "done": True,
            "total_duration": 1000000,
            "eval_count": 10,
            "eval_duration": 500000,
        }
        mock_http_client.request.return_value = MagicMock()
        mock_http_client.request.return_value.json.return_value = mock_response

        request = OllamaGenerateRequest(model="llama3.2:latest", prompt="test prompt")
        result = await ollama_client.generate(request)

        assert result.response == "Generated text"
        assert result.model == "llama3.2:latest"
        assert result.done is True

    @pytest.mark.asyncio
    async def test_connection_error(self, ollama_client, mock_http_client):
        """Test connection error handling."""
        from app.ollama_client import OllamaConnectionError

        mock_http_client.request.side_effect = httpx.ConnectError("Connection failed")

        with pytest.raises(OllamaConnectionError):
            await ollama_client.list_models()

    @pytest.mark.asyncio
    async def test_model_not_found_error(self, ollama_client, mock_http_client):
        """Test model not found error handling."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404)
        )
        mock_http_client.request.return_value = mock_response

        with pytest.raises(OllamaModelError):
            await ollama_client.list_models()

    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_client, mock_http_client):
        """Test successful health check."""
        mock_response = {"models": []}
        mock_http_client.request.return_value = MagicMock()
        mock_http_client.request.return_value.json.return_value = mock_response

        healthy = await ollama_client.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_client, mock_http_client):
        """Test failed health check."""
        mock_http_client.request.side_effect = httpx.ConnectError("Connection failed")

        healthy = await ollama_client.health_check()
        assert healthy is False

    @pytest.mark.asyncio
    async def test_context_manager(self, ollama_client, mock_http_client):
        """Test async context manager."""
        async with ollama_client:
            assert ollama_client._client is not None

        # After exiting context, client should be closed
        assert ollama_client._client is None

    @pytest.mark.asyncio
    async def test_retry_logic(self, ollama_client, mock_http_client):
        """Test retry logic on failures."""
        # First two calls fail, third succeeds
        mock_http_client.request.side_effect = [
            httpx.TimeoutException("Timeout"),
            httpx.TimeoutException("Timeout"),
            MagicMock(json=MagicMock(return_value={"models": []})),
        ]

        models = await ollama_client.list_models()
        assert models == []

        # Should have been called 3 times due to retries
        assert mock_http_client.request.call_count == 3
