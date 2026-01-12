"""Ollama client for local AI model integration."""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.models import (
    OllamaEmbedRequest,
    OllamaEmbedResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelInfo,
)


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""

    pass


class OllamaConnectionError(OllamaClientError):
    """Exception raised when connection to Ollama fails."""

    pass


class OllamaModelError(OllamaClientError):
    """Exception raised when model operations fail."""

    pass


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_client()

    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            )

    async def _close_client(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    async def _make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request to Ollama API with retry logic."""
        await self._ensure_client()
        if self._client is None:
            raise OllamaConnectionError("HTTP client not initialized")

        try:
            response = await self._client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {e}"
            )
        except httpx.TimeoutException as e:
            raise OllamaConnectionError(f"Request timeout: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelError(f"Model or endpoint not found: {endpoint}")
            raise OllamaClientError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error: {e}")

    async def list_models(self) -> List[OllamaModelInfo]:
        """List available models in Ollama."""
        data = await self._make_request("GET", "/api/tags")
        models = []
        for model_data in data.get("models", []):
            models.append(OllamaModelInfo(**model_data))
        return models

    async def check_model(self, model_name: str) -> bool:
        """Check if a model is available."""
        models = await self.list_models()
        return any(model.name == model_name for model in models)

    async def pull_model(self, model_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Pull a model from the registry."""
        async with self._client.stream(
            "POST", "/api/pull", json={"name": model_name}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    async def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model."""
        return await self._make_request(
            "DELETE", "/api/delete", json={"name": model_name}
        )

    async def embed(self, request: OllamaEmbedRequest) -> OllamaEmbedResponse:
        """Generate embeddings for text."""
        data = await self._make_request(
            "POST", "/api/embeddings", json=request.model_dump()
        )
        # Add model to response since Ollama API doesn't include it
        data["model"] = request.model
        return OllamaEmbedResponse(**data)

    async def embed_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        model_name = model or settings.ollama_embed_model

        # Check if model is available
        if not await self.check_model(model_name):
            raise OllamaModelError(f"Model {model_name} not available")

        tasks = []
        for text in texts:
            request = OllamaEmbedRequest(model=model_name, prompt=text)
            tasks.append(self.embed(request))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        embeddings = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            embeddings.append(result.embedding)

        return embeddings

    async def generate(self, request: OllamaGenerateRequest) -> OllamaGenerateResponse:
        """Generate text using a language model."""
        data = await self._make_request(
            "POST", "/api/generate", json=request.model_dump()
        )
        return OllamaGenerateResponse(**data)

    async def generate_stream(
        self, request: OllamaGenerateRequest
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response."""
        request.stream = True

        async with self._client.stream(
            "POST", "/api/generate", json=request.model_dump()
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get("done"):
                            break
                        yield data.get("response", "")
                    except json.JSONDecodeError:
                        continue

    async def health_check(self) -> bool:
        """Check if Ollama service is healthy."""
        try:
            await self._make_request("GET", "/api/tags")
            return True
        except Exception:
            return False

    async def get_model_info(self, model_name: str) -> Optional[OllamaModelInfo]:
        """Get information about a specific model."""
        models = await self.list_models()
        for model in models:
            if model.name == model_name:
                return model
        return None
