"""Tests for MCP client Docker toolkit integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.mcp_client import DockerMCPClient, MCPClientError, MCPToolExecutionError


class TestDockerMCPClient:
    """Test suite for Docker MCP Client."""

    @pytest.fixture
    def mcp_client(self):
        """Create MCP client instance."""
        return DockerMCPClient()

    @pytest.fixture
    def mock_client(self):
        """Mock httpx AsyncClient."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_initialization(self, mcp_client):
        """Test MCP client initialization."""
        assert mcp_client.coordinator_url == "http://localhost:8001"
        assert mcp_client.timeout == 30.0
        assert mcp_client._client is None
        assert isinstance(mcp_client.available_tools, dict)

    @pytest.mark.asyncio
    async def test_connect_success(self, mcp_client, mock_client):
        """Test successful MCP coordinator connection."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={})
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_client):
            await mcp_client.connect()

            assert mcp_client._client == mock_client
            # Should call /health first, then /tools for refresh
            assert mock_client.get.call_count >= 2

    @pytest.mark.asyncio
    async def test_connect_failure(self, mcp_client):
        """Test MCP coordinator connection failure."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get.side_effect = Exception("Connection failed")

            with pytest.raises(
                MCPClientError, match="MCP Coordinator connection failed"
            ):
                await mcp_client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, mcp_client, mock_client):
        """Test client disconnection."""
        mcp_client._client = mock_client
        await mcp_client.disconnect()

        mock_client.aclose.assert_called_once()
        assert mcp_client._client is None

    @pytest.mark.asyncio
    async def test_refresh_available_tools_success(self, mcp_client, mock_client):
        """Test successful tool refresh."""
        mcp_client._client = mock_client

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "brave_search": {"category": "search", "description": "Web search"}
            }
        )
        mock_client.get.return_value = mock_response

        await mcp_client.refresh_available_tools()

        assert "brave_search" in mcp_client.available_tools
        assert mcp_client.available_tools["brave_search"]["category"] == "search"

    @pytest.mark.asyncio
    async def test_refresh_available_tools_failure_fallback(
        self, mcp_client, mock_client
    ):
        """Test tool refresh failure with fallback to defaults."""
        mcp_client._client = mock_client
        mock_client.get.side_effect = Exception("Network error")

        await mcp_client.refresh_available_tools()

        # Should fall back to default tools
        assert len(mcp_client.available_tools) == 5
        assert "brave_search" in mcp_client.available_tools
        assert "arxiv" in mcp_client.available_tools

    def test_get_default_tools(self, mcp_client):
        """Test default tools structure."""
        default_tools = mcp_client._get_default_tools()

        assert len(default_tools) == 5
        assert "brave_search" in default_tools
        assert "github_official" in default_tools

        brave_tool = default_tools["brave_search"]
        assert brave_tool["category"] == "search"
        assert "parameters" in brave_tool

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mcp_client, mock_client):
        """Test successful tool execution."""
        mcp_client._client = mock_client

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"result": "search results"})
        mock_client.post.return_value = mock_response

        result = await mcp_client.execute_tool(
            "search", "brave_search", {"query": "test query"}
        )

        assert result["result"] == "search results"
        assert "_metadata" in result
        assert result["_metadata"]["tool_name"] == "brave_search"
        assert result["_metadata"]["category"] == "search"
        assert "execution_time" in result["_metadata"]

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, mcp_client, mock_client):
        """Test tool execution timeout."""
        mcp_client._client = mock_client
        mock_client.post.side_effect = httpx.TimeoutException("Request timeout")

        with pytest.raises(MCPToolExecutionError, match="Tool execution timed out"):
            await mcp_client.execute_tool("search", "brave_search", {})

    @pytest.mark.asyncio
    async def test_execute_tool_http_error(self, mcp_client, mock_client):
        """Test tool execution HTTP error."""
        mcp_client._client = mock_client

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        with pytest.raises(MCPToolExecutionError, match="Tool execution failed"):
            await mcp_client.execute_tool("search", "brave_search", {})

    @pytest.mark.asyncio
    async def test_execute_tool_generic_error(self, mcp_client, mock_client):
        """Test tool execution generic error."""
        mcp_client._client = mock_client
        mock_client.post.side_effect = Exception("Network error")

        with pytest.raises(MCPToolExecutionError, match="Tool execution error"):
            await mcp_client.execute_tool("search", "brave_search", {})

    @pytest.mark.asyncio
    async def test_execute_parallel_tools_success(self, mcp_client, mock_client):
        """Test successful parallel tool execution."""
        mcp_client._client = mock_client

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"result": "tool result"})
        mock_client.post.return_value = mock_response

        tool_requests = [
            {
                "category": "search",
                "tool_name": "brave_search",
                "parameters": {"query": "test1"},
            },
            {
                "category": "search",
                "tool_name": "arxiv",
                "parameters": {"query": "test2"},
            },
        ]

        results = await mcp_client.execute_parallel_tools(tool_requests)

        assert len(results) == 2
        for result in results:
            assert "result" in result
            assert "_metadata" in result

    @pytest.mark.asyncio
    async def test_execute_parallel_tools_with_failures(self, mcp_client, mock_client):
        """Test parallel tool execution with some failures."""
        mcp_client._client = mock_client

        # Mock successful response
        success_response = AsyncMock()
        success_response.raise_for_status = MagicMock()
        success_response.json = MagicMock(return_value={"result": "success"})

        # Mock failed response
        mock_client.post.side_effect = [success_response, Exception("Tool failed")]

        tool_requests = [
            {"category": "search", "tool_name": "brave_search"},
            {"category": "search", "tool_name": "arxiv"},
        ]

        results = await mcp_client.execute_parallel_tools(tool_requests)

        assert len(results) == 2
        assert results[0]["result"] == "success"
        assert "error" in results[1]
        assert results[1]["_metadata"]["error"] is True

    @pytest.mark.asyncio
    async def test_get_tool_info_all_tools(self, mcp_client):
        """Test getting all tool information."""
        mcp_client.available_tools = {"tool1": {"category": "search"}}

        tools = await mcp_client.get_tool_info()

        assert tools == {"tool1": {"category": "search"}}

    @pytest.mark.asyncio
    async def test_get_tool_info_by_category(self, mcp_client):
        """Test getting tools filtered by category."""
        mcp_client.available_tools = {
            "brave_search": {"category": "search"},
            "github": {"category": "code"},
            "arxiv": {"category": "search"},
        }

        search_tools = await mcp_client.get_tool_info(category="search")

        assert len(search_tools) == 2
        assert "brave_search" in search_tools
        assert "arxiv" in search_tools

    @pytest.mark.asyncio
    async def test_get_tool_info_specific_tool(self, mcp_client):
        """Test getting specific tool information."""
        mcp_client.available_tools = {
            "brave_search": {"category": "search", "description": "Web search"}
        }

        tool_info = await mcp_client.get_tool_info(tool_name="brave_search")

        assert tool_info["category"] == "search"
        assert tool_info["description"] == "Web search"

    @pytest.mark.asyncio
    async def test_get_tool_info_unknown_tool(self, mcp_client):
        """Test getting information for unknown tool."""
        tool_info = await mcp_client.get_tool_info(tool_name="unknown_tool")

        assert tool_info == {}

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mcp_client, mock_client):
        """Test successful health check."""
        mcp_client._client = mock_client
        mcp_client.available_tools = {"tool1": {}, "tool2": {}}

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"status": "healthy"})
        mock_client.get.return_value = mock_response

        health = await mcp_client.health_check()

        assert health["status"] == "healthy"
        assert health["available_tools"] == 2
        assert "last_check" in health

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mcp_client, mock_client):
        """Test failed health check."""
        mcp_client._client = mock_client
        mock_client.get.side_effect = Exception("Connection failed")

        health = await mcp_client.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "last_check" in health

    @pytest.mark.asyncio
    async def test_context_manager(self, mcp_client, mock_client):
        """Test async context manager."""
        with patch("httpx.AsyncClient", return_value=mock_client):
            async with mcp_client:
                assert mcp_client._client == mock_client

            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_connection_auto_connect(self, mcp_client, mock_client):
        """Test automatic connection establishment."""
        with patch("httpx.AsyncClient", return_value=mock_client):
            await mcp_client._ensure_connection()

            assert mcp_client._client == mock_client

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        """Test client with custom timeout."""
        client = DockerMCPClient(timeout=60.0)
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_custom_coordinator_url(self):
        """Test client with custom coordinator URL."""
        client = DockerMCPClient(coordinator_url="http://custom:9090")
        assert client.coordinator_url == "http://custom:9090"

    @pytest.mark.asyncio
    async def test_execute_tool_metadata(self, mcp_client, mock_client):
        """Test tool execution metadata."""
        mcp_client._client = mock_client

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"data": "result"})
        mock_client.post.return_value = mock_response

        result = await mcp_client.execute_tool(
            "code", "github_official", {"query": "test"}
        )

        metadata = result["_metadata"]
        assert metadata["tool_name"] == "github_official"
        assert metadata["category"] == "code"
        assert isinstance(metadata["execution_time"], float)
        assert "timestamp" in metadata
        assert "execution_id" in metadata

    @pytest.mark.asyncio
    async def test_parallel_tools_empty_list(self, mcp_client):
        """Test parallel execution with empty tool list."""
        results = await mcp_client.execute_parallel_tools([])
        assert results == []

    def test_get_tool_placeholder(self, mcp_client):
        """Test placeholder LangChain tool method."""
        # Method is not implemented, should not crash
        result = mcp_client.get_tool("test_tool")
        assert result is None
