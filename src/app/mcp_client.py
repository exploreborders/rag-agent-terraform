"""Docker MCP Toolkit Client for LangGraph agent integration."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    """Base exception for MCP client errors."""

    pass


class MCPToolExecutionError(MCPClientError):
    """Exception raised when MCP tool execution fails."""

    pass


class DockerMCPClient:
    """Client für Docker MCP Toolkit Integration.

    Dieser Client kommuniziert mit dem MCP-Coordinator-Service,
    der in einem separaten Docker-Container läuft und MCP Tools managed.
    """

    def __init__(self, coordinator_url: str = None, timeout: float = 30.0):
        """Initialisiere MCP Client.

        Args:
            coordinator_url: URL des MCP-Coordinator-Services
            timeout: Timeout für Tool-Ausführung
        """
        self.coordinator_url = coordinator_url or settings.mcp_coordinator_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        # Verfügbare MCP Tools (werden vom Coordinator gemeldet)
        self.available_tools: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Docker MCP Client initialized with coordinator: {self.coordinator_url}"
        )

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """Stelle Verbindung zum MCP-Coordinator her."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

        try:
            # Teste Verbindung
            response = await self._client.get(f"{self.coordinator_url}/health")
            response.raise_for_status()
            logger.info("Connected to MCP Coordinator")

            # Hole verfügbare Tools
            await self.refresh_available_tools()

        except Exception as e:
            logger.error(f"Failed to connect to MCP Coordinator: {e}")
            raise MCPClientError(f"MCP Coordinator connection failed: {e}")

    async def disconnect(self):
        """Trenne Verbindung."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_connection(self):
        """Stelle sicher, dass Verbindung aktiv ist."""
        if self._client is None:
            await self.connect()

    async def refresh_available_tools(self):
        """Aktualisiere Liste der verfügbaren Tools."""
        await self._ensure_connection()

        try:
            response = await self._client.get(f"{self.coordinator_url}/tools")
            response.raise_for_status()
            self.available_tools = response.json()
            logger.info(f"Refreshed available tools: {len(self.available_tools)} tools")
        except Exception as e:
            logger.warning(f"Failed to refresh tools: {e}")
            # Fallback: Verwende bekannte Tools
            self.available_tools = self._get_default_tools()

    def _get_default_tools(self) -> Dict[str, Dict[str, Any]]:
        """Default verfügbare Tools für Docker MCP Toolkit."""
        return {
            "brave_search": {
                "category": "search",
                "description": "Web search using Brave Search API",
                "parameters": {"query": "string", "count": "integer"},
            },
            "arxiv": {
                "category": "search",
                "description": "Academic paper search on arXiv",
                "parameters": {"query": "string", "max_results": "integer"},
            },
            "perplexity": {
                "category": "search",
                "description": "AI-powered research and summarization",
                "parameters": {"query": "string"},
            },
            "github_official": {
                "category": "code",
                "description": "GitHub repository and code search",
                "parameters": {"query": "string", "language": "string"},
            },
            "context7": {
                "category": "code",
                "description": "Code documentation and examples",
                "parameters": {"topic": "string"},
            },
        }

    async def execute_tool(
        self, category: str, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Führe MCP Tool aus.

        Args:
            category: Tool-Kategorie (search, code, database, etc.)
            tool_name: Name des spezifischen Tools
            parameters: Tool-Parameter

        Returns:
            Tool-Ergebnis

        Raises:
            MCPToolExecutionError: Bei Ausführungsfehlern
        """
        await self._ensure_connection()

        execution_id = f"mcp_{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()

        try:
            # Erstelle Anfrage an MCP-Coordinator
            request_data = {
                "execution_id": execution_id,
                "category": category,
                "tool_name": tool_name,
                "parameters": parameters,
                "timestamp": start_time.isoformat(),
            }

            logger.info(
                f"Executing MCP tool: {category}.{tool_name} (ID: {execution_id})"
            )

            # Sende Anfrage an Coordinator
            response = await self._client.post(
                f"{self.coordinator_url}/execute-mcp-tool",
                json=request_data,
                timeout=self.timeout,
            )

            response.raise_for_status()
            result = response.json()

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                f"MCP tool {tool_name} executed successfully in {execution_time:.2f}s"
            )

            # Erweitere Result um Metadaten
            result["_metadata"] = {
                "execution_id": execution_id,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "category": category,
                "timestamp": datetime.utcnow().isoformat(),
            }

            return result

        except httpx.TimeoutException:
            logger.error(f"MCP tool {tool_name} timed out after {self.timeout}s")
            raise MCPToolExecutionError(f"Tool execution timed out: {tool_name}")

        except httpx.HTTPStatusError as e:
            logger.error(
                f"MCP tool {tool_name} failed with status {e.response.status_code}: {e.response.text}"
            )
            raise MCPToolExecutionError(
                f"Tool execution failed: {tool_name} - {e.response.text}"
            )

        except Exception as e:
            logger.error(f"MCP tool {tool_name} execution failed: {e}")
            raise MCPToolExecutionError(f"Tool execution error: {tool_name} - {str(e)}")

    async def execute_parallel_tools(
        self, tool_requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Führe mehrere Tools parallel aus.

        Args:
            tool_requests: Liste von Tool-Anfragen
                [{"category": "search", "tool_name": "brave_search", "parameters": {...}}]

        Returns:
            Liste der Tool-Ergebnisse
        """
        tasks = []
        for request in tool_requests:
            task = self.execute_tool(
                request["category"], request["tool_name"], request.get("parameters", {})
            )
            tasks.append(task)

        # Führe alle Tools parallel aus
        logger.info(f"Executing {len(tasks)} MCP tools in parallel")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verarbeite Ergebnisse (Exceptions werden als Fehler-Results behandelt)
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "error": str(result),
                        "tool_request": tool_requests[i],
                        "_metadata": {"execution_time": 0.0, "error": True},
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def get_tool_info(
        self, category: str = None, tool_name: str = None
    ) -> Dict[str, Any]:
        """Hole Informationen über verfügbare Tools.

        Args:
            category: Optional: Filter nach Kategorie
            tool_name: Optional: Spezifischer Tool-Name

        Returns:
            Tool-Informationen
        """
        if not self.available_tools:
            await self.refresh_available_tools()

        if tool_name:
            return self.available_tools.get(tool_name, {})
        elif category:
            return {
                name: info
                for name, info in self.available_tools.items()
                if info.get("category") == category
            }
        else:
            return self.available_tools

    async def health_check(self) -> Dict[str, Any]:
        """Führe Health Check für MCP System durch."""
        try:
            await self._ensure_connection()

            # Teste Coordinator-Verbindung
            response = await self._client.get(f"{self.coordinator_url}/health")
            response.raise_for_status()

            coordinator_health = response.json()

            return {
                "status": "healthy",
                "coordinator": coordinator_health,
                "available_tools": len(self.available_tools),
                "last_check": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat(),
            }

    # Kompatibilität mit LangChain Tool Interface
    def get_tool(self, tool_name: str):
        """Erstelle LangChain-kompatibles Tool für MCP Tool."""
        # Hier könnte eine Wrapper-Klasse implementiert werden
        # die MCP Tools als LangChain Tools verfügbar macht
        pass


# Globale MCP Client Instance
mcp_client = DockerMCPClient()
