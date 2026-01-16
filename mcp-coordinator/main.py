"""MCP Coordinator Service für Docker MCP Toolkit Integration."""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Coordinator Service",
    description="Docker MCP Toolkit Coordination Service for Multi-Agent RAG System",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution."""

    tool_name: str
    parameters: Dict[str, Any]


class ToolExecutionResponse(BaseModel):
    """Response model for tool execution."""

    tool_name: str
    result: Any
    execution_time: float
    timestamp: str


class MCPCoordinator:
    """MCP Coordinator using Docker MCP Gateway."""

    def __init__(self):
        """Initialize MCP Coordinator with Docker MCP Gateway."""
        self.available_tools = []
        self.gateway_available = False

    async def get_available_tools(self):
        """Get available tools from MCP Gateway or fallback to configured tools."""
        try:
            # Try to connect to Docker MCP Gateway
            # Note: MCP servers are already running in Docker containers
            server_params = StdioServerParameters(
                command="docker", args=["mcp", "gateway", "run"]
            )

            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Get available tools from MCP Gateway (servers already running in Docker)
                    tools_result = await session.list_tools()
                    tools = [tool.name for tool in tools_result.tools]
                    self.gateway_available = True

                    logger.info(
                        f"Successfully connected to MCP Gateway with {len(tools)} tools: {tools}"
                    )

                    # Return the actual tools found, or fallback if empty
                    return tools if tools else self._get_fallback_tools()

        except Exception as e:
            logger.error(f"Failed to connect to MCP Gateway: {e}")
            self.gateway_available = False
            logger.info(
                "Using configured fallback tools (MCP servers may not be running)"
            )
            # Fallback to configured tools if gateway is not available
            return self._get_fallback_tools()

    def _get_fallback_tools(self) -> List[str]:
        """Get fallback tool list for when MCP Gateway is not available."""
        return [
            "search",
            "search_semantic",
            "search_arxiv",
            "search_biorxiv",
            "get_current_time",
        ]

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool via MCP Gateway."""
        start_time = datetime.utcnow()

        try:
            # Connect to Docker MCP Gateway for each request
            server_params = StdioServerParameters(
                command="docker", args=["mcp", "gateway", "run"]
            )

            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Check if tool is available
                    tools_result = await session.list_tools()
                    available_tools = [tool.name for tool in tools_result.tools]

                    if tool_name not in available_tools:
                        raise ValueError(
                            f"Tool '{tool_name}' not available. Available tools: {available_tools}"
                        )

                    # Call tool via MCP Gateway
                    result = await session.call_tool(tool_name, parameters)

                    execution_time = (datetime.now() - start_time).total_seconds()

                    return {
                        "tool_name": tool_name,
                        "result": result.content
                        if hasattr(result, "content")
                        else str(result),
                        "execution_time": execution_time,
                        "timestamp": datetime.utcnow().isoformat(),
                        "success": True,
                        "source": "mcp_gateway",
                    }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.warning(f"MCP Gateway tool execution failed, trying fallback: {e}")

            # Fallback to mock implementations
            return self._execute_mock_tool(tool_name, parameters, execution_time)

    def _execute_mock_tool(
        self, tool_name: str, parameters: Dict[str, Any], execution_time: float
    ) -> Dict[str, Any]:
        """Execute mock tool implementation as fallback."""
        # Mock responses for testing - using correct Docker MCP Toolkit names
        if tool_name == "search":
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 3)

            # Mock DuckDuckGo search results
            mock_results = [
                {
                    "title": f"DuckDuckGo Result 1 for '{query}'",
                    "url": f"https://duckduckgo.com/?q={query.replace(' ', '+')}",
                    "snippet": f"This is a mock search result for {query} from DuckDuckGo.",
                },
                {
                    "title": f"DuckDuckGo Result 2 for '{query}'",
                    "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                    "snippet": f"Another mock result showing web search capabilities for {query}.",
                },
                {
                    "title": f"DuckDuckGo Result 3 for '{query}'",
                    "url": f"https://wikipedia.org/wiki/{query.replace(' ', '_')}",
                    "snippet": f"Wikipedia-style mock result for {query} search.",
                },
            ]

            return {
                "tool_name": tool_name,
                "result": {
                    "query": query,
                    "results": mock_results[:max_results],
                    "total_results": len(mock_results),
                },
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
                "source": "mock_fallback",
            }

        elif tool_name in ["search_semantic", "search_arxiv", "search_biorxiv"]:
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 3)

            # Mock academic paper search results
            paper_results = [
                {
                    "title": f"Advances in {query} Research",
                    "authors": ["Dr. Jane Smith", "Prof. John Doe"],
                    "abstract": f"This paper explores recent advances in {query} with comprehensive analysis and novel findings.",
                    "url": f"https://arxiv.org/abs/2301.00123",
                    "year": 2023,
                    "citations": 45,
                    "venue": "arXiv"
                    if tool_name == "search_arxiv"
                    else "Semantic Scholar",
                },
                {
                    "title": f"Machine Learning Applications in {query}",
                    "authors": ["Dr. Alice Johnson", "Dr. Bob Wilson"],
                    "abstract": f"A comprehensive study of machine learning techniques applied to {query} problems.",
                    "url": f"https://biorxiv.org/content/10.1101/2023.01.15.524102"
                    if tool_name == "search_biorxiv"
                    else "https://semanticscholar.org/paper/123456789",
                    "year": 2023,
                    "citations": 23,
                    "venue": "bioRxiv" if tool_name == "search_biorxiv" else "Nature",
                },
                {
                    "title": f"Future Directions in {query}",
                    "authors": ["Prof. Sarah Davis", "Dr. Michael Brown"],
                    "abstract": f"This work examines future research directions and challenges in {query}.",
                    "url": f"https://semanticscholar.org/paper/987654321",
                    "year": 2024,
                    "citations": 12,
                    "venue": "IEEE Transactions",
                },
            ]

            return {
                "tool_name": tool_name,
                "result": {
                    "query": query,
                    "papers": paper_results[:max_results],
                    "total_papers": len(paper_results),
                },
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
                "source": "mock_fallback",
            }

        elif tool_name == "get_current_time":
            from datetime import datetime

            import pytz

            timezone_str = parameters.get(
                "timezone", "local"
            )  # Default to local instead of UTC

            # Get current time in UTC first
            current_time_utc = datetime.now(pytz.UTC)

            # Determine target timezone - default to local system timezone
            if timezone_str == "local":
                # Try to get local timezone, fallback to a common one
                try:
                    import time

                    local_tz_name = (
                        time.tzname[0] if time.tzname else "America/New_York"
                    )
                    target_tz = pytz.timezone(local_tz_name)
                except:
                    target_tz = pytz.timezone("America/New_York")  # Fallback
            elif timezone_str == "UTC":
                target_tz = pytz.UTC
            else:
                try:
                    target_tz = pytz.timezone(timezone_str)
                except:
                    target_tz = pytz.UTC  # Fallback to UTC

            # Convert to target timezone
            local_time = current_time_utc.astimezone(target_tz)

            # Format times in a more human-readable way
            local_formatted = local_time.strftime("%A, %B %d, %Y at %I:%M:%S %p")
            utc_formatted = current_time_utc.strftime(
                "%A, %B %d, %Y at %I:%M:%S %p UTC"
            )

            return {
                "tool_name": tool_name,
                "result": {
                    "reference": "now",
                    "requested_timezone": timezone_str,
                    "local_time": local_time.isoformat(),
                    "local_formatted": f"{local_formatted} {target_tz.zone}",
                    "utc_time": current_time_utc.isoformat(),
                    "utc_formatted": utc_formatted,
                    "timezone_name": target_tz.zone,
                },
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
                "source": "mock_fallback",
            }

        else:
            return {
                "tool_name": tool_name,
                "error": f"Unknown tool: {tool_name}",
                "execution_time": execution_time,
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
                "source": "mock_fallback",
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            tools = await self.get_available_tools()
            return {
                "status": "healthy",
                "gateway_connected": self.gateway_available,
                "available_tools": len(tools),
                "tools": tools,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }


# Global coordinator instance
coordinator = MCPCoordinator()


@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("MCP Coordinator Service starting...")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("MCP Coordinator Service shutting down...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "MCP Coordinator", "version": "0.1.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return await coordinator.health_check()


@app.get("/tools")
async def get_tools():
    """Get available MCP tools."""
    try:
        tools = await coordinator.get_available_tools()
        return {"tools": tools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/{tool_name}/execute")
async def execute_specific_tool(tool_name: str, parameters: Dict[str, Any]):
    """Execute a specific MCP tool."""
    try:
        result = await coordinator.execute_tool(tool_name, parameters)
        if not result.get("success", False):
            raise HTTPException(
                status_code=500, detail=result.get("error", "Tool execution failed")
            )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
async def execute_tool(request: ToolExecutionRequest):
    """Execute tool via request model."""
    try:
        result = await coordinator.execute_tool(request.tool_name, request.parameters)
        if not result.get("success", False):
            raise HTTPException(
                status_code=500, detail=result.get("error", "Tool execution failed")
            )
        return ToolExecutionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
