"""State definition for the multi-agent RAG system using LangGraph."""

from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage


class DockerMultiAgentRAGState(TypedDict):
    """State for the Docker-based Multi-Agent RAG System.

    This state manages the flow between query processing, retrieval,
    MCP tool execution, and response generation while maintaining
    security boundaries (no critical data to LLM).
    """

    # Query-Management
    query: str  # Original user query
    sanitized_query: str  # Query with sensitive data removed

    # Agenten-Koordination
    agent_tasks: Dict[str, Any]  # Tasks assigned to agents
    agent_results: Dict[str, Any]  # Results from agent execution
    intent_classification: Optional[Dict[str, Any]]  # Query intent analysis results

    # Retrieval Results (full content for LLM processing)
    retrieved_results: List[Dict[str, Any]]  # Document retrieval results with content
    allowed_doc_ids: List[str]  # Authorized document IDs
    retrieval_confidence: Optional[float]  # Confidence in retrieval results

    # MCP Tool Results
    mcp_search_results: Optional[Dict[str, Any]]  # Search tool results

    # Finale Antwort
    final_response: Optional[str]  # Generated response
    confidence_score: float  # Response confidence (0-1)
    sources: List[Dict[str, Any]]  # Cited sources metadata

    # Monitoring & Debugging
    processing_time: float  # Total processing time
    agent_metrics: Dict[str, Any]  # Performance metrics per agent


class AgentMetrics(TypedDict):
    """Metrics collected for each agent execution."""

    agent_name: str  # Name of the agent
    execution_time: float  # Time spent executing
    success: bool  # Whether execution was successful
    tool_calls: List[str]  # Tools called by agent
    result_count: int  # Number of results generated
    error_message: Optional[str]  # Error message if failed


class SecurityContext(TypedDict):
    """Security context for agent execution."""

    user_id: str  # User identifier
    user_level: str  # Access level (basic/standard/admin)
    allowed_domains: List[str]  # Permitted document domains
    blocked_terms: List[str]  # Terms that trigger filtering
    max_results: int  # Maximum results per agent


def create_initial_state(
    query: str, user_id: str = "anonymous", user_level: str = "standard"
) -> DockerMultiAgentRAGState:
    """Create initial state for a new query.

    Args:
        query: The user's query
        user_id: User identifier for access control
        user_level: User's access level

    Returns:
        Initialized state dictionary
    """
    return {
        "query": query,
        "sanitized_query": query,  # Will be sanitized by first agent
        "agent_tasks": {},
        "agent_results": {},
        "intent_classification": {},
        "retrieved_results": [],
        "retrieval_confidence": 0.0,
        "allowed_doc_ids": [],
        "mcp_search_results": None,
        "final_response": None,
        "confidence_score": 0.0,
        "sources": [],
        "processing_time": 0.0,
        "agent_metrics": {},
    }


def sanitize_state_for_llm(state: DockerMultiAgentRAGState) -> Dict[str, Any]:
    """Create a safe version of state for LLM consumption.

    Removes any sensitive data and ensures only metadata is passed to LLMs.

    Args:
        state: Full state object

    Returns:
        Sanitized state safe for LLM consumption
    """
    # Only include safe fields
    safe_fields = {
        "sanitized_query",
        "retrieved_results",
        "mcp_search_results",
        "confidence_score",
        "sources",
    }

    return {k: v for k, v in state.items() if k in safe_fields}
