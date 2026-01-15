"""Basis-LangGraph für Multi-Agenten-RAG-System (Phase 1 Platzhalter)."""

import logging
import re
from datetime import datetime
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END

from app.multi_agent_state import DockerMultiAgentRAGState
from app.mcp_client import DockerMCPClient
from app.models import OllamaGenerateRequest
from app.config import settings

logger = logging.getLogger(__name__)


def placeholder_node(state: DockerMultiAgentRAGState) -> Dict[str, Any]:
    """Platzhalter-Node für Phase 1 - gibt State unverändert zurück."""
    logger.info("Placeholder node executed - Phase 1 implementation pending")
    return {}


async def query_processor_agent(
    state: DockerMultiAgentRAGState,
) -> Dict[str, Any]:
    """Query Processor Agent: Uses LLM-powered intent analysis to dispatch queries to appropriate agents.

    This agent performs:
    1. Query sanitization (removes sensitive data)
    2. LLM-powered intent classification (RAG, search, code, academic)
    3. Smart agent task assignment with priority levels

    Returns routing decision for next agent(s) to execute.
    """
    import re
    from datetime import datetime

    start_time = datetime.utcnow()
    query = state.get("query", "").strip()

    if not query:
        logger.warning("Empty query received")
        return {
            "sanitized_query": "",
            "agent_tasks": {},
            "processing_time": (datetime.utcnow() - start_time).total_seconds(),
        }

    logger.info(f"Processing query: '{query[:100]}...'")

    # Step 1: Sanitize query (remove sensitive data)
    sanitized_query = sanitize_query_for_security(query)

    # Step 2: Classify intent and determine agent needs using LLM
    intent_classification = await classify_query_intent_with_llm(sanitized_query)

    # Step 3: Create agent task assignments based on intent
    agent_tasks = create_agent_tasks(sanitized_query, intent_classification)

    processing_time = (datetime.utcnow() - start_time).total_seconds()

    logger.info(
        f"Query processed in {processing_time:.2f}s - Intent: {intent_classification['primary_intent']}"
    )

    return {
        "sanitized_query": sanitized_query,
        "agent_tasks": agent_tasks,
        "intent_classification": intent_classification,
        "processing_time": processing_time,
    }


def sanitize_query_for_security(query: str) -> str:
    """Sanitize query by removing sensitive information."""
    # Define sensitive patterns (from the plan)
    SENSITIVE_PATTERNS = {
        "personal": [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit cards
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
        ],
        "corporate": [
            r"\b(confidential|internal|secret)\b",
            r"\b(password|token|key|credential)\b",
            r"\b(employee|salary|compensation)\b",
        ],
    }

    sanitized = query

    # Replace sensitive patterns with placeholders
    for category, patterns in SENSITIVE_PATTERNS.items():
        for pattern in patterns:
            sanitized = re.sub(
                pattern,
                f"[REDACTED_{category.upper()}]",
                sanitized,
                flags=re.IGNORECASE,
            )

    # Remove excessive whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    if sanitized != query:
        logger.warning("Query sanitized - sensitive data removed")

    return sanitized


async def classify_query_intent_with_llm(query: str) -> Dict[str, Any]:
    """Use LLM to intelligently classify query intent and determine agent needs."""
    from app.ollama_client import OllamaClient

    try:
        ollama_client = OllamaClient()

        # Create a comprehensive prompt for intent classification
        intent_prompt = f"""
You are an expert query analyzer for a multi-agent RAG system. Analyze this query and determine the SINGLE BEST primary intent.

CRITICAL: If the query mentions "this document", "this file", "the pdf", "explain this", or refers to a specific uploaded document, classify as "rag_only".

QUERY: "{query}"

 Choose ONE primary intent:
 - "rag_only": Questions about specific uploaded documents, files, or existing knowledge base ("this document", "explain this", "according to the file")
 - "web_search": General knowledge questions, current events, explanations
 - "academic": Research papers, scientific studies, academic topics

 Respond ONLY with a valid JSON object:
 {{
     "primary_intent": "rag_only|web_search|academic",
     "needs_rag": true|false,
     "needs_web_search": true|false,
    "complexity": "simple|moderate|complex",
    "confidence": 0.0,
    "reasoning": "brief explanation",
    "special_requirements": []
}}
"""

        # Get LLM analysis
        request = OllamaGenerateRequest(
            model=settings.ollama_model,
            prompt=intent_prompt,
            options={
                "temperature": 0.1,  # Low temperature for consistent analysis
                "num_predict": 500,  # Max tokens
            },
        )
        response_obj = await ollama_client.generate(request)
        response = (
            response_obj.response
            if hasattr(response_obj, "response")
            else str(response_obj)
        )

        # Parse JSON response - be more robust with LLM output
        import json
        import re

        try:
            # Try to extract JSON from the response (LLM might add extra text)
            json_match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found in response", response, 0)

            # Validate required fields
            required_fields = [
                "primary_intent",
                "needs_rag",
                "needs_web_search",
            ]
            if not all(field in analysis for field in required_fields):
                raise ValueError("Missing required fields in JSON response")

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback if LLM doesn't return valid JSON
            logger.warning(f"LLM returned invalid JSON for intent analysis: {e}")
            analysis = fallback_intent_classification(query)

        # Validate and ensure all required fields
        analysis.setdefault("primary_intent", "web_search")
        analysis.setdefault("needs_rag", False)
        analysis.setdefault("needs_web_search", True)
        analysis.setdefault("complexity", "moderate")
        analysis.setdefault("confidence", 0.7)
        analysis.setdefault("reasoning", "LLM-based analysis")
        analysis.setdefault("special_requirements", [])

        logger.info(
            f"LLM Intent Analysis: {analysis['primary_intent']} (confidence: {analysis['confidence']})"
        )

        return analysis

    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e}")
        return fallback_intent_classification(query)


def fallback_intent_classification(query: str) -> Dict[str, Any]:
    """Fallback intent classification using simple heuristics when LLM fails."""
    query_lower = query.lower()

    # Simple keyword-based classification as backup
    if any(
        phrase in query_lower
        for phrase in [
            "this document",
            "this file",
            "the pdf",
            "according to",
            "explain this",
        ]
    ):
        return {
            "primary_intent": "rag_only",
            "needs_rag": True,
            "needs_web_search": False,
            "complexity": "simple",
            "confidence": 0.9,
            "reasoning": "Document-specific query detected",
            "special_requirements": [],
        }

    if any(
        word in query_lower
        for word in ["research", "paper", "study", "academic", "arxiv"]
    ):
        return {
            "primary_intent": "academic",
            "needs_rag": True,  # Check existing docs first
            "needs_web_search": True,
            "complexity": "complex",
            "confidence": 0.7,
            "reasoning": "Academic/research query detected",
            "special_requirements": ["academic_sources"],
        }

    # Default: general web search
    return {
        "primary_intent": "web_search",
        "needs_rag": True,  # Always check existing docs
        "needs_web_search": True,
        "complexity": "moderate",
        "confidence": 0.6,
        "reasoning": "General knowledge query - using web search",
        "special_requirements": [],
    }


def create_agent_tasks(query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
    """Create task assignments for agents based on query intent."""
    tasks = {}

    # Always include retrieval agent for RAG as baseline (check existing docs first)
    # This ensures we always check the knowledge base before external sources
    tasks["retrieval"] = {
        "query": query,
        "intent": "baseline_retrieval",
        "max_results": 10,
        "priority": "high" if intent["primary_intent"] == "rag_only" else "normal",
    }

    # Include MCP research agent for web/academic search
    if intent.get("needs_web_search", False) or intent["primary_intent"] in [
        "web_search",
        "academic",
    ]:
        tasks["mcp_research"] = {
            "query": query,
            "intent": "web_research",
            "tools": ["search", "search_semantic"]
            if intent["primary_intent"] == "academic"
            else ["search"],
            "max_results": 8,
            "priority": "high"
            if intent["primary_intent"] in ["web_search", "academic"]
            else "normal",
        }

    return tasks


async def retrieval_agent(
    state: DockerMultiAgentRAGState,
) -> Dict[str, Any]:
    """Retrieval Agent: Performs vector search and returns metadata-only results.

    This agent integrates with the existing RAG system but only returns safe metadata
    (no full content or LLM-generated answers) to maintain security boundaries.
    """
    from app.vector_store import VectorStore
    from app.ollama_client import OllamaClient
    from datetime import datetime

    start_time = datetime.utcnow()
    sanitized_query = state.get("sanitized_query", "")
    agent_tasks = state.get("agent_tasks", {})

    if not sanitized_query:
        logger.warning("Retrieval Agent: No sanitized query provided")
        return {"retrieved_results": []}

    # Get task configuration for this agent
    task_config = agent_tasks.get("retrieval", {})
    max_results = task_config.get("max_results", 10)
    document_ids = task_config.get("document_ids")  # Optional document filtering

    logger.info(
        f"Retrieval Agent: Searching for '{sanitized_query}' (max_results: {max_results})"
    )

    try:
        # Initialize components (similar to RAG agent but without LLM)
        vector_store = VectorStore()
        ollama_client = OllamaClient()

        await vector_store.initialize_schema()

        # Generate query embedding
        query_embeddings = await ollama_client.embed_batch([sanitized_query])
        query_vector = query_embeddings[0]

        # Perform vector search (similar to RAG agent)
        search_filters = {}
        if document_ids:
            # Note: Current vector store implementation doesn't support document filtering
            # This would need to be enhanced in the vector store
            logger.info(f"Document filtering requested for IDs: {document_ids}")

        search_results = await vector_store.similarity_search(
            query_vector=query_vector,
            top_k=max_results,
            filters=search_filters,
            threshold=0.4,  # Similarity threshold
        )

        # Convert to results with content for response generation
        retrieved_results = []
        for result in search_results:
            result_entry = {
                "document_id": result["document_id"],
                "filename": result["filename"],
                "content_type": result["content_type"],
                "content": result[
                    "content"
                ],  # Include actual content for response generation
                "size": result.get("size", 0),
                "uploaded_at": result.get("uploaded_at", ""),
                "similarity_score": result["similarity_score"],
                "chunk_count": result.get("chunk_count", 0),
                "metadata": result.get("metadata", {}),
            }
            retrieved_results.append(result_entry)

        # Calculate confidence based on best similarity score
        if retrieved_results:
            similarity_scores = []
            for r in retrieved_results:
                score = r.get("similarity_score")
                if isinstance(score, (int, float)):
                    similarity_scores.append(float(score))
            confidence_score = max(similarity_scores) if similarity_scores else 0.0
        else:
            confidence_score = 0.0

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"Retrieval Agent: Found {len(retrieved_results)} results in {processing_time:.2f}s"
        )
        logger.info(
            f"Retrieval Agent: Sample result: {retrieved_results[0] if retrieved_results else 'No results'}"
        )

        return {
            "retrieved_results": retrieved_results,
            "retrieval_confidence": confidence_score,
            "retrieval_count": len(retrieved_results),
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Retrieval Agent failed: {e}")
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return {
            "retrieved_results": [],
            "retrieval_confidence": 0.0,
            "retrieval_count": 0,
            "processing_time": processing_time,
            "error": f"Retrieval failed: {str(e)}",
        }


async def mcp_research_agent(
    state: DockerMultiAgentRAGState,
) -> Dict[str, Any]:
    """MCP Research Agent: Performs external research using MCP tools.

    This agent executes web searches, academic paper searches, and other research
    tasks using the direct MCP tool integration. It orchestrates multiple tools
    concurrently and aggregates results for downstream processing.

    Args:
        state: Current LangGraph state

    Returns:
        Dictionary containing research results and metadata
    """
    from datetime import datetime
    import asyncio

    start_time = datetime.utcnow()
    sanitized_query = state.get("sanitized_query", "")
    agent_tasks = state.get("agent_tasks", {})

    if not sanitized_query:
        logger.warning("MCP Research Agent: No sanitized query provided")
        return {"mcp_search_results": {}}

    # Get research task configuration
    research_task = agent_tasks.get("mcp_research", {})
    if not research_task:
        logger.info("MCP Research Agent: No research task assigned, skipping")
        return {"mcp_search_results": {}}

    query = research_task.get("query", sanitized_query)
    tools_config = research_task.get("tools", [])
    max_results = research_task.get("max_results", 8)

    logger.info(f"MCP Research Agent: Researching '{query}' with tools: {tools_config}")

    # Prepare tool execution requests
    tool_requests = []

    # Map intent-based tool selection to actual tool names
    tool_mapping = {
        "search": "search",
        "search_semantic": "search_semantic",
        "search_arxiv": "search_arxiv",
        "search_biorxiv": "search_biorxiv",
    }

    for tool in tools_config:
        if tool in tool_mapping:
            actual_tool = tool_mapping[tool]
            if actual_tool == "search":
                tool_requests.append(
                    {"tool_name": actual_tool, "parameters": {"query": query}}
                )
            elif actual_tool == "search_semantic":
                tool_requests.append(
                    {
                        "tool_name": actual_tool,
                        "parameters": {
                            "query": query,
                            "max_results": min(max_results, 5),
                        },
                    }
                )
            elif actual_tool in ["search_arxiv", "search_biorxiv"]:
                tool_requests.append(
                    {
                        "tool_name": actual_tool,
                        "parameters": {
                            "query": query,
                            "max_results": min(max_results, 3),
                        },
                    }
                )

    # Add time reference tool for temporal context
    tool_requests.append(
        {"tool_name": "get_current_time", "parameters": {"timezone": "UTC"}}
    )

    # Execute tools concurrently using direct MCP integration
    logger.info(f"Executing {len(tool_requests)} MCP research tools concurrently")
    tool_results = await execute_mcp_research_tools(tool_requests)

    # Aggregate and rank results
    aggregated_results = aggregate_research_results(tool_results, max_results)

    processing_time = (datetime.utcnow() - start_time).total_seconds()

    logger.info(
        f"MCP Research Agent: Completed research in {processing_time:.2f}s - "
        f"found {aggregated_results['total_results']} results"
    )

    return {
        "mcp_search_results": aggregated_results,
        "research_execution_time": processing_time,
        "research_tools_used": len(tool_requests),
        "research_timestamp": datetime.utcnow().isoformat(),
    }


async def execute_mcp_research_tools(
    tool_requests: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Execute MCP research tools concurrently using direct integration.

    Args:
        tool_requests: List of tool execution requests

    Returns:
        List of tool execution results
    """
    import asyncio
    from app.main import call_mcp_tool_directly

    # Create concurrent tasks for all tool executions
    tasks = []
    for request in tool_requests:
        tool_name = request["tool_name"]
        parameters = request["parameters"]
        task = call_mcp_tool_directly(tool_name, parameters)
        tasks.append(task)

    # Execute all tools concurrently
    logger.info(f"Starting concurrent execution of {len(tasks)} research tools")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results, handling exceptions
    processed_results = []
    for i, result in enumerate(results):
        tool_request = tool_requests[i]
        if isinstance(result, Exception):
            logger.warning(
                f"MCP research tool {tool_request['tool_name']} failed: {result}"
            )
            # Create error result
            processed_results.append(
                {
                    "tool_name": tool_request["tool_name"],
                    "error": str(result),
                    "success": False,
                    "execution_time": 0.0,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        else:
            processed_results.append(result)

    return processed_results


def aggregate_research_results(
    tool_results: List[Dict[str, Any]], max_results: int = 8
) -> Dict[str, Any]:
    """Aggregate and rank research results from multiple MCP tools.

    Args:
        tool_results: Raw results from MCP tool executions
        max_results: Maximum total results to return

    Returns:
        Aggregated and ranked research results
    """
    aggregated = {
        "web_search": [],
        "academic_papers": [],
        "arxiv_papers": [],
        "biorxiv_papers": [],
        "time_reference": None,
        "total_results": 0,
        "successful_tools": 0,
        "failed_tools": 0,
        "aggregation_timestamp": datetime.utcnow().isoformat(),
    }

    # Process each tool result
    for result in tool_results:
        if not result.get("success", False):
            aggregated["failed_tools"] += 1
            continue

        tool_name = result.get("tool_name")
        tool_data = result.get("result", {})

        if tool_name == "search":
            # Web search results
            web_results = tool_data.get("results", [])
            for item in web_results:
                aggregated["web_search"].append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("summary", ""),
                        "source": "web_search",
                        "tool": "search",
                        "relevance_score": 0.8,  # Placeholder scoring
                    }
                )

        elif tool_name == "search_semantic":
            # Academic papers from semantic search
            papers = tool_data.get("papers", [])
            for paper in papers:
                aggregated["academic_papers"].append(
                    {
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "abstract": paper.get("abstract", ""),
                        "url": paper.get("url", ""),
                        "year": paper.get("year", ""),
                        "source": "semantic_search",
                        "tool": "search_semantic",
                        "relevance_score": 0.9,  # Higher relevance for semantic search
                    }
                )

        elif tool_name == "search_arxiv":
            # ArXiv papers
            papers = tool_data.get("papers", [])
            for paper in papers:
                aggregated["arxiv_papers"].append(
                    {
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "abstract": paper.get("abstract", ""),
                        "url": paper.get("url", ""),
                        "arxiv_id": paper.get("arxiv_id", ""),
                        "categories": paper.get("categories", []),
                        "published": paper.get("published", ""),
                        "source": "arxiv",
                        "tool": "search_arxiv",
                        "relevance_score": 0.85,
                    }
                )

        elif tool_name == "search_biorxiv":
            # bioRxiv papers
            papers = tool_data.get("papers", [])
            for paper in papers:
                aggregated["biorxiv_papers"].append(
                    {
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "abstract": paper.get("abstract", ""),
                        "url": paper.get("url", ""),
                        "doi": paper.get("doi", ""),
                        "categories": paper.get("categories", []),
                        "published": paper.get("published", ""),
                        "source": "biorxiv",
                        "tool": "search_biorxiv",
                        "relevance_score": 0.85,
                    }
                )

        elif tool_name == "get_current_time":
            # Time reference
            aggregated["time_reference"] = {
                "utc_time": tool_data.get("utc_time"),
                "requested_timezone": tool_data.get("requested_timezone"),
                "reference": tool_data.get("reference"),
                "tool": "get_current_time",
            }

        aggregated["successful_tools"] += 1

    # Apply result limits and calculate totals
    # Prioritize higher relevance scores and limit per category
    for category in ["web_search", "academic_papers", "arxiv_papers", "biorxiv_papers"]:
        results = aggregated[category]
        # Sort by relevance score descending
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        # Limit results per category
        max_per_category = max_results // 4  # Distribute across categories
        aggregated[category] = results[:max_per_category]

    # Calculate total results
    aggregated["total_results"] = sum(
        len(aggregated[cat])
        for cat in ["web_search", "academic_papers", "arxiv_papers", "biorxiv_papers"]
    )

    logger.info(
        f"Aggregated research results: {aggregated['total_results']} total, "
        f"{aggregated['successful_tools']} successful tools, {aggregated['failed_tools']} failed"
    )

    return aggregated


async def results_aggregator_placeholder(
    state: DockerMultiAgentRAGState,
) -> Dict[str, Any]:
    """Platzhalter für Results Aggregator."""
    logger.info("Results Aggregator placeholder executed")

    # Combine retrieval and MCP results
    retrieved_results = state.get("retrieved_results", [])
    mcp_results = state.get("mcp_search_results", {})

    logger.info(
        f"Results Aggregator: Found {len(retrieved_results)} retrieved results, "
        f"MCP results: {bool(mcp_results)}"
    )
    logger.info(
        f"Results Aggregator: mcp_results keys: {list(mcp_results.keys()) if mcp_results else 'None'}"
    )
    logger.info(f"Results Aggregator: Full state keys: {list(state.keys())}")
    if retrieved_results:
        logger.info(
            f"Results Aggregator: First retrieved result keys: {list(retrieved_results[0].keys())}"
        )
        logger.info(
            f"Results Aggregator: First retrieved result filename: {retrieved_results[0].get('filename', 'N/A')}"
        )

    # Create sources from available data (must match QuerySource model)
    sources = []

    # Add retrieval results
    for result in retrieved_results:
        sources.append(
            {
                "document_id": result.get("document_id", ""),
                "filename": result.get("filename", ""),
                "content_type": result.get(
                    "content_type", "application/pdf"
                ),  # Default if not available
                "chunk_text": result.get(
                    "content",
                    f"Retrieved document: {result.get('filename', 'Unknown')}. Similarity: {result.get('similarity_score', 0.0):.2f}",
                ),  # Use actual content
                "similarity_score": result.get("similarity_score", 0.0),
                "metadata": {
                    "source": "retrieval",
                    "size": result.get("size", 0),
                    "uploaded_at": result.get("uploaded_at", ""),
                    "chunk_count": result.get("chunk_count", 0),
                },
            }
        )

    # Add MCP results if available
    if mcp_results and "web_search" in mcp_results:
        for item in mcp_results["web_search"][:3]:  # Limit to 3
            sources.append(
                {
                    "document_id": f"mcp_web_{hash(item.get('url', '')) % 1000}",
                    "filename": item.get("title", "Web Result")[:50],
                    "content_type": "text/html",  # Web content
                    "chunk_text": item.get(
                        "snippet", "Web search result without snippet"
                    ),  # Use snippet as content
                    "similarity_score": item.get("relevance_score", 0.7),
                    "metadata": {
                        "source": "web_search",
                        "url": item.get("url", ""),
                        "tool": item.get("tool", "search"),
                    },
                }
            )

    # Calculate confidence based on available sources
    confidence_score = 0.5  # Default
    if sources:
        avg_similarity = sum(s.get("similarity_score", 0) for s in sources) / len(
            sources
        )
        confidence_score = min(0.9, avg_similarity + 0.2)  # Boost confidence slightly

    return {
        "confidence_score": confidence_score,
        "sources": sources[:5],  # Limit total sources
    }


async def response_generator_placeholder(
    state: DockerMultiAgentRAGState,
) -> Dict[str, Any]:
    """Response Generator: Creates LLM-powered answers using retrieved content."""
    from app.ollama_client import OllamaClient
    from app.config import settings
    from datetime import datetime

    logger.info("Response Generator executing with LLM")

    start_time = datetime.utcnow()
    sanitized_query = state.get("sanitized_query", "Unknown query")
    sources = state.get("sources", [])
    confidence_score = state.get("confidence_score", 0.0)

    try:
        # Prepare context from sources
        context_parts = []
        retrieval_sources = []

        for source in sources:
            if source.get("metadata", {}).get("source") == "retrieval":
                # Use the chunk_text which now contains actual content
                context_parts.append(source.get("chunk_text", ""))
                retrieval_sources.append(source)

        context = "\n\n".join(context_parts) if context_parts else ""

        # If we have retrieval sources with content, generate LLM response
        if context and retrieval_sources:
            ollama_client = OllamaClient()

            system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from documents.
If the context doesn't contain enough information to answer the question, say so clearly but also provide any relevant insights you can derive.

Context from documents:
{context}

Answer the user's question based on the context above. Be concise but comprehensive. Include specific references to the source documents when relevant."""

            generate_request = OllamaGenerateRequest(
                model=settings.ollama_model,
                prompt=sanitized_query,
                system=system_prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent answers
                    "top_p": 0.9,
                    "num_predict": 1024,
                },
            )

            completion = await ollama_client.generate(generate_request)
            answer = completion.response.strip()

        else:
            # Fallback response when no content available
            answer = f"I found {len(sources)} relevant source(s) for your query: '{sanitized_query}'. "
            if confidence_score > 0.7:
                answer += "I'm fairly confident in these results."
            else:
                answer += "These results may need further verification."

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return {
            "final_response": answer,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        # Fallback to simple response
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        fallback_response = f"I found {len(sources)} relevant source(s) for your query: '{sanitized_query}'. "
        if confidence_score > 0.7:
            fallback_response += "I'm fairly confident in these results."
        else:
            fallback_response += "These results may need further verification."

        return {
            "final_response": fallback_response,
            "processing_time": processing_time,
            "error": f"LLM generation failed: {str(e)}",
        }


async def validation_agent_placeholder(
    state: DockerMultiAgentRAGState,
) -> Dict[str, Any]:
    """Platzhalter für Validation Agent."""
    logger.info("Validation Agent placeholder executed")

    # Basic validation: check if we have a response and sources
    final_response = state.get("final_response", "")
    sources = state.get("sources", [])
    confidence_score = state.get("confidence_score", 0.0)

    # Simple validation logic
    validation_passed = bool(final_response and len(final_response) > 10)
    if sources:
        validation_passed = validation_passed and (confidence_score > 0.3)

    # Update processing time to include validation
    processing_time = state.get("processing_time", 0.0) + 0.1

    return {
        "processing_complete": validation_passed,
        "validation_passed": validation_passed,
        "processing_time": processing_time,
    }


def route_after_query_processor(state: DockerMultiAgentRAGState) -> str:
    """Route decision after query processor based on agent tasks."""
    agent_tasks = state.get("agent_tasks", {})

    # Always run retrieval first as baseline RAG (it's always included in create_agent_tasks)
    if "retrieval" in agent_tasks:
        return "retrieval_agent"

    # Fallback: should not happen since retrieval is always included
    logger.warning("No retrieval task found, routing to results aggregator")
    return "results_aggregator"


def create_docker_multi_agent_graph(mcp_client: DockerMCPClient = None) -> StateGraph:
    """Erstelle Basis-Multi-Agenten-Graph mit Platzhaltern für Phase 1.

    Args:
        mcp_client: MCP Client für Tool-Ausführung (optional für Phase 1)

    Returns:
        Konfigurierter StateGraph
    """
    logger.info("Creating Docker Multi-Agent RAG Graph (Phase 1)")

    # Erstelle Graph mit State-Definition
    builder = StateGraph(DockerMultiAgentRAGState)

    # Füge Agenten-Nodes hinzu (Phase 1: Platzhalter)
    builder.add_node("query_processor", query_processor_agent)
    builder.add_node("retrieval_agent", retrieval_agent)
    builder.add_node("mcp_search_agent", mcp_research_agent)
    builder.add_node("results_aggregator", results_aggregator_placeholder)
    builder.add_node("response_generator", response_generator_placeholder)
    builder.add_node("validation_agent", validation_agent_placeholder)

    # Definiere Ausführungsfluss mit bedingtem Routing
    # Query processor routes to either retrieval or directly to aggregator
    builder.add_conditional_edges(
        "query_processor",
        route_after_query_processor,
        {
            "retrieval_agent": "retrieval_agent",
            "results_aggregator": "results_aggregator",
        },
    )

    # Retrieval agent always goes to MCP search (parallel execution)
    builder.add_edge("retrieval_agent", "mcp_search_agent")

    # MCP search goes to results aggregator
    builder.add_edge("mcp_search_agent", "results_aggregator")

    # Continue with response generation
    builder.add_edge("results_aggregator", "response_generator")
    builder.add_edge("response_generator", "validation_agent")
    builder.add_edge("validation_agent", END)

    # Setze Startpunkt
    builder.set_entry_point("query_processor")

    logger.info("Docker Multi-Agent Graph created with conditional routing")
    return builder


async def test_graph_execution():
    """Test-Funktion für Graph-Ausführung (Phase 1)."""
    from app.multi_agent_state import create_initial_state

    # Erstelle Test-Graph
    graph = create_docker_multi_agent_graph()
    compiled_graph = graph.compile()

    # Erstelle Test-State
    initial_state = create_initial_state("What is machine learning?")

    # Führe Graph aus
    logger.info("Testing graph execution...")
    result = await compiled_graph.ainvoke(initial_state)

    logger.info(
        f"Graph execution completed: {result.get('processing_complete', False)}"
    )
    return result


if __name__ == "__main__":
    # Für direkten Test des Moduls
    import asyncio

    asyncio.run(test_graph_execution())
