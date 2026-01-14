"""RAG Agent FastAPI Application with monitoring and metrics."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from starlette_exporter import PrometheusMiddleware, handle_metrics

from app.config import settings
from app.models import (
    HealthStatus,
    QueryRequest,
    QueryResponse,
)
from app.rag_agent import RAGAgent

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom registry for RAG metrics to avoid conflicts with starlette-exporter
rag_registry = CollectorRegistry()

# Metrics - will be initialized only when app starts
RAG_DOCUMENTS_PROCESSED = None
RAG_QUERIES_PROCESSED = None
HTTP_REQUESTS_TOTAL = None

# Global instances
rag_agent: Optional[RAGAgent] = None
multi_agent_graph = None  # LangGraph Multi-Agent System


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    global rag_agent, RAG_DOCUMENTS_PROCESSED, RAG_QUERIES_PROCESSED

    # Startup
    logger.info("Starting RAG Agent application...")
    try:
        # Initialize custom metrics after starlette-exporter
        global RAG_DOCUMENTS_PROCESSED, RAG_QUERIES_PROCESSED, HTTP_REQUESTS_TOTAL
        RAG_DOCUMENTS_PROCESSED = Counter(
            "rag_agent_documents_processed_total",
            "Total number of documents processed by RAG agent",
            registry=rag_registry,
        )
        RAG_QUERIES_PROCESSED = Counter(
            "rag_agent_queries_processed_total",
            "Total number of queries processed by RAG agent",
            registry=rag_registry,
        )

        # Initialize HTTP metrics
        HTTP_REQUESTS_TOTAL = Counter(
            "rag_http_requests_total",
            "Total number of HTTP requests",
            ["method", "endpoint", "status"],
            registry=rag_registry,
        )

        rag_agent = RAGAgent()
        logger.info("RAG Agent initialized successfully")

        # Initialize document counter with current count from database
        try:
            documents_data = await rag_agent.list_documents(limit=1000, offset=0)
            if RAG_DOCUMENTS_PROCESSED:
                RAG_DOCUMENTS_PROCESSED.inc(len(documents_data))
                logger.info(f"Initialized document counter to {len(documents_data)}")
        except Exception as e:
            logger.warning(f"Failed to initialize document counter: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Agent: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG Agent application...")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="Retrieval-Augmented Generation system with document processing capabilities",
    version=settings.version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom middleware to track HTTP requests
@app.middleware("http")
async def track_requests(request, call_next):
    """Middleware to track HTTP requests for metrics."""
    response = await call_next(request)

    # Track the request
    if HTTP_REQUESTS_TOTAL:
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        ).inc()

    return response


# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware, app_name="rag-agent", prefix="rag")
app.add_route("/metrics", handle_metrics)


# Custom metrics endpoint for RAG-specific metrics
@app.get("/metrics/rag")
async def rag_metrics():
    """Custom metrics endpoint for RAG-specific metrics."""
    from prometheus_client import generate_latest

    return Response(generate_latest(rag_registry), media_type="text/plain")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Agent API",
        "version": settings.version,
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Health check endpoint with Multi-Agent support."""
    try:
        services_status = {}

        # Check legacy RAG agent health
        if rag_agent is not None:
            try:
                await rag_agent.vector_store.health_check()
                services_status["vector_store"] = "healthy"
            except Exception:
                services_status["vector_store"] = "unhealthy"

            try:
                await rag_agent.ollama_client.health_check()
                services_status["ollama_client"] = "healthy"
            except Exception:
                services_status["ollama_client"] = "unhealthy"

            services_status["rag_agent"] = "healthy"
        else:
            services_status["rag_agent"] = "not_initialized"

        # Check Multi-Agent System
        services_status["multi_agent_system"] = (
            "healthy" if multi_agent_graph else "not_initialized"
        )

        # Check MCP Coordinator
        try:
            import httpx

            mcp_url = f"{settings.mcp_coordinator_url}/health"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(mcp_url)
                if response.status_code == 200:
                    services_status["mcp_coordinator"] = "healthy"
                else:
                    services_status["mcp_coordinator"] = "unhealthy"
        except Exception:
            services_status["mcp_coordinator"] = "unreachable"

        # Determine overall status
        critical_services = ["vector_store", "ollama_client", "rag_agent"]
        overall_status = (
            "healthy"
            if all(
                services_status.get(service) == "healthy"
                for service in critical_services
            )
            else "degraded"
        )

        from datetime import datetime

        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat()[:19] + "Z",
            version=settings.version,
            services=services_status,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """Upload and process a document."""
    try:
        if rag_agent is None:
            raise HTTPException(status_code=503, detail="RAG Agent not initialized")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Save uploaded file
        content = await file.read()

        # Write file to disk
        upload_dir = rag_agent.document_loader.upload_dir
        upload_dir.mkdir(exist_ok=True)
        file_location = upload_dir / file.filename

        with open(file_location, "wb") as f:
            f.write(content)

        # Process document
        result = await rag_agent.process_document(
            file_path=file.filename,
            content_type=file.content_type or "application/octet-stream",
        )

        if RAG_DOCUMENTS_PROCESSED:
            RAG_DOCUMENTS_PROCESSED.inc()

        return result

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Document processing failed: {str(e)}"
        )


@app.get("/documents")
async def list_documents(
    limit: int = 100,
    offset: int = 0,
):
    """List all uploaded documents."""
    try:
        # Get documents from RAG agent
        documents_data = await rag_agent.list_documents(limit=limit, offset=offset)

        # Convert to flat document objects for frontend compatibility
        documents = []
        for doc_data in documents_data:
            doc_flat = {
                "id": doc_data.get("id", ""),
                "filename": doc_data.get("filename", ""),
                "content_type": doc_data.get("content_type", ""),
                "size": doc_data.get("size", 0),
                "uploaded_at": doc_data.get("uploaded_at", ""),
                "status": doc_data.get("status", "processed"),
                "chunks_count": doc_data.get("chunks_count", 0),
            }
            documents.append(doc_flat)

        # Return just the documents array for frontend compatibility
        return documents

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        # Return empty array instead of error to keep frontend working
        return []


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    try:
        if rag_agent is None:
            raise HTTPException(status_code=503, detail="RAG Agent not initialized")

        success = await rag_agent.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system (Legacy Endpoint)."""
    try:
        if rag_agent is None:
            raise HTTPException(status_code=503, detail="RAG Agent not initialized")

        # Process query
        result = await rag_agent.query(
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            filters=request.filters,
        )

        if RAG_QUERIES_PROCESSED:
            RAG_QUERIES_PROCESSED.inc()

        return result

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.post("/agents/query")
async def multi_agent_query(request: QueryRequest):
    """Multi-Agenten-Abfrage mit LangGraph (Phase 1)."""
    try:
        if multi_agent_graph is None:
            # Fallback to legacy system
            logger.warning("Multi-Agent System not available, using legacy RAG")
            return await query_documents(request)

        # Create initial state
        from app.multi_agent_state import create_initial_state

        initial_state = create_initial_state(
            query=request.query,
            user_id="api_user",  # Simplified for Phase 1
            user_level="standard",
        )

        # Add query parameters
        if request.document_ids:
            initial_state["allowed_doc_ids"] = request.document_ids

        # Execute multi-agent graph
        logger.info(f"Executing multi-agent query: {request.query}")
        result = await multi_agent_graph.ainvoke(initial_state)

        # Format response
        response = {
            "query": request.query,
            "answer": result.get("final_response", "No response generated"),
            "sources": result.get("sources", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "processing_time": result.get("processing_time", 0.0),
            "total_sources": len(result.get("sources", [])),
            # Phase 1 specific fields
            "agent_metrics": result.get("agent_metrics", {}),
            "mcp_results": {
                "search": result.get("mcp_search_results"),
                "code": result.get("mcp_code_results"),
            },
            "phase": "multi_agent_v1",
        }

        # Update metrics
        if RAG_QUERIES_PROCESSED:
            RAG_QUERIES_PROCESSED.inc()

        return response

    except Exception as e:
        logger.error(f"Multi-agent query failed: {e}")
        # Fallback to legacy system
        logger.info("Falling back to legacy RAG system")
        return await query_documents(request)


@app.get("/agents/status")
async def agent_system_status():
    """Status des Multi-Agenten-Systems."""
    try:
        status_info = {
            "multi_agent_enabled": multi_agent_graph is not None,
            "legacy_rag_available": rag_agent is not None,
            "mcp_coordinator_connected": False,
            "agents": {
                "query_processor": "placeholder",
                "retrieval_agent": "placeholder",
                "mcp_search_agent": "placeholder",
                "mcp_code_agent": "placeholder",
                "results_aggregator": "placeholder",
                "response_generator": "placeholder",
                "validation_agent": "placeholder",
            },
        }

        # Check MCP Coordinator connectivity
        try:
            import httpx

            mcp_url = f"{settings.mcp_coordinator_url}/health"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(mcp_url)
                if response.status_code == 200:
                    status_info["mcp_coordinator_connected"] = True
        except Exception:
            pass

        return status_info

    except Exception as e:
        logger.error(f"Agent status check failed: {e}")
        return {"error": str(e), "multi_agent_enabled": False}


@app.post("/test-mcp-tools")
async def test_mcp_tools(
    query: str = Query("machine learning", description="Search query"),
):
    """Test MCP Tools: Paper Search and DuckDuckGo Search."""
    try:
        import httpx

        results = {
            "query": query,
            "tools_tested": ["search", "search_semantic", "get_current_time"],
            "results": {},
        }

        # Test Web Search (DuckDuckGo via MCP Toolkit)
        try:
            mcp_url = f"{settings.mcp_coordinator_url}/tools/search/execute"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(mcp_url, json={"query": query})
                if response.status_code == 200:
                    results["results"]["web_search"] = response.json()
                else:
                    results["results"]["web_search"] = {
                        "error": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            results["results"]["web_search"] = {"error": str(e)}

        # Test Academic Paper Search (Semantic Scholar)
        try:
            mcp_url = f"{settings.mcp_coordinator_url}/tools/search_semantic/execute"
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(mcp_url, json={"query": query})
                if response.status_code == 200:
                    results["results"]["academic_papers"] = response.json()
                else:
                    results["results"]["academic_papers"] = {
                        "error": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            results["results"]["academic_papers"] = {"error": str(e)}

        # Test Time Reference
        try:
            mcp_url = f"{settings.mcp_coordinator_url}/tools/get_current_time/execute"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(mcp_url, json={"timezone": "UTC"})
                if response.status_code == 200:
                    results["results"]["time_reference"] = response.json()
                else:
                    results["results"]["time_reference"] = {
                        "error": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            results["results"]["time_reference"] = {"error": str(e)}

        return results

    except Exception as e:
        logger.error(f"MCP tools test failed: {e}")
        return {"error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("RAG Agent application started")

    # Initialize Multi-Agent System (Phase 1)
    global multi_agent_graph
    try:
        # Import multi-agent modules
        from app.multi_agent_graph import create_docker_multi_agent_graph
        from app.graph_persistence import setup_graph_persistence
        from app.mcp_client import mcp_client

        # Setup graph persistence
        checkpointer = await setup_graph_persistence(settings.database_url)

        # Initialize MCP client
        await mcp_client.connect()

        # Create and compile multi-agent graph
        graph = create_docker_multi_agent_graph(mcp_client)
        multi_agent_graph = graph.compile(checkpointer=checkpointer)

        logger.info("Multi-Agent System initialized successfully")

    except Exception as e:
        logger.warning(f"Multi-Agent System initialization failed: {e}")
        logger.info("Continuing with legacy RAG system only")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("RAG Agent application shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
