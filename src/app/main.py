"""RAG Agent FastAPI Application with monitoring and metrics."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
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

# Global RAG agent instance
rag_agent: Optional[RAGAgent] = None


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
    """Health check endpoint."""
    try:
        # Check RAG agent health
        if rag_agent is None:
            raise HTTPException(status_code=503, detail="RAG Agent not initialized")

        # Check vector store connection
        await rag_agent.vector_store.health_check()

        # Check Ollama client
        await rag_agent.ollama_client.health_check()

        from datetime import datetime

        return HealthStatus(
            status="healthy",
            timestamp=datetime.utcnow().isoformat()[:19] + "Z",
            version=settings.version,
            services={
                "rag_agent": "healthy",
                "vector_store": "healthy",
                "ollama_client": "healthy",
            },
        )

        from datetime import datetime

        return HealthStatus(
            status="healthy",
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
    """Query the RAG system."""
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


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("RAG Agent application started")


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
