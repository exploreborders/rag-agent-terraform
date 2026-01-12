"""FastAPI application for the RAG Agent system."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.config import settings
from app.models import (
    QueryRequest,
    QueryResponse,
    DocumentResponse,
    HealthStatus,
    ErrorResponse,
)
from app.rag_agent import RAGAgent, RAGAgentError
from app.document_loader import DocumentProcessingError, UnsupportedFileTypeError

# Configure structured logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
shared_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
]
structlog.configure(
    processors=shared_processors,
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global RAG agent instance
rag_agent: Optional[RAGAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    global rag_agent

    # Startup
    logger.info("Starting RAG Agent API server", version=settings.version)

    try:
        rag_agent = RAGAgent()
        await rag_agent.initialize()
        logger.info("RAG Agent initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize RAG Agent", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG Agent API server")
    if rag_agent:
        await rag_agent.cleanup()


# Create FastAPI application
app = FastAPI(
    title="RAG Agent Terraform API",
    description="API for the RAG Agent system with document processing and Q&A capabilities",
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


async def get_rag_agent() -> RAGAgent:
    """Dependency to get the RAG agent instance."""
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="RAG Agent not initialized")
    return rag_agent


@app.get("/health", response_model=HealthStatus)
async def health_check(agent: RAGAgent = Depends(get_rag_agent)):
    """Health check endpoint for all services."""
    try:
        health = await agent.health_check()
        status_code = 200 if health.status == "healthy" else 503
        return JSONResponse(status_code=status_code, content=health.dict())
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content=HealthStatus(
                status="unhealthy", timestamp="unknown", services={"error": str(e)}
            ).dict(),
        )


@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...), agent: RAGAgent = Depends(get_rag_agent)
):
    """Upload and process a document for the RAG system."""
    start_time = time.time()

    try:
        # Validate file type
        if file.content_type not in agent.document_loader.get_supported_types():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported types: {agent.document_loader.get_supported_types()}",
            )

        # Validate file size
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size} bytes. Maximum size: {settings.max_upload_size} bytes",
            )

        # Save uploaded file
        file_path = agent.document_loader.save_uploaded_file(
            type("FileObj", (), {"read": lambda: content})(), file.filename
        )

        # Process document
        result = await agent.process_document(file_path.name, file.content_type)

        processing_time = time.time() - start_time
        logger.info(
            "Document uploaded and processed",
            filename=file.filename,
            file_size=file_size,
            chunks_count=result.chunks_count,
            processing_time=processing_time,
        )

        return result

    except UnsupportedFileTypeError as e:
        logger.warning(
            "Unsupported file type",
            filename=file.filename,
            content_type=file.content_type,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except DocumentProcessingError as e:
        logger.error("Document processing failed", filename=file.filename, error=str(e))
        raise HTTPException(
            status_code=422, detail=f"Document processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(
            "Unexpected error during document upload",
            filename=file.filename,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    session_id: Optional[str] = Query(None, description="Conversation session ID"),
    agent: RAGAgent = Depends(get_rag_agent),
):
    """Query the RAG system for answers based on processed documents."""
    start_time = time.time()

    try:
        # Validate request
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Execute query
        result = await agent.query(
            query=request.query,
            document_ids=request.document_ids,
            top_k=request.top_k,
            session_id=session_id,
        )

        processing_time = time.time() - start_time
        logger.info(
            "RAG query processed",
            query_length=len(request.query),
            sources_count=len(result.sources),
            processing_time=processing_time,
            session_id=session_id,
        )

        return result

    except RAGAgentError as e:
        logger.error("RAG query failed", query=request.query, error=str(e))
        raise HTTPException(
            status_code=422, detail=f"Query processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error("Unexpected error during query", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents(
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of documents to return"
    ),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    agent: RAGAgent = Depends(get_rag_agent),
):
    """List all processed documents."""
    try:
        documents = await agent.list_documents(limit=limit, offset=offset)
        logger.info(
            "Documents listed", count=len(documents), limit=limit, offset=offset
        )
        return documents
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@app.get("/documents/{document_id}", response_model=Dict[str, Any])
async def get_document(document_id: str, agent: RAGAgent = Depends(get_rag_agent)):
    """Get details of a specific document."""
    try:
        document = await agent.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        logger.info("Document retrieved", document_id=document_id)
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, agent: RAGAgent = Depends(get_rag_agent)):
    """Delete a document and its associated chunks."""
    try:
        deleted = await agent.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")

        logger.info("Document deleted", document_id=document_id)
        return {"message": "Document deleted successfully", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete document")


@app.get("/stats")
async def get_stats(agent: RAGAgent = Depends(get_rag_agent)):
    """Get system statistics."""
    try:
        stats = await agent.get_stats()
        logger.info("System stats retrieved", stats=stats)
        return stats
    except Exception as e:
        logger.error("Failed to get system stats", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve system statistics"
        )


@app.post("/cache/clear")
async def clear_cache(agent: RAGAgent = Depends(get_rag_agent)):
    """Clear all system caches."""
    try:
        cleared = await agent.clear_cache()
        if cleared:
            logger.info("System caches cleared")
            return {"message": "All caches cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear caches")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to clear caches", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to clear caches")


# Error handlers
@app.exception_handler(RAGAgentError)
async def rag_agent_exception_handler(request, exc: RAGAgentError):
    """Handle RAG agent specific errors."""
    logger.error("RAG Agent error", error=str(exc))
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="RAG Agent Error", message=str(exc), timestamp="now"
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            timestamp="now",
        ).dict(),
    )


# Startup message
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info(
        "RAG Agent API started",
        host=settings.api_host,
        port=settings.api_port,
        environment=settings.environment,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
