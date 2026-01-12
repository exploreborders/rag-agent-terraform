"""Pydantic data models for the RAG Agent API."""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata for a processed document."""

    filename: str
    content_type: str
    size: int
    upload_time: str
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    checksum: Optional[str] = None


class DocumentResponse(BaseModel):
    """Response model for document operations."""

    id: str
    metadata: DocumentMetadata
    status: str = "processed"
    chunks_count: int = 0
    embeddings_count: int = 0


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(
        ..., min_length=1, max_length=1000, description="User query text"
    )
    document_ids: Optional[List[str]] = Field(
        default_factory=list, description="Specific document IDs to search"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of top results to return"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional search filters"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query content."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        # Prevent injection attacks
        if re.search(r"[<>]", v):
            raise ValueError("Query contains invalid characters")
        return v.strip()


class QuerySource(BaseModel):
    """Source document information for query results."""

    document_id: str
    filename: str
    content_type: str
    chunk_text: str
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    query: str
    answer: str
    sources: List[QuerySource] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    total_sources: int = 0


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall health status")
    timestamp: str
    version: str = "0.1.0"
    services: Dict[str, str] = Field(
        default_factory=dict, description="Individual service statuses"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str


class AgentMemory(BaseModel):
    """Agent conversation memory model."""

    session_id: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search."""

    query_vector: List[float] = Field(
        ..., description="Query vector for similarity search"
    )
    top_k: int = Field(
        default=5, ge=1, le=50, description="Number of results to return"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Search filters"
    )
    threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Similarity threshold"
    )


class VectorSearchResult(BaseModel):
    """Result model for vector similarity search."""

    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    similarity_score: float
    distance: Optional[float] = None


class DocumentChunk(BaseModel):
    """Document chunk model for processing."""

    id: str
    document_id: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str


class OllamaModelInfo(BaseModel):
    """Information about available Ollama models."""

    name: str
    size: str
    modified_at: str
    digest: str


class OllamaEmbedRequest(BaseModel):
    """Request model for Ollama embeddings."""

    model: str = Field(..., description="Model name for embeddings")
    prompt: str = Field(..., description="Text to embed")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Model options"
    )


class OllamaEmbedResponse(BaseModel):
    """Response model for Ollama embeddings."""

    embedding: List[float]
    model: str
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None


class OllamaGenerateRequest(BaseModel):
    """Request model for Ollama text generation."""

    model: str = Field(..., description="Model name for generation")
    prompt: str = Field(..., description="Text prompt for generation")
    system: Optional[str] = Field(default=None, description="System message")
    template: Optional[str] = Field(default=None, description="Prompt template")
    context: Optional[List[int]] = Field(
        default=None, description="Context from previous conversation"
    )
    stream: bool = Field(default=False, description="Stream response")
    raw: bool = Field(default=False, description="Return raw response")
    format: Optional[str] = Field(default=None, description="Response format")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Model options"
    )


class OllamaGenerateResponse(BaseModel):
    """Response model for Ollama text generation."""

    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
