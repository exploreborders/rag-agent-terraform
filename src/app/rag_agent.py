"""Hybrid RAG agent combining LangChain orchestration with LlamaIndex indexing."""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import settings
from app.document_loader import DocumentLoader
from app.models import (
    DocumentResponse,
    HealthStatus,
    OllamaGenerateRequest,
    QueryResponse,
    QuerySource,
)
from app.ollama_client import OllamaClient
from app.redis_memory import AgentMemory
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGAgentError(Exception):
    """Base exception for RAG agent errors."""

    pass


class RAGAgent:
    """Hybrid RAG agent combining LangChain orchestration with LlamaIndex indexing."""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        vector_store: Optional[VectorStore] = None,
        memory: Optional[AgentMemory] = None,
        document_loader: Optional[DocumentLoader] = None,
    ):
        """Initialize the RAG agent.

        Args:
            ollama_client: Ollama client for AI model integration
            vector_store: Vector database for document embeddings
            memory: Redis-backed memory for conversation context
            document_loader: Document processing pipeline
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.vector_store = vector_store or VectorStore()
        self.memory = memory or AgentMemory()
        self.document_loader = document_loader or DocumentLoader()

        # Initialize components
        self._initialized = False

        logger.info(
            "RAG Agent initialized with hybrid LangChain/LlamaIndex architecture"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return

        try:
            # Initialize vector store schema
            await self.vector_store.initialize_schema()

            # Mark as initialized
            self._initialized = True
            logger.info("RAG Agent fully initialized")

        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
            raise RAGAgentError(f"Initialization failed: {e}")

    async def cleanup(self):
        """Clean up resources."""
        # Components handle their own cleanup via context managers
        self._initialized = False
        logger.info("RAG Agent cleaned up")

    async def _ensure_initialized(self):
        """Ensure the agent is initialized."""
        if not self._initialized:
            await self.initialize()

    def _generate_query_hash(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a hash for query caching.

        Args:
            query: Query text
            filters: Query filters

        Returns:
            SHA-256 hash string
        """
        content = f"{query}|{str(sorted(filters.items()) if filters else '')}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def process_document(
        self, file_path: str, content_type: str
    ) -> DocumentResponse:
        """Process and index a document.

        Args:
            file_path: Path to the document file
            content_type: MIME content type

        Returns:
            Document processing response
        """
        await self._ensure_initialized()

        try:
            # Process document
            doc_path = self.document_loader.upload_dir / file_path
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found: {doc_path}")

            processed_doc = self.document_loader.process_document(
                doc_path, content_type
            )

            # Store document metadata
            doc_metadata = processed_doc["metadata"]
            document_id = await self.vector_store.store_document(
                processed_doc["document_id"], doc_metadata
            )

            # Generate embeddings for chunks
            chunks_data = []
            for chunk in processed_doc["chunks"]:
                # Check cache first
                text_hash = hashlib.sha256(chunk["content"].encode()).hexdigest()
                cached_embedding = await self.memory.get_cached_embedding(text_hash)

                if cached_embedding:
                    embedding = cached_embedding
                else:
                    # Generate new embedding
                    embeddings = await self.ollama_client.embed_batch(
                        [chunk["content"]]
                    )
                    embedding = embeddings[0]

                    # Cache embedding
                    await self.memory.cache_embedding(text_hash, embedding)

                # Add embedding to chunk
                chunk["embedding"] = embedding
                chunks_data.append(chunk)

            # Store chunks with embeddings
            chunks_stored = await self.vector_store.store_chunks(chunks_data)

            # Create response
            response = DocumentResponse(
                id=document_id,
                metadata=doc_metadata,
                status="processed",
                chunks_count=len(chunks_data),
                embeddings_count=chunks_stored,
            )

            logger.info(f"Successfully processed document: {doc_metadata['filename']}")
            return response

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise RAGAgentError(f"Document processing failed: {e}")

    async def query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> QueryResponse:
        """Process a RAG query.

        Args:
            query: User query text
            document_ids: Specific document IDs to search
            top_k: Number of top results to return
            session_id: Conversation session ID
            **kwargs: Additional parameters

        Returns:
            Query response with answer and sources
        """
        await self._ensure_initialized()

        start_time = datetime.utcnow()

        try:
            # Check cache first
            query_hash = self._generate_query_hash(
                query, {"document_ids": document_ids}
            )
            cached_result = await self.memory.get_cached_query_result(query_hash)

            if cached_result:
                logger.info("Returning cached query result")
                return QueryResponse(**cached_result)

            # Generate query embedding
            query_embeddings = await self.ollama_client.embed_batch([query])
            query_vector = query_embeddings[0]

            # Perform vector search
            search_filters = {}
            if document_ids:
                # For multiple documents, we'll need to search each one
                # For now, search all and filter results
                pass

            search_results = await self.vector_store.similarity_search(
                query_vector=query_vector,
                top_k=top_k * 2,  # Get more results for filtering
                filters=search_filters,
                threshold=settings.similarity_threshold,
            )

            # Filter by document_ids if specified
            if document_ids:
                filtered_results = [
                    result
                    for result in search_results
                    if result["document_id"] in document_ids
                ][:top_k]
            else:
                filtered_results = search_results[:top_k]

            # Prepare context from search results
            context_parts = []
            sources = []

            for result in filtered_results:
                context_parts.append(result["content"])
                sources.append(
                    QuerySource(
                        document_id=result["document_id"],
                        filename=result["filename"],
                        content_type=result["content_type"],
                        chunk_text=result["content"],
                        similarity_score=result["similarity_score"],
                        metadata=result.get("metadata", {}),
                    )
                )

            context = "\n\n".join(context_parts)

            # Get conversation context if session provided
            conversation_context = ""
            if session_id:
                conversation = await self.memory.get_conversation(session_id)
                if conversation and conversation.get("messages"):
                    # Use last few messages for context
                    recent_messages = conversation["messages"][-4:]  # Last 4 messages
                    conversation_context = "\n".join(
                        [
                            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                            for msg in recent_messages
                        ]
                    )

            # Generate response using LLM
            system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer the question, say so clearly.

Context from documents:
{context}

{f"Previous conversation:{conversation_context}" if conversation_context else ""}

Answer the user's question based on the context above. Be concise but comprehensive."""

            generate_request = {
                "model": settings.ollama_model,
                "prompt": query,
                "system": system_prompt,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent answers
                    "top_p": 0.9,
                    "num_predict": 512,
                },
            }

            # For now, we'll implement a simple completion
            # In a full implementation, this would use streaming
            completion = await self.ollama_client.generate(
                OllamaGenerateRequest(**generate_request)
            )

            answer = completion.response.strip()

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Create response
            response = QueryResponse(
                query=query,
                answer=answer,
                sources=sources,
                confidence_score=(
                    sum(s.similarity_score for s in sources) / len(sources)
                    if sources
                    else 0.0
                ),
                processing_time=processing_time,
                total_sources=len(sources),
            )

            # Cache the result
            await self.memory.cache_query_result(query_hash, response.dict())

            # Update conversation memory if session provided
            if session_id:
                messages = [
                    {
                        "role": "user",
                        "content": query,
                        "timestamp": start_time.isoformat(),
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ]
                await self.memory.update_conversation(session_id, messages)

            logger.info(
                f"Processed query in {processing_time:.2f}s, found {len(sources)} sources"
            )
            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise RAGAgentError(f"Query processing failed: {e}")

    async def list_documents(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all processed documents.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of document metadata
        """
        await self._ensure_initialized()
        return await self.vector_store.list_documents(limit=limit, offset=offset)

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document metadata or None
        """
        await self._ensure_initialized()
        return await self.vector_store.get_document(document_id)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks.

        Args:
            document_id: Document identifier

        Returns:
            True if deleted successfully
        """
        await self._ensure_initialized()
        return await self.vector_store.delete_document(document_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics.

        Returns:
            Dictionary with system statistics
        """
        await self._ensure_initialized()

        try:
            # Get document and chunk counts
            total_chunks = await self.vector_store.get_chunk_count()
            total_docs = len(await self.vector_store.list_documents(limit=1000))

            # Get cache stats
            cache_stats = await self.memory.get_stats()

            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_queries": 42,  # Placeholder - would need query logging to track this
                "average_response_time": 1.2,  # Placeholder - would need response time tracking
                "uptime_seconds": 3600,  # Placeholder - would need uptime tracking
                "cache_stats": cache_stats,
                "ollama_available": await self.ollama_client.health_check(),
                "vector_store_healthy": await self.vector_store.health_check(),
                "memory_healthy": await self.memory.health_check(),
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> HealthStatus:
        """Comprehensive health check of all components.

        Returns:
            Health status of all services
        """
        services_status = {}

        try:
            services_status["ollama"] = (
                "healthy" if await self.ollama_client.health_check() else "unhealthy"
            )
        except Exception:
            services_status["ollama"] = "unhealthy"

        try:
            services_status["vector_store"] = (
                "healthy" if await self.vector_store.health_check() else "unhealthy"
            )
        except Exception:
            services_status["vector_store"] = "unhealthy"

        try:
            services_status["redis"] = (
                "healthy" if await self.memory.health_check() else "unhealthy"
            )
        except Exception:
            services_status["redis"] = "unhealthy"

        # Overall status
        overall_status = (
            "healthy"
            if all(status == "healthy" for status in services_status.values())
            else "degraded"
        )

        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            services=services_status,
        )

    async def clear_cache(self) -> bool:
        """Clear all caches.

        Returns:
            True if successful
        """
        try:
            await self.memory.clear_all_cache()
            logger.info("All caches cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
            return False
