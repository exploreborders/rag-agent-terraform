"""PostgreSQL vector store operations with pgvector."""

import logging
from typing import Any, Dict, List, Optional

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store errors."""

    pass


class VectorStoreConnectionError(VectorStoreError):
    """Exception raised when database connection fails."""

    pass


class VectorStoreQueryError(VectorStoreError):
    """Exception raised when vector queries fail."""

    pass


class VectorStore:
    """PostgreSQL vector store with pgvector extension."""

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize vector store.

        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string or settings.database_url
        self._pool: Optional[asyncpg.Pool] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self._pool = await asyncpg.create_pool(
                self.connection_string, min_size=1, max_size=10, command_timeout=60
            )

            # Register pgvector extension
            async with self._pool.acquire() as conn:
                await register_vector(conn)
                logger.info("Connected to PostgreSQL with pgvector support")

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise VectorStoreConnectionError(f"Database connection failed: {e}")

    async def disconnect(self):
        """Disconnect from database."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from PostgreSQL")

    async def _ensure_connection(self):
        """Ensure database connection is active."""
        if not self._pool:
            await self.connect()

    async def initialize_schema(self):
        """Initialize database schema for vector storage."""
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            # Create pgvector extension if it doesn't exist
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create documents table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    upload_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    page_count INTEGER,
                    word_count INTEGER,
                    checksum TEXT,
                    metadata JSONB DEFAULT '{}'
                );
            """
            )

            # Create document chunks table with vector column
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    embedding vector({settings.vector_dimension}),
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """
            )

            # Create indexes for better performance
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
                ON document_chunks(document_id);
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """
            )

            logger.info("Database schema initialized successfully")

    async def store_document(self, document_id: str, metadata: Dict[str, Any]) -> str:
        """Store document metadata.

        Args:
            document_id: Unique document identifier
            metadata: Document metadata

        Returns:
            Document ID
        """
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (id, filename, content_type, size, upload_time,
                                     page_count, word_count, checksum, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    content_type = EXCLUDED.content_type,
                    size = EXCLUDED.size,
                    upload_time = EXCLUDED.upload_time,
                    page_count = EXCLUDED.page_count,
                    word_count = EXCLUDED.word_count,
                    checksum = EXCLUDED.checksum,
                    metadata = EXCLUDED.metadata
            """,
                (
                    document_id,
                    metadata.get("filename"),
                    metadata.get("content_type"),
                    metadata.get("size"),
                    metadata.get("upload_time"),
                    metadata.get("page_count"),
                    metadata.get("word_count"),
                    metadata.get("checksum"),
                    metadata.get("metadata", {}),
                ),
            )

        logger.info(f"Stored document metadata: {document_id}")
        return document_id

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Store document chunks with embeddings.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Number of chunks stored
        """
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            # Prepare chunk data
            chunk_data = []
            for chunk in chunks:
                embedding = chunk.get("embedding")
                if embedding and isinstance(embedding, list):
                    embedding_str = f"[{','.join(map(str, embedding))}]"
                else:
                    embedding_str = None

                chunk_data.append(
                    (
                        chunk["id"],
                        chunk["document_id"],
                        chunk["content"],
                        chunk["chunk_index"],
                        chunk["total_chunks"],
                        embedding_str,
                        chunk.get("metadata", {}),
                    )
                )

            # Bulk insert chunks
            await conn.executemany(
                """
                INSERT INTO document_chunks (id, document_id, content, chunk_index,
                                          total_chunks, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """,
                chunk_data,
            )

        logger.info(f"Stored {len(chunks)} document chunks")
        return len(chunks)

    async def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            filters: Optional search filters
            threshold: Optional similarity threshold

        Returns:
            List of similar documents with scores
        """
        await self._ensure_connection()

        # Convert query vector to pgvector format
        query_vec = f"[{','.join(map(str, query_vector))}]"

        # Build query
        base_query = """
            SELECT
                dc.id,
                dc.document_id,
                dc.content,
                dc.chunk_index,
                dc.total_chunks,
                dc.metadata,
                dc.created_at,
                d.filename,
                d.content_type,
                1 - (dc.embedding <=> $1) as similarity_score
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL
        """

        params = [query_vec]
        param_count = 1

        # Add filters
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                param_count += 1
                if key == "document_id":
                    filter_conditions.append(f"dc.document_id = ${param_count}")
                    params.append(value)
                elif key == "content_type":
                    filter_conditions.append(f"d.content_type = ${param_count}")
                    params.append(value)

            if filter_conditions:
                base_query += " AND " + " AND ".join(filter_conditions)

        # Add threshold filter
        if threshold is not None:
            param_count += 1
            base_query += f" AND (1 - (dc.embedding <=> $1)) >= ${param_count}"
            params.append(threshold)

        # Add ordering and limit
        base_query += " ORDER BY dc.embedding <=> $1 LIMIT $2"
        params.append(top_k)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(base_query, *params)

        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "chunk_index": row["chunk_index"],
                    "total_chunks": row["total_chunks"],
                    "metadata": dict(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat(),
                    "filename": row["filename"],
                    "content_type": row["content_type"],
                    "similarity_score": float(row["similarity_score"]),
                }
            )

        logger.info(f"Similarity search returned {len(results)} results")
        return results

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document metadata or None if not found
        """
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1", document_id
            )

        if row:
            return {
                "id": row["id"],
                "filename": row["filename"],
                "content_type": row["content_type"],
                "size": row["size"],
                "upload_time": row["upload_time"].isoformat(),
                "page_count": row["page_count"],
                "word_count": row["word_count"],
                "checksum": row["checksum"],
                "metadata": dict(row["metadata"]) if row["metadata"] else {},
            }

        return None

    async def list_documents(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all documents.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of document metadata
        """
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM documents
                ORDER BY upload_time DESC
                LIMIT $1 OFFSET $2
            """,
                limit,
                offset,
            )

        documents = []
        for row in rows:
            documents.append(
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "content_type": row["content_type"],
                    "size": row["size"],
                    "upload_time": row["upload_time"].isoformat(),
                    "page_count": row["page_count"],
                    "word_count": row["word_count"],
                    "checksum": row["checksum"],
                    "metadata": dict(row["metadata"]) if row["metadata"] else {},
                }
            )

        return documents

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks.

        Args:
            document_id: Document identifier

        Returns:
            True if document was deleted, False if not found
        """
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE id = $1", document_id
            )

        deleted = result.split()[-1] != "0"
        if deleted:
            logger.info(f"Deleted document: {document_id}")
        return deleted

    async def get_chunk_count(self, document_id: Optional[str] = None) -> int:
        """Get total number of chunks.

        Args:
            document_id: Optional document ID filter

        Returns:
            Number of chunks
        """
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            if document_id:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM document_chunks WHERE document_id = $1",
                    document_id,
                )
            else:
                result = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")

        return result or 0

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            await self._ensure_connection()
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
