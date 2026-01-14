"""PostgreSQL vector store operations with pgvector."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import asyncpg
import numpy as np

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
                self.connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60,
            )

            # Note: pgvector registration removed - using string format for vector storage

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
            metadata: Document metadata dictionary

        Returns:
            Document ID
        """
        logger.info(
            f"Storing document {document_id} with metadata keys: {list(metadata.keys())}"
        )
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            # Convert upload_time string to datetime if needed
            upload_time = metadata.get("upload_time")
            if isinstance(upload_time, str):
                from datetime import datetime

                upload_time = datetime.fromisoformat(upload_time.replace("Z", "+00:00"))

            import json

            params = (
                document_id,
                metadata.get("filename"),
                metadata.get("content_type"),
                metadata.get("size"),
                upload_time,
                metadata.get("page_count") or 0,  # Default to 0 if None
                metadata.get("word_count") or 0,  # Default to 0 if None
                metadata.get("checksum"),
                json.dumps(metadata),  # Convert entire metadata dict to JSON string
            )
            print(f"DEBUG: Executing query with {len(params)} parameters")
            print(f"DEBUG: Parameter types: {[type(p).__name__ for p in params]}")
            print(f"DEBUG: Parameter values: {params}")
            print(f"DEBUG: metadata keys: {list(metadata.keys())}")
            print(f"DEBUG: metadata.metadata: {metadata.get('metadata')}")

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
                *params,  # Unpack the tuple
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
                # Convert to PostgreSQL vector string format for storage
                if embedding is not None:
                    if hasattr(embedding, "tolist"):  # numpy array
                        embedding = embedding.tolist()
                    if isinstance(embedding, list):
                        # Convert to PostgreSQL vector string format: '[val1,val2,val3]'
                        embedding = f"[{','.join(str(float(x)) for x in embedding)}]"
                    elif not isinstance(embedding, str):
                        embedding = None
                logger.info(
                    f"Chunk embedding type: {type(embedding)}, sample: {str(embedding)[:100] if embedding is not None else None}"
                )

                chunk_data.append(
                    (
                        chunk["id"],
                        chunk["document_id"],
                        chunk["content"],
                        chunk["chunk_index"],
                        chunk["total_chunks"],
                        embedding,  # Pass as string for PostgreSQL vector type
                        json.dumps(chunk.get("metadata", {})),
                    )
                )

                chunk_data.append(
                    (
                        chunk["id"],
                        chunk["document_id"],
                        chunk["content"],
                        chunk["chunk_index"],
                        chunk["total_chunks"],
                        embedding,  # Pass as numpy array for pgvector
                        json.dumps(chunk.get("metadata", {})),
                    )
                )

            # Bulk insert chunks
            logger.info(f"Bulk inserting {len(chunk_data)} chunks")
            try:
                result = await conn.executemany(
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
                logger.info(f"executemany result: {result}")
            except Exception as e:
                logger.error(f"Error in executemany: {e}")
                raise

        result = len(chunks)
        logger.info(f"Stored {result} document chunks, returning {result}")
        return result

    async def similarity_search(
        self,
        query_vector: Union[List[float], np.ndarray],
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

        # Convert query vector to list, then to PostgreSQL vector string format
        try:
            # Try to convert to list (works for both lists and numpy arrays)
            query_vec_list = list(query_vector)
        except TypeError:
            # If that fails, assume it's already a list
            query_vec_list = query_vector

        # Format as PostgreSQL vector string: '[val1,val2,val3]'
        query_vec_str = f"[{','.join(str(float(x)) for x in query_vec_list)}]"
        params = [query_vec_str]
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

        # Add ordering and limit (order by similarity descending)
        param_count += 1
        base_query += f" ORDER BY (1 - (dc.embedding <=> $1)) DESC LIMIT ${param_count}"
        params.append(top_k)

        async with self._pool.acquire() as conn:
            logger.info(f"Executing similarity search query with {len(params)} params")
            logger.info(f"Query: {base_query}")
            logger.info(f"Params: {[type(p).__name__ for p in params]}")
            rows = await conn.fetch(base_query, *params)
            logger.info(f"Query returned {len(rows)} rows")

        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "chunk_index": row["chunk_index"],
                    "total_chunks": row["total_chunks"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
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
                "metadata": row["metadata"] if row["metadata"] else {},
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
                SELECT
                    d.*,
                    COUNT(dc.id) as chunks_count
                FROM documents d
                LEFT JOIN document_chunks dc ON d.id = dc.document_id
                GROUP BY d.id, d.filename, d.content_type, d.size, d.upload_time,
                         d.page_count, d.word_count, d.checksum, d.metadata
                ORDER BY d.upload_time DESC
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
                    "uploaded_at": (
                        row["upload_time"]
                        .isoformat()
                        .replace("+00:00", "Z")
                        .replace("+01:00", "Z")
                        if hasattr(row["upload_time"], "isoformat")
                        and row["upload_time"] is not None
                        else str(row["upload_time"])
                        .replace("+00:00", "Z")
                        .replace("+01:00", "Z")
                        if row["upload_time"] is not None
                        else "2024-01-01T00:00:00Z"
                    ),
                    "status": "completed",  # Documents in DB are successfully processed
                    "chunks_count": int(row["chunks_count"]),
                    "page_count": row["page_count"],
                    "word_count": row["word_count"],
                    "checksum": row["checksum"],
                    "metadata": row["metadata"] if row["metadata"] else {},
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
        """Check if the vector store is healthy."""
        try:
            await self._ensure_connection()
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False

    async def _clear_all_test_data(self):
        """Clear all data for testing purposes. USE WITH CAUTION."""
        await self._ensure_connection()
        async with self._pool.acquire() as conn:
            # Clear all test data - be very careful with this in production!
            chunks_deleted = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
            docs_deleted = await conn.fetchval("SELECT COUNT(*) FROM documents")
            await conn.execute("DELETE FROM document_chunks")
            await conn.execute("DELETE FROM documents")
            logger.info(
                f"Cleared test data: {docs_deleted} documents, {chunks_deleted} chunks"
            )
