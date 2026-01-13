"""Redis-backed agent memory and caching."""

import logging
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class RedisMemoryError(Exception):
    """Base exception for Redis memory errors."""

    pass


class RedisConnectionError(RedisMemoryError):
    """Exception raised when Redis connection fails."""

    pass


class AgentMemory:
    """Redis-backed agent memory for conversation context and caching."""

    def __init__(self, redis_url: Optional[str] = None, ttl: int = None):
        """Initialize Redis memory.

        Args:
            redis_url: Redis connection URL
            ttl: Default time-to-live for cached items in seconds
        """
        self.redis_url = redis_url or settings.redis_url
        self.ttl = ttl or settings.cache_ttl
        self._redis: Optional[redis.Redis] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Connect to Redis."""
        try:
            self._redis = redis.Redis.from_url(
                self.redis_url,
                decode_responses=False,  # Keep bytes for pickle
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            await self._redis.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise RedisConnectionError(f"Redis connection failed: {e}")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Disconnected from Redis")

    async def _ensure_connection(self):
        """Ensure Redis connection is active."""
        if not self._redis:
            await self.connect()

    async def _serialize(self, data: Any) -> bytes:
        """Serialize data for Redis storage."""
        return pickle.dumps(data)

    async def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from Redis storage."""
        if data is None:
            return None
        return pickle.loads(data)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a key-value pair with optional TTL.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        await self._ensure_connection()
        if self._redis is None:
            return False

        try:
            serialized_value = await self._serialize(value)
            ttl_value = ttl or self.ttl

            await self._redis.setex(key, ttl_value, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        await self._ensure_connection()
        if self._redis is None:
            return None

        try:
            serialized_value = await self._redis.get(key)
            if serialized_value is None:
                return None
            return await self._deserialize(serialized_value)
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        await self._ensure_connection()
        if self._redis is None:
            return False

        try:
            result = await self._redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        await self._ensure_connection()
        if self._redis is None:
            return False

        try:
            return bool(await self._redis.exists(key))
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key.

        Args:
            key: Cache key
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        await self._ensure_connection()
        if self._redis is None:
            return False

        try:
            return bool(await self._redis.expire(key, ttl))
        except Exception as e:
            logger.error(f"Failed to set expiration for key {key}: {e}")
            return False

    # Conversation Memory Methods
    async def store_conversation(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store conversation history for a session.

        Args:
            session_id: Unique session identifier
            messages: List of conversation messages
            context: Additional conversation context

        Returns:
            True if successful
        """
        conversation_data = {
            "session_id": session_id,
            "messages": messages,
            "context": context or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        key = f"conversation:{session_id}"
        return await self.set(
            key, conversation_data, ttl=self.ttl * 24
        )  # Keep conversations longer

    async def get_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation history for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Conversation data or None if not found
        """
        key = f"conversation:{session_id}"
        return await self.get(key)

    async def update_conversation(
        self,
        session_id: str,
        new_messages: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update conversation with new messages.

        Args:
            session_id: Unique session identifier
            new_messages: New messages to add
            context: Updated context

        Returns:
            True if successful
        """
        existing = await self.get_conversation(session_id)
        if existing:
            existing["messages"].extend(new_messages)
            if context:
                existing["context"].update(context)
            existing["updated_at"] = datetime.utcnow().isoformat()
            conversation_data = existing
        else:
            conversation_data = {
                "session_id": session_id,
                "messages": new_messages,
                "context": context or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }

        key = f"conversation:{session_id}"
        return await self.set(key, conversation_data, ttl=self.ttl * 24)

    async def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            True if successful
        """
        key = f"conversation:{session_id}"
        return await self.delete(key)

    # Query Result Caching
    async def cache_query_result(
        self,
        query_hash: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache query results for performance.

        Args:
            query_hash: Hash of the query for caching key
            result: Query result to cache
            ttl: Cache TTL (uses default if not specified)

        Returns:
            True if successful
        """
        key = f"query:{query_hash}"
        cache_ttl = ttl or self.ttl
        return await self.set(key, result, ttl=cache_ttl)

    async def get_cached_query_result(
        self, query_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached query result.

        Args:
            query_hash: Hash of the query

        Returns:
            Cached result or None if not found
        """
        key = f"query:{query_hash}"
        return await self.get(key)

    # Document Processing Cache
    async def cache_document_chunks(
        self, document_id: str, chunks: List[Dict[str, Any]]
    ) -> bool:
        """Cache processed document chunks.

        Args:
            document_id: Document identifier
            chunks: Processed document chunks

        Returns:
            True if successful
        """
        key = f"document_chunks:{document_id}"
        return await self.set(
            key, chunks, ttl=self.ttl * 24
        )  # Keep document chunks longer

    async def get_cached_document_chunks(
        self, document_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached document chunks.

        Args:
            document_id: Document identifier

        Returns:
            Cached chunks or None if not found
        """
        key = f"document_chunks:{document_id}"
        return await self.get(key)

    # Embedding Cache
    async def cache_embedding(self, text_hash: str, embedding: List[float]) -> bool:
        """Cache text embeddings.

        Args:
            text_hash: Hash of the text
            embedding: Text embedding vector

        Returns:
            True if successful
        """
        key = f"embedding:{text_hash}"
        return await self.set(
            key, embedding, ttl=self.ttl * 24
        )  # Keep embeddings longer

    async def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embedding.

        Args:
            text_hash: Hash of the text

        Returns:
            Cached embedding or None if not found
        """
        key = f"embedding:{text_hash}"
        return await self.get(key)

    # Health Check
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            await self._ensure_connection()
            if self._redis is None:
                return False
            await self._redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    # Utility Methods
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis memory statistics."""
        await self._ensure_connection()
        if self._redis is None:
            return {}

        try:
            info = await self._redis.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "total_connections_received": info.get("total_connections_received", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}

    async def clear_all_cache(self) -> bool:
        """Clear all cached data (use with caution)."""
        await self._ensure_connection()
        if self._redis is None:
            return False

        try:
            await self._redis.flushdb()
            logger.warning("Cleared all Redis cache data")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
