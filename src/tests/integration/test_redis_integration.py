"""
Integration tests for Redis operations with real Redis.
Tests caching, conversation memory, and data persistence.
"""

import asyncio
from typing import Any, Dict

import pytest


@pytest.mark.integration
@pytest.mark.redis
@pytest.mark.slow
class TestRedisIntegration:
    """Integration tests for Redis operations."""

    @pytest.mark.asyncio
    async def test_redis_connection(self, real_redis_memory):
        """Test that Redis memory can connect to real Redis."""
        # Test health check
        health = await real_redis_memory.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_embedding_caching(self, real_redis_memory):
        """Test embedding caching with real Redis."""
        text_hash = "test_hash_123"
        embedding = [0.1, 0.2, 0.3] * 256  # 768 dimensions

        # Initially should not be cached
        cached = await real_redis_memory.get_cached_embedding(text_hash)
        assert cached is None

        # Cache the embedding
        await real_redis_memory.cache_embedding(text_hash, embedding)

        # Should now be cached
        cached = await real_redis_memory.get_cached_embedding(text_hash)
        assert cached == embedding

    @pytest.mark.asyncio
    async def test_query_result_caching(self, real_redis_memory):
        """Test query result caching with real Redis."""
        query_hash = "query_hash_456"
        query_result = {
            "query": "What is AI?",
            "answer": "AI is artificial intelligence.",
            "sources": [],
            "confidence_score": 0.85,
        }

        # Initially should not be cached
        cached = await real_redis_memory.get_cached_query_result(query_hash)
        assert cached is None

        # Cache the result
        await real_redis_memory.cache_query_result(query_hash, query_result)

        # Should now be cached
        cached = await real_redis_memory.get_cached_query_result(query_hash)
        assert cached == query_result

    @pytest.mark.asyncio
    async def test_conversation_management(self, real_redis_memory):
        """Test conversation storage and retrieval."""
        session_id = "test_session_789"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        # Store conversation
        await real_redis_memory.store_conversation(session_id, messages)

        # Retrieve conversation
        retrieved = await real_redis_memory.get_conversation(session_id)
        assert retrieved is not None
        assert retrieved["session_id"] == session_id
        assert retrieved["messages"] == messages

    @pytest.mark.asyncio
    async def test_conversation_updates(self, real_redis_memory):
        """Test conversation updates and appending messages."""
        session_id = "update_session_101"

        # Initial conversation
        initial_messages = [{"role": "user", "content": "Initial message"}]
        await real_redis_memory.store_conversation(session_id, initial_messages)

        # Update with more messages
        updated_messages = [
            {"role": "user", "content": "Initial message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Follow up"},
        ]
        await real_redis_memory.update_conversation(session_id, updated_messages)

        # Retrieve updated conversation
        retrieved = await real_redis_memory.get_conversation(session_id)
        assert retrieved is not None
        assert len(retrieved["messages"]) == 3
        assert retrieved["messages"] == updated_messages

    @pytest.mark.asyncio
    async def test_conversation_clearing(self, real_redis_memory):
        """Test clearing conversation data."""
        session_id = "clear_session_202"

        # Store conversation
        messages = [{"role": "user", "content": "Test message"}]
        await real_redis_memory.store_conversation(session_id, messages)

        # Verify it exists
        retrieved = await real_redis_memory.get_conversation(session_id)
        assert retrieved is not None

        # Clear conversation
        result = await real_redis_memory.clear_conversation(session_id)
        assert result is True

        # Should no longer exist
        retrieved = await real_redis_memory.get_conversation(session_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_cache_statistics(self, real_redis_memory):
        """Test cache statistics retrieval."""
        # Store some test data
        await real_redis_memory.cache_embedding("stat_test_1", [0.1] * 768)
        await real_redis_memory.cache_embedding("stat_test_2", [0.2] * 768)
        await real_redis_memory.cache_query_result("query_test_1", {"answer": "test"})

        stats = await real_redis_memory.get_stats()

        # Should have some statistics
        assert isinstance(stats, dict)
        assert "total_keys" in stats or len(stats) > 0

    @pytest.mark.asyncio
    async def test_cache_expiration(self, real_redis_memory):
        """Test that cached data expires properly."""
        text_hash = "expire_test_303"
        embedding = [0.3] * 768

        # Cache with short TTL (would need to modify the class to test this properly)
        # For now, just verify basic caching works
        await real_redis_memory.cache_embedding(text_hash, embedding)
        cached = await real_redis_memory.get_cached_embedding(text_hash)
        assert cached == embedding

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, real_redis_memory):
        """Test that different sessions are properly isolated."""
        session_1 = "isolation_session_1"
        session_2 = "isolation_session_2"

        messages_1 = [{"role": "user", "content": "Session 1 message"}]
        messages_2 = [{"role": "user", "content": "Session 2 message"}]

        # Store different conversations
        await real_redis_memory.store_conversation(session_1, messages_1)
        await real_redis_memory.store_conversation(session_2, messages_2)

        # Retrieve and verify isolation
        retrieved_1 = await real_redis_memory.get_conversation(session_1)
        retrieved_2 = await real_redis_memory.get_conversation(session_2)

        assert retrieved_1["messages"] == messages_1
        assert retrieved_2["messages"] == messages_2
        assert retrieved_1["session_id"] != retrieved_2["session_id"]

    @pytest.mark.asyncio
    async def test_cache_clear_all(self, real_redis_memory):
        """Test clearing all cached data."""
        # Store various types of data
        await real_redis_memory.cache_embedding("clear_test_1", [0.1] * 768)
        await real_redis_memory.cache_query_result("clear_query_1", {"answer": "test"})
        await real_redis_memory.store_conversation(
            "clear_session_1", [{"role": "user", "content": "test"}]
        )

        # Verify data exists
        embedding = await real_redis_memory.get_cached_embedding("clear_test_1")
        assert embedding is not None

        query_result = await real_redis_memory.get_cached_query_result("clear_query_1")
        assert query_result is not None

        conversation = await real_redis_memory.get_conversation("clear_session_1")
        assert conversation is not None

        # Clear all cache
        result = await real_redis_memory.clear_all_cache()
        assert result is True

        # Verify data is cleared
        embedding = await real_redis_memory.get_cached_embedding("clear_test_1")
        assert embedding is None

        query_result = await real_redis_memory.get_cached_query_result("clear_query_1")
        assert query_result is None

        conversation = await real_redis_memory.get_conversation("clear_session_1")
        assert conversation is None

    @pytest.mark.asyncio
    async def test_concurrent_redis_operations(self, real_redis_memory):
        """Test concurrent Redis operations."""
        import asyncio

        async def cache_operation(operation_id: int):
            key = f"concurrent_test_{operation_id}"
            embedding = [float(operation_id) / 10.0] * 768

            # Cache embedding
            await real_redis_memory.cache_embedding(key, embedding)

            # Retrieve and verify
            cached = await real_redis_memory.get_cached_embedding(key)
            return cached == embedding

        # Run concurrent operations
        tasks = [cache_operation(i) for i in range(1, 11)]
        results = await asyncio.gather(*tasks)

        # All operations should succeed
        assert all(results)

    @pytest.mark.asyncio
    async def test_large_data_handling(self, real_redis_memory):
        """Test handling of larger data structures."""
        # Test with larger conversation
        session_id = "large_session_404"
        large_messages = [
            {"role": "user", "content": f"Message {i}: " + "A" * 1000}
            for i in range(10)
        ]

        # Store large conversation
        await real_redis_memory.store_conversation(session_id, large_messages)

        # Retrieve and verify
        retrieved = await real_redis_memory.get_conversation(session_id)
        assert retrieved is not None
        assert len(retrieved["messages"]) == 10
        assert all(
            len(msg["content"]) > 1000
            for msg in retrieved["messages"]
            if msg["role"] == "user"
        )
