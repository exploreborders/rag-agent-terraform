"""
Basic integration tests for Redis operations.
Tests core Redis connectivity and basic operations.
"""

import pytest


@pytest.mark.integration
@pytest.mark.redis
class TestRedisIntegration:
    """Basic integration tests for Redis operations."""

    @pytest.mark.asyncio
    async def test_redis_connection(self, real_redis_memory):
        """Test Redis memory initialization and connection."""
        assert real_redis_memory is not None
        assert real_redis_memory._redis is not None

    @pytest.mark.asyncio
    async def test_redis_health_check(self, real_redis_memory):
        """Test Redis memory health check."""
        health = await real_redis_memory.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_basic_redis_operations(self, real_redis_memory):
        """Test basic Redis set/get operations."""
        # Test setting and getting data
        test_key = "integration_test_key"
        test_data = {"message": "Redis integration test", "status": "working"}

        # Store data
        result = await real_redis_memory.set(test_key, test_data)
        assert result is True

        # Retrieve data
        retrieved = await real_redis_memory.get(test_key)
        assert retrieved == test_data
