"""
Simple integration test to validate the infrastructure works.
"""

import pytest


@pytest.mark.integration
@pytest.mark.database
async def test_integration_infrastructure(real_vector_store, real_redis_memory):
    """Test that the integration infrastructure is properly set up."""
    # Test database connection
    health = await real_vector_store.health_check()
    assert health is True

    # Test Redis connection
    health = await real_redis_memory.health_check()
    assert health is True

    # Test basic database operations
    stats = await real_vector_store.get_chunk_count()
    assert isinstance(stats, int)

    # Test basic Redis operations
    stats = await real_redis_memory.get_stats()
    assert isinstance(stats, dict)
