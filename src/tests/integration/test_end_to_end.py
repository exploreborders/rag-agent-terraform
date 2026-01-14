"""
Basic end-to-end integration tests for RAG system workflows.
Tests core functionality without complex scenarios.
"""

import pytest


@pytest.mark.integration
class TestEndToEndIntegration:
    """Basic end-to-end tests for RAG workflows."""

    @pytest.mark.asyncio
    async def test_rag_agent_initialization(self, real_rag_agent):
        """Test that the RAG agent can be initialized."""
        # Test that the agent can be initialized
        assert real_rag_agent is not None
        assert real_rag_agent.vector_store is not None
        assert real_rag_agent.memory is not None
        assert real_rag_agent.ollama_client is not None
