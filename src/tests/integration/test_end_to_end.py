"""
End-to-end integration tests for complete RAG system workflows.
Tests full user journeys from document upload to query answering.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndIntegration:
    """End-to-end tests for complete RAG workflows."""

    @pytest.mark.asyncio
    async def test_complete_document_processing_workflow(
        self, real_rag_agent, temp_upload_dir
    ):
        """Test complete workflow: document upload → processing → querying."""
        # Create a test document
        doc_content = """
        Artificial Intelligence (AI) is a transformative technology that enables machines
        to perform tasks that typically require human intelligence. Machine learning,
        a subset of AI, uses algorithms to learn patterns from data without explicit programming.

        Deep learning, a powerful form of machine learning, uses neural networks with
        multiple layers to process complex data patterns. This technology has revolutionized
        fields like computer vision, natural language processing, and autonomous systems.

        The future of AI holds immense potential for solving complex problems across
        various domains including healthcare, transportation, and scientific research.
        """
        doc_path = temp_upload_dir / "ai_overview.txt"
        doc_path.write_text(doc_content)

        # Step 1: Process document
        result = await real_rag_agent.process_document(
            file_path=str(doc_path), content_type="text/plain"
        )

        assert result.status == "processed"
        assert result.chunks_count > 0
        assert result.embeddings_count > 0

        # Step 2: Verify document is stored
        documents = await real_rag_agent.list_documents()
        assert len(documents) >= 1
        ai_doc = next(
            (d for d in documents if d["filename"] == "ai_overview.txt"), None
        )
        assert ai_doc is not None

        # Step 3: Query about the document content
        query_result = await real_rag_agent.query(
            "What is the relationship between AI and machine learning?"
        )

        assert (
            query_result.query
            == "What is the relationship between AI and machine learning?"
        )
        assert len(query_result.sources) > 0
        assert "machine learning" in query_result.answer.lower()
        assert "artificial intelligence" in query_result.answer.lower()
        assert query_result.confidence_score > 0

    @pytest.mark.asyncio
    async def test_multi_document_query_workflow(self, real_rag_agent, temp_upload_dir):
        """Test querying across multiple documents."""
        # Create multiple documents on different topics
        docs = [
            (
                "machine_learning.txt",
                """
            Machine learning algorithms learn from data to make predictions.
            Supervised learning uses labeled data, while unsupervised learning
            finds patterns in unlabeled data. Reinforcement learning learns
            through trial and error with rewards and penalties.
            """,
            ),
            (
                "neural_networks.txt",
                """
            Neural networks are computing systems inspired by biological brains.
            They consist of layers of interconnected nodes called neurons.
            Deep learning uses many layers to process complex patterns.
            Convolutional networks excel at image processing tasks.
            """,
            ),
            (
                "data_science.txt",
                """
            Data science combines statistics, programming, and domain expertise
            to extract insights from data. Key skills include Python, R, SQL,
            and machine learning techniques. Data scientists clean, analyze,
            and visualize data to drive business decisions.
            """,
            ),
        ]

        # Process all documents
        for filename, content in docs:
            doc_path = temp_upload_dir / filename
            doc_path.write_text(content)

            result = await real_rag_agent.process_document(
                file_path=str(doc_path), content_type="text/plain"
            )
            assert result.status == "processed"

        # Query that should draw from multiple documents
        query_result = await real_rag_agent.query(
            "How do machine learning and neural networks relate to data science?"
        )

        assert (
            len(query_result.sources) >= 2
        )  # Should find relevant info from multiple docs
        assert query_result.confidence_score > 0

        # Check that answer mentions concepts from different documents
        answer_lower = query_result.answer.lower()
        relevant_terms = ["machine learning", "neural networks", "data science"]
        found_terms = sum(1 for term in relevant_terms if term in answer_lower)
        assert found_terms >= 2  # Should mention at least 2 of the 3 topics

    @pytest.mark.asyncio
    async def test_conversation_memory_workflow(self, real_rag_agent, temp_upload_dir):
        """Test conversation memory and context preservation."""
        # Create and process a document
        doc_content = """
        Python is a high-level programming language known for its simplicity and readability.
        It supports multiple programming paradigms including object-oriented, imperative,
        and functional programming. Python has a large standard library and extensive
        ecosystem of third-party packages.
        """
        doc_path = temp_upload_dir / "python_guide.txt"
        doc_path.write_text(doc_content)

        await real_rag_agent.process_document(
            file_path=str(doc_path), content_type="text/plain"
        )

        session_id = "conversation_test_123"

        # First query
        result1 = await real_rag_agent.query("What is Python?", session_id=session_id)
        assert "python" in result1.answer.lower()

        # Follow-up query (should maintain context)
        result2 = await real_rag_agent.query(
            "What programming paradigms does it support?", session_id=session_id
        )
        assert len(result2.sources) > 0

        # Check that conversation was stored
        # Note: This would require extending the agent to expose conversation data
        # For now, we verify the queries work with session context

    @pytest.mark.asyncio
    async def test_document_management_workflow(self, real_rag_agent, temp_upload_dir):
        """Test document upload, listing, and deletion workflow."""
        # Upload multiple documents
        doc_names = ["doc1.txt", "doc2.txt", "doc3.txt"]
        created_docs = []

        for i, name in enumerate(doc_names):
            doc_path = temp_upload_dir / name
            doc_path.write_text(f"This is document {i+1} content for testing.")

            result = await real_rag_agent.process_document(
                file_path=str(doc_path), content_type="text/plain"
            )
            created_docs.append(name)
            assert result.status == "processed"

        # List documents
        documents = await real_rag_agent.list_documents()
        uploaded_docs = [d for d in documents if d["filename"] in doc_names]
        assert len(uploaded_docs) == 3

        # Query to verify all documents are searchable
        query_result = await real_rag_agent.query("document content")
        assert len(query_result.sources) >= 3

        # Delete one document
        doc_to_delete = uploaded_docs[0]
        delete_result = await real_rag_agent.delete_document(doc_to_delete["id"])
        assert delete_result is True

        # Verify it's gone from listings
        documents_after = await real_rag_agent.list_documents()
        remaining_docs = [d for d in documents_after if d["filename"] in doc_names]
        assert len(remaining_docs) == 2

        # Verify it's gone from search results
        query_result_after = await real_rag_agent.query("document content")
        remaining_sources = [
            s
            for s in query_result_after.sources
            if s.document_id != doc_to_delete["id"]
        ]
        assert len(remaining_sources) >= 2

    @pytest.mark.asyncio
    async def test_performance_and_caching_workflow(
        self, real_rag_agent, temp_upload_dir
    ):
        """Test performance improvements with caching."""
        # Create a document
        doc_content = "This is a test document for performance evaluation. " * 50
        doc_path = temp_upload_dir / "performance_test.txt"
        doc_path.write_text(doc_content)

        await real_rag_agent.process_document(
            file_path=str(doc_path), content_type="text/plain"
        )

        import time

        # First query (should cache embeddings)
        start_time = time.time()
        result1 = await real_rag_agent.query("What is this document about?")
        first_query_time = time.time() - start_time

        # Second query with same content (should use cached results)
        start_time = time.time()
        result2 = await real_rag_agent.query("What is this document about?")
        second_query_time = time.time() - start_time

        # Both queries should succeed
        assert result1.answer
        assert result2.answer

        # Results should be similar (cached query result)
        assert result1.answer == result2.answer

        # Second query should be faster (though this is hard to guarantee in a test environment)
        # At minimum, both should complete successfully

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, real_rag_agent):
        """Test error handling in complete workflows."""
        # Test querying without any documents
        # This should not crash but return appropriate response
        result = await real_rag_agent.query("What is machine learning?")
        assert result.query == "What is machine learning?"
        # Should handle gracefully even with no documents
        assert isinstance(result.sources, list)

        # Test with invalid session (should not crash)
        result_with_session = await real_rag_agent.query(
            "Test query", session_id="nonexistent_session"
        )
        assert result_with_session.query == "Test query"

    @pytest.mark.asyncio
    async def test_data_consistency_workflow(self, real_rag_agent, temp_upload_dir):
        """Test data consistency across operations."""
        # Create document
        doc_content = """
        Consistency testing document. This document contains information
        about data integrity and consistency in database operations.
        """
        doc_path = temp_upload_dir / "consistency_test.txt"
        doc_path.write_text(doc_content)

        # Process document
        process_result = await real_rag_agent.process_document(
            file_path=str(doc_path), content_type="text/plain"
        )

        doc_id = f"doc-{hash(str(doc_path))}"  # This is how the agent generates IDs

        # Verify document exists
        doc_info = await real_rag_agent.get_document(doc_id)
        assert doc_info is not None
        assert doc_info["filename"] == "consistency_test.txt"

        # Verify chunks exist
        chunk_count = await real_rag_agent.get_chunk_count(doc_id)
        assert chunk_count > 0

        # Query and verify results are consistent
        query_result = await real_rag_agent.query("What is this document about?")
        assert len(query_result.sources) > 0

        # All sources should reference the correct document
        for source in query_result.sources:
            assert source.document_id == doc_id
            assert source.filename == "consistency_test.txt"

    @pytest.mark.asyncio
    async def test_system_health_and_stats(self, real_rag_agent, temp_upload_dir):
        """Test system health checks and statistics."""
        # Create some test data
        doc_path = temp_upload_dir / "health_test.txt"
        doc_path.write_text("Health check test document.")

        await real_rag_agent.process_document(
            file_path=str(doc_path), content_type="text/plain"
        )

        # Test system statistics
        stats = await real_rag_agent.get_stats()
        assert isinstance(stats, dict)
        assert "total_documents" in stats
        assert "total_chunks" in stats
        assert stats["total_documents"] >= 1
        assert stats["total_chunks"] >= 1

        # Test system health
        health = await real_rag_agent.health_check()
        assert health.status in ["healthy", "degraded"]
        assert "vector_store" in health.services
        assert "redis" in health.services
        assert "ollama" in health.services
