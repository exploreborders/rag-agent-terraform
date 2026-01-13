#!/usr/bin/env python3

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, "/app/src")

from app.config import settings
from app.vector_store import VectorStore
from app.ollama_client import OllamaClient
from app.models import OllamaEmbedRequest


async def test_similarity_search():
    """Test the similarity search with a known query."""

    # Initialize clients
    ollama_client = OllamaClient(base_url=settings.ollama_base_url)

    vector_store = VectorStore()
    await vector_store.connect()

    try:
        # Generate embedding for "machine learning"
        query_text = "What is machine learning?"
        print(f"Generating embedding for: '{query_text}'")

        request = OllamaEmbedRequest(
            model=settings.ollama_embed_model, prompt=query_text
        )
        response = await ollama_client.embed(request)
        query_vector = response.embedding

        print(f"Query vector dimensions: {len(query_vector)}")
        print(f"First 5 values: {query_vector[:5]}")

        # Perform similarity search
        print("\nPerforming similarity search...")
        results = await vector_store.similarity_search(
            query_vector=query_vector,
            top_k=5,
            threshold=0.0,  # Lower threshold to see all results
        )

        print(f"Found {len(results)} results")
        if len(results) == 0:
            print("No results found. Let me check if chunks exist...")
            chunk_count = await vector_store.get_chunk_count()
            print(f"Total chunks in database: {chunk_count}")

            # Try with no threshold at all
            print("Trying similarity search with no threshold and top_k=1...")
            results_no_threshold = await vector_store.similarity_search(
                query_vector=query_vector, top_k=1, threshold=None
            )
            print(f"Results with no threshold: {len(results_no_threshold)}")

        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            print(f"  Document: {result['filename']}")
            print(f"  Similarity: {result['similarity_score']:.4f}")
            print(f"  Content preview: {result['content'][:100]}...")

        # Also test with the default threshold
        print("\nTesting with threshold 0.7...")
        results_filtered = await vector_store.similarity_search(
            query_vector=query_vector, top_k=5, threshold=0.7
        )
        print(f"Found {len(results_filtered)} results above threshold 0.7")

    finally:
        await vector_store.disconnect()


if __name__ == "__main__":
    asyncio.run(test_similarity_search())
