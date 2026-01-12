#!/usr/bin/env python3
"""
RAG System Evaluation Script

This script evaluates the performance of the RAG system by running test queries
and measuring retrieval accuracy, answer quality, and response times.
"""

import asyncio
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

# Add the app directory to the path so we can import modules
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app.rag_agent import RAGAgent

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for RAG system performance."""

    def __init__(self, rag_agent: RAGAgent):
        self.rag_agent = rag_agent
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "response_times": [],
            "answer_lengths": [],
            "source_counts": [],
        }

    async def evaluate_query(
        self,
        query: str,
        expected_sources: Optional[List[str]] = None,
        expected_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single query and return metrics."""
        start_time = time.time()

        try:
            response = await self.rag_agent.query(query)
            response_time = time.time() - start_time

            # Basic metrics
            answer_length = len(response.answer or "")
            source_count = len(response.sources or [])

            # Keyword matching (if expected keywords provided)
            keyword_matches = 0
            if expected_keywords:
                answer_lower = (response.answer or "").lower()
                keyword_matches = sum(
                    1
                    for keyword in expected_keywords
                    if keyword.lower() in answer_lower
                )

            # Source accuracy (if expected sources provided)
            source_accuracy = 0.0
            if expected_sources and response.sources:
                found_sources = {source.document_id for source in response.sources}
                expected_set = set(expected_sources)
                if expected_set:
                    source_accuracy = len(found_sources & expected_set) / len(
                        expected_set
                    )

            result = {
                "query": query,
                "success": True,
                "response_time": response_time,
                "answer_length": answer_length,
                "source_count": source_count,
                "keyword_matches": keyword_matches,
                "source_accuracy": source_accuracy,
                "error": None,
            }

        except Exception as e:
            response_time = time.time() - start_time
            result = {
                "query": query,
                "success": False,
                "response_time": response_time,
                "answer_length": 0,
                "source_count": 0,
                "keyword_matches": 0,
                "source_accuracy": 0.0,
                "error": str(e),
            }

        # Update aggregate metrics
        self.metrics["total_queries"] += 1
        if result["success"]:
            self.metrics["successful_queries"] += 1
        self.metrics["response_times"].append(result["response_time"])
        self.metrics["answer_lengths"].append(result["answer_length"])
        self.metrics["source_counts"].append(result["source_count"])

        return result

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from all evaluations."""
        if not self.metrics["response_times"]:
            return {}

        return {
            "total_queries": self.metrics["total_queries"],
            "success_rate": self.metrics["successful_queries"]
            / self.metrics["total_queries"],
            "avg_response_time": statistics.mean(self.metrics["response_times"]),
            "median_response_time": statistics.median(self.metrics["response_times"]),
            "min_response_time": min(self.metrics["response_times"]),
            "max_response_time": max(self.metrics["response_times"]),
            "avg_answer_length": statistics.mean(self.metrics["answer_lengths"]),
            "avg_source_count": statistics.mean(self.metrics["source_counts"]),
        }


async def load_test_queries(test_file: Path) -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(test_file, "r") as f:
        data = json.load(f)

    queries = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "query" in item:
                queries.append(
                    {
                        "query": item["query"],
                        "expected_sources": item.get("expected_sources", []),
                        "expected_keywords": item.get("expected_keywords", []),
                    }
                )
    elif isinstance(data, dict) and "queries" in data:
        for item in data["queries"]:
            if isinstance(item, dict) and "query" in item:
                queries.append(
                    {
                        "query": item["query"],
                        "expected_sources": item.get("expected_sources", []),
                        "expected_keywords": item.get("expected_keywords", []),
                    }
                )

    return queries


async def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument(
        "--test-set",
        "-t",
        type=str,
        required=True,
        help="Path to JSON file containing test queries",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="evaluation_results.json",
        help="Output file for results (default: evaluation_results.json)",
    )

    args = parser.parse_args()

    test_file = Path(args.test_set)
    if not test_file.exists():
        logger.error(f"Test file does not exist: {test_file}")
        sys.exit(1)

    logger.info("Starting RAG system evaluation...")
    logger.info(f"Test file: {test_file}")

    try:
        # Initialize RAG agent
        rag_agent = RAGAgent()

        await rag_agent.initialize()

        # Clear caches to ensure fresh evaluation
        logger.info("Clearing caches for fresh evaluation...")
        await rag_agent.clear_cache()

        # Load test queries
        test_queries = await load_test_queries(test_file)
        if not test_queries:
            logger.error("No valid test queries found in file")
            sys.exit(1)

        logger.info(f"Loaded {len(test_queries)} test queries")

        # Initialize evaluator
        evaluator = RAGEvaluator(rag_agent)

        # Run evaluations
        results = []
        for i, test_query in enumerate(test_queries, 1):
            logger.info(
                f"Evaluating query {i}/{len(test_queries)}: {test_query['query'][:50]}..."
            )

            result = await evaluator.evaluate_query(
                query=test_query["query"],
                expected_sources=test_query.get("expected_sources"),
                expected_keywords=test_query.get("expected_keywords"),
            )
            results.append(result)

        # Generate summary
        summary = evaluator.get_summary_stats()

        # Prepare final results
        evaluation_results = {
            "timestamp": time.time(),
            "test_file": str(test_file),
            "summary": summary,
            "individual_results": results,
        }

        # Save results
        output_file = Path(args.output)
        with open(output_file, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        logger.info(f"Evaluation completed. Results saved to {output_file}")

        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Total Queries: {summary.get('total_queries', 0)}")
        print(".1f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".1f")
        print(".1f")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
