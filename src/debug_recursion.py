#!/usr/bin/env python3
"""Test script to debug the recursion issue in the multi-agent system."""

import asyncio
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Imports
from app.config import settings
from app.mcp_client import mcp_client
from app.multi_agent_graph import create_docker_multi_agent_graph, query_processor_agent
from app.multi_agent_state import create_initial_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_query_processor():
    """Test the query processor agent in isolation."""
    print("🧪 Testing Query Processor Agent...")

    # Create initial state
    state = create_initial_state("What is machine learning?")

    try:
        # Test query processor
        result = await query_processor_agent(state)
        print("✅ Query Processor Agent succeeded:")
        print(f"   Sanitized query: {result.get('sanitized_query', 'N/A')}")
        print(
            f"   Intent: {result.get('intent_classification', {}).get('primary_intent', 'N/A')}"
        )
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        return True
    except Exception as e:
        print(f"❌ Query Processor Agent failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_mcp_client():
    """Test MCP client connection."""
    print("🧪 Testing MCP Client...")

    try:
        await mcp_client.connect()
        print("✅ MCP Client connected successfully")
        health = await mcp_client.health_check()
        print(f"   Status: {health.get('status')}")
        print(f"   Available tools: {health.get('available_tools', 0)}")
        return True
    except Exception as e:
        print(f"❌ MCP Client failed: {e}")
        return False


async def test_graph_creation():
    """Test graph creation without execution."""
    print("🧪 Testing Graph Creation...")

    try:
        graph = create_docker_multi_agent_graph()
        print("✅ Graph created successfully")
        print(f"   Nodes: {len(graph.nodes)}")
        print(f"   Edges: {len(graph.edges)}")
        return True
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_full_graph():
    """Test full graph execution with minimal query."""
    print("🧪 Testing Full Graph Execution...")

    try:
        # Create graph
        graph = create_docker_multi_agent_graph()
        compiled_graph = graph.compile()

        # Create initial state
        state = create_initial_state("Test query")

        # Execute graph
        result = await compiled_graph.ainvoke(state)

        print("✅ Full graph execution succeeded")
        print(f"   Final response: {result.get('final_response', 'N/A')[:100]}...")
        print(f"   Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"   Agent tasks: {list(result.get('agent_tasks', {}).keys())}")
        return True

    except Exception as e:
        print(f"❌ Full graph execution failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🔍 Debugging Multi-Agent Recursion Issue")
    print("=" * 50)

    # Test components individually
    tests = [
        ("MCP Client", test_mcp_client),
        ("Query Processor", test_query_processor),
        ("Graph Creation", test_graph_creation),
        ("Full Graph", test_full_graph),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        results[test_name] = await test_func()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")

    # Analyze results
    all_passed = all(results.values())
    if all_passed:
        print(
            "\n🎉 All tests passed! The recursion issue might be in the web request handling."
        )
    else:
        failed_tests = [name for name, success in results.items() if not success]
        print(f"\n⚠️  Tests failed: {', '.join(failed_tests)}")
        print("The recursion issue is likely in one of these components.")


if __name__ == "__main__":
    asyncio.run(main())
