#!/usr/bin/env python3
"""Test the routing function directly."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from app.multi_agent_graph import route_after_query_processor
from app.multi_agent_state import create_initial_state


def test_routing():
    """Test the routing function with different agent task configurations."""

    # Test 1: Single workflow (legacy)
    print("🧪 Test 1: Single workflow (legacy)")
    state = create_initial_state("What is machine learning?")
    state["agent_tasks"] = {
        "retrieval": {"query": "test", "intent": "baseline_retrieval"}
    }
    result = route_after_query_processor(state)
    print(f"  Routing result: {result}")
    assert result == "retrieval_agent"

    # Test 2: Time query (special case)
    print("\n🧪 Test 2: Time query (special case)")
    state = create_initial_state("What time is it?")
    state["agent_tasks"] = {"mcp_research": {"query": "test", "intent": "time_lookup"}}
    state["intent_classification"] = {"primary_intent": "time_query"}
    result = route_after_query_processor(state)
    print(f"  Routing result: {result}")
    assert result == "mcp_search_agent"

    # Test 3: Multi-workflow (parallel)
    print("\n🧪 Test 3: Multi-workflow (parallel)")
    state = create_initial_state("Research AI and what time it is")
    state["agent_tasks"] = {
        "academic_workflow_1": {"intent": "academic_research", "query": "Research AI"},
        "time_query_workflow_2": {"intent": "time_lookup", "query": "what time it is"},
    }
    result = route_after_query_processor(state)
    print(f"  Routing result: {result}")
    assert result == "multi_workflow_coordinator"

    print("\n✅ All routing tests passed!")


if __name__ == "__main__":
    test_routing()
