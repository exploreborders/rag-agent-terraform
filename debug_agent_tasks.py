#!/usr/bin/env python3
"""Debug the actual agent_tasks created by query processor."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from app.multi_agent_graph import query_processor_agent
from app.multi_agent_state import create_initial_state


async def debug_agent_tasks():
    """Debug what agent_tasks are created for multi-intent queries."""

    query = "Research climate change impacts and current time in Antarctica"
    print(f"🔍 Debugging agent_tasks for: '{query}'")
    print("=" * 80)

    # Create state
    state = create_initial_state(query)

    # Process through query processor
    result = await query_processor_agent(state)

    # Extract agent_tasks
    agent_tasks = result.get("agent_tasks", {})
    print(f"📋 Agent Tasks Created: {len(agent_tasks)}")
    print()

    for task_name, task_config in agent_tasks.items():
        print(f"🔧 {task_name}:")
        for key, value in task_config.items():
            if isinstance(value, list) and len(value) > 3:
                print(f"    {key}: [{', '.join(str(x) for x in value[:3])}, ...]")
            else:
                print(f"    {key}: {value}")
        print()

    # Check routing logic
    from app.multi_agent_graph import route_after_query_processor

    # Create mock state with these agent_tasks
    mock_state = create_initial_state(query)
    mock_state["agent_tasks"] = agent_tasks
    mock_state["intent_classification"] = result.get("intent_classification", {})

    routing_result = route_after_query_processor(mock_state)
    print(f"🎯 Routing Decision: {routing_result}")

    # Check for workflow tasks
    workflow_tasks = [
        task
        for task in agent_tasks.keys()
        if task.endswith("_workflow_1")
        or task.endswith("_workflow_2")
        or task.endswith("_workflow_3")
    ]
    print(f"🔀 Workflow Tasks Detected: {len(workflow_tasks)} - {workflow_tasks}")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(debug_agent_tasks())
