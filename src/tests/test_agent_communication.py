"""Tests for agent communication Redis message queue."""

"""Tests for agent communication Redis message queue."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_communication import (
    AgentCommunicationError,
    DockerAgentMessageQueue,
)


class TestDockerAgentMessageQueue:
    """Test suite for Docker Agent Message Queue."""

    @pytest.fixture
    def message_queue(self):
        """Create message queue instance."""
        return DockerAgentMessageQueue()

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock()
        mock.close = AsyncMock()
        mock.publish = AsyncMock()
        mock.pubsub = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_initialization(self, message_queue):
        """Test message queue initialization."""
        assert message_queue.redis_url == "redis://localhost:6379"
        assert message_queue.redis is None
        assert len(message_queue.agent_channels) == 6
        assert "query_processor" in message_queue.agent_channels
        assert "coordinator" in message_queue.agent_channels

    @pytest.mark.asyncio
    async def test_connect_success(self, message_queue, mock_redis):
        """Test successful Redis connection."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            await message_queue.connect()

            assert message_queue.redis == mock_redis
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure(self, message_queue):
        """Test Redis connection failure."""
        with patch(
            "redis.asyncio.from_url", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(
                AgentCommunicationError, match="Redis connection failed"
            ):
                await message_queue.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, message_queue, mock_redis):
        """Test Redis disconnection."""
        message_queue.redis = mock_redis
        await message_queue.disconnect()

        mock_redis.close.assert_called_once()
        assert message_queue.redis is None

    @pytest.mark.asyncio
    async def test_send_task_to_agent_success(self, message_queue, mock_redis):
        """Test successful task sending to agent."""
        message_queue.redis = mock_redis

        task_id = await message_queue.send_task_to_agent(
            "query_processor", {"query": "test query"}
        )

        assert task_id.startswith("task_")
        mock_redis.publish.assert_called_once()

        # Verify message structure
        call_args = mock_redis.publish.call_args
        channel, message_str = call_args[0]

        assert channel == "rag:agents:query_processor"
        message = json.loads(message_str)

        assert message["task_id"] == task_id
        assert message["agent"] == "query_processor"
        assert message["priority"] == "normal"
        assert message["task"] == {"query": "test query"}
        assert message["sender"] == "orchestrator"
        assert "timestamp" in message

    @pytest.mark.asyncio
    async def test_send_task_to_agent_unknown_agent(self, message_queue, mock_redis):
        """Test sending task to unknown agent."""
        message_queue.redis = mock_redis

        with pytest.raises(
            AgentCommunicationError, match="Unknown agent: unknown_agent"
        ):
            await message_queue.send_task_to_agent("unknown_agent", {})

    @pytest.mark.asyncio
    async def test_send_task_to_agent_publish_failure(self, message_queue, mock_redis):
        """Test task sending failure."""
        message_queue.redis = mock_redis
        mock_redis.publish.side_effect = Exception("Publish failed")

        with pytest.raises(AgentCommunicationError, match="Task sending failed"):
            await message_queue.send_task_to_agent("query_processor", {})

    @pytest.mark.asyncio
    async def test_send_broadcast_all_agents(self, message_queue, mock_redis):
        """Test broadcasting to all agents."""
        message_queue.redis = mock_redis

        await message_queue.send_broadcast({"message": "test broadcast"})

        assert mock_redis.publish.call_count == 6  # All agents

    @pytest.mark.asyncio
    async def test_send_broadcast_specific_agents(self, message_queue, mock_redis):
        """Test broadcasting to specific agents."""
        message_queue.redis = mock_redis

        await message_queue.send_broadcast(
            {"message": "test broadcast"}, ["query_processor", "retrieval_agent"]
        )

        assert mock_redis.publish.call_count == 2

    @pytest.mark.asyncio
    async def test_listen_for_results_success(self, message_queue, mock_redis):
        """Test successful result listening."""
        message_queue.redis = mock_redis

        # Mock pubsub as AsyncMock
        mock_pubsub = AsyncMock()
        mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

        # Mock subscribe and unsubscribe
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()

        # Create an async iterator for messages
        async def mock_listen():
            messages = [
                {
                    "type": "message",
                    "data": json.dumps(
                        {"status": "completed", "result": "test result"}
                    ),
                }
            ]
            for message in messages:
                yield message

        mock_pubsub.listen = mock_listen

        callback_called = False
        callback_result = None

        async def test_callback(agent_name, result):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result

        await message_queue.listen_for_results("query_processor", test_callback)

        assert callback_called
        assert callback_result["status"] == "completed"
        assert callback_result["result"] == "test result"

    @pytest.mark.asyncio
    async def test_listen_for_results_unknown_agent(self, message_queue, mock_redis):
        """Test listening for unknown agent results."""
        message_queue.redis = mock_redis

        with pytest.raises(
            AgentCommunicationError, match="Unknown agent: unknown_agent"
        ):
            await message_queue.listen_for_results("unknown_agent", lambda x, y: None)

    @pytest.mark.asyncio
    async def test_listen_for_results_invalid_json(self, message_queue, mock_redis):
        """Test handling of invalid JSON in result messages."""
        message_queue.redis = mock_redis

        mock_pubsub = AsyncMock()
        mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

        messages = [{"type": "message", "data": "invalid json"}]
        mock_pubsub.listen = AsyncMock(return_value=messages)

        await message_queue.listen_for_results("query_processor", lambda x, y: None)

        # Should not crash, should log error

    @pytest.mark.asyncio
    async def test_send_result_success(self, message_queue, mock_redis):
        """Test successful result sending."""
        message_queue.redis = mock_redis

        await message_queue.send_result("query_processor", {"result": "success"})

        mock_redis.publish.assert_called_once()

        call_args = mock_redis.publish.call_args
        channel, message_str = call_args[0]

        assert channel == "rag:agents:query_processor_results"
        message = json.loads(message_str)

        assert message["agent"] == "query_processor"
        assert message["result"] == {"result": "success"}
        assert message["status"] == "success"
        assert "timestamp" in message

    @pytest.mark.asyncio
    async def test_send_result_with_task_id(self, message_queue, mock_redis):
        """Test result sending with task ID."""
        message_queue.redis = mock_redis

        await message_queue.send_result(
            "retrieval_agent", {"data": "retrieved"}, "task_123"
        )

        call_args = mock_redis.publish.call_args
        message = json.loads(call_args[0][1])

        assert message["task_id"] == "task_123"

    @pytest.mark.asyncio
    async def test_send_result_unknown_agent(self, message_queue, mock_redis):
        """Test sending result for unknown agent."""
        message_queue.redis = mock_redis

        with pytest.raises(
            AgentCommunicationError, match="Unknown agent: unknown_agent"
        ):
            await message_queue.send_result("unknown_agent", {})

    @pytest.mark.asyncio
    async def test_get_agent_status_available(self, message_queue, mock_redis):
        """Test getting available agent status."""
        message_queue.redis = mock_redis

        status = await message_queue.get_agent_status("query_processor")

        assert status["agent"] == "query_processor"
        assert status["status"] == "available"
        assert "last_check" in status

    @pytest.mark.asyncio
    async def test_get_agent_status_unavailable(self, message_queue, mock_redis):
        """Test getting unavailable agent status."""
        message_queue.redis = mock_redis
        mock_redis.ping.side_effect = Exception("Connection lost")

        status = await message_queue.get_agent_status("query_processor")

        assert status["agent"] == "query_processor"
        assert status["status"] == "unavailable"
        assert "error" in status

    @pytest.mark.asyncio
    async def test_cleanup_stale_messages(self, message_queue, mock_redis):
        """Test message cleanup functionality."""
        message_queue.redis = mock_redis

        await message_queue.cleanup_stale_messages(3600)

        # Should not crash - implementation is minimal

    @pytest.mark.asyncio
    async def test_context_manager(self, message_queue, mock_redis):
        """Test async context manager."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            async with message_queue:
                assert message_queue.redis == mock_redis

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_connection_auto_connect(self, message_queue, mock_redis):
        """Test automatic connection establishment."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            await message_queue._ensure_connection()

            assert message_queue.redis == mock_redis

    @pytest.mark.asyncio
    async def test_high_priority_task(self, message_queue, mock_redis):
        """Test sending high priority task."""
        message_queue.redis = mock_redis

        await message_queue.send_task_to_agent(
            "coordinator", {"urgent": "task"}, "high"
        )

        call_args = mock_redis.publish.call_args
        message = json.loads(call_args[0][1])

        assert message["priority"] == "high"

    @pytest.mark.asyncio
    async def test_broadcast_partial_failure(self, message_queue, mock_redis):
        """Test broadcast with partial agent failures."""
        message_queue.redis = mock_redis

        # Mock send_task_to_agent to fail for some agents
        original_send = message_queue.send_task_to_agent
        call_count = 0

        async def mock_send(agent_name, task):
            nonlocal call_count
            call_count += 1
            if agent_name == "retrieval_agent":
                raise Exception("Agent unavailable")
            return await original_send(agent_name, task)

        message_queue.send_task_to_agent = mock_send

        await message_queue.send_broadcast({"message": "test"})

        # Should continue despite one failure
        assert call_count == 6
