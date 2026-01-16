"""Redis-based message queue for agent communication in Docker environment."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class AgentMessageQueueError(Exception):
    """Base exception for agent message queue errors."""

    pass


class AgentCommunicationError(AgentMessageQueueError):
    """Exception raised when agent communication fails."""

    pass


class DockerAgentMessageQueue:
    """Redis Message Queue für Multi-Agenten-Kommunikation in Docker-Umgebung.

    Diese Queue ermöglicht es Agenten, Tasks zu senden und Ergebnisse zu empfangen,
    während sie in separaten Docker-Containern laufen.
    """

    def __init__(self, redis_url: str = None):
        """Initialisiere Message Queue.

        Args:
            redis_url: Redis-Verbindungsstring
        """
        self.redis_url = redis_url or settings.redis_url
        self.redis: Optional[redis.Redis] = None

        # Standard-Message-Queue-Channels für Agenten
        self.agent_channels = {
            "query_processor": "rag:agents:query_processor",
            "retrieval_agent": "rag:agents:retrieval",
            "mcp_agent": "rag:agents:mcp",
            "aggregator": "rag:agents:aggregator",
            "validator": "rag:agents:validator",
            "coordinator": "rag:coordinator",
        }

        # Result Channels (separate für jeden Agenten)
        self.result_channels = {
            agent: f"{channel}_results"
            for agent, channel in self.agent_channels.items()
        }

    async def connect(self):
        """Stelle Verbindung zu Redis her."""
        try:
            self.redis = redis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis for agent communication")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise AgentCommunicationError(f"Redis connection failed: {e}")

    async def disconnect(self):
        """Trenne Verbindung zu Redis."""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.info("Disconnected from Redis")

    async def _ensure_connection(self):
        """Stelle sicher, dass Redis-Verbindung aktiv ist."""
        if not self.redis:
            await self.connect()

    async def send_task_to_agent(
        self, agent_name: str, task: Dict[str, Any], priority: str = "normal"
    ) -> str:
        """Sende Task an spezifischen Agenten.

        Args:
            agent_name: Name des Ziel-Agenten
            task: Task-Daten
            priority: Priorität (normal/high/low)

        Returns:
            Task-ID für Tracking

        Raises:
            AgentCommunicationError: Wenn Agent-Channel nicht gefunden wird
        """
        await self._ensure_connection()

        if agent_name not in self.agent_channels:
            raise AgentCommunicationError(f"Unknown agent: {agent_name}")

        channel = self.agent_channels[agent_name]
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        message = {
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent_name,
            "priority": priority,
            "task": task,
            "sender": "orchestrator",
        }

        try:
            await self.redis.publish(channel, json.dumps(message))
            logger.info(f"Task {task_id} sent to agent {agent_name}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to send task to {agent_name}: {e}")
            raise AgentCommunicationError(f"Task sending failed: {e}")

    async def send_broadcast(
        self, message: Dict[str, Any], target_agents: List[str] = None
    ):
        """Sende Broadcast-Message an mehrere Agenten.

        Args:
            message: Nachricht-Inhalt
            target_agents: Liste der Ziel-Agenten (None = alle)
        """
        await self._ensure_connection()

        agents = target_agents or list(self.agent_channels.keys())

        for agent_name in agents:
            if agent_name in self.agent_channels:
                try:
                    await self.send_task_to_agent(
                        agent_name, {"type": "broadcast", "message": message}
                    )
                except Exception as e:
                    logger.warning(f"Failed to broadcast to {agent_name}: {e}")

    async def listen_for_results(
        self,
        agent_name: str,
        callback: Callable[[str, Dict[str, Any]], None],
        timeout: int = 300,
    ):
        """Höre auf Ergebnisse von einem Agenten.

        Args:
            agent_name: Name des Agenten
            callback: Callback-Funktion für Ergebnisse
            timeout: Timeout in Sekunden
        """
        await self._ensure_connection()

        if agent_name not in self.result_channels:
            raise AgentCommunicationError(f"Unknown agent: {agent_name}")

        result_channel = self.result_channels[agent_name]

        pubsub = self.redis.pubsub()
        await pubsub.subscribe(result_channel)

        logger.info(f"Listening for results from {agent_name} on {result_channel}")

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        result = json.loads(message["data"])
                        await callback(agent_name, result)

                        # Prüfe auf Completion-Signal
                        if result.get("status") == "completed":
                            break

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in result message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing result: {e}")

        except Exception as e:
            logger.error(f"Error listening for results from {agent_name}: {e}")
        finally:
            await pubsub.unsubscribe(result_channel)

    async def send_result(
        self, agent_name: str, result: Dict[str, Any], task_id: str = None
    ):
        """Sende Ergebnis zurück an Orchestrator.

        Args:
            agent_name: Name des sendenden Agenten
            result: Ergebnis-Daten
            task_id: Optionale Task-ID für Korrelation
        """
        await self._ensure_connection()

        if agent_name not in self.result_channels:
            raise AgentCommunicationError(f"Unknown agent: {agent_name}")

        result_channel = self.result_channels[agent_name]

        message = {
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result,
            "task_id": task_id,
            "status": result.get("status", "success"),
        }

        try:
            await self.redis.publish(result_channel, json.dumps(message))
            logger.info(f"Result sent from {agent_name} to orchestrator")
        except Exception as e:
            logger.error(f"Failed to send result from {agent_name}: {e}")
            raise AgentCommunicationError(f"Result sending failed: {e}")

    async def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Hole Status eines Agenten.

        Args:
            agent_name: Name des Agenten

        Returns:
            Status-Informationen
        """
        await self._ensure_connection()

        # Prüfe Redis-Connectivity als Proxy für Agent-Health
        try:
            await self.redis.ping()
            return {
                "agent": agent_name,
                "status": "available",
                "last_check": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Agent {agent_name} health check failed: {e}")
            return {
                "agent": agent_name,
                "status": "unavailable",
                "error": str(e),
                "last_check": datetime.utcnow().isoformat(),
            }

    async def cleanup_stale_messages(self, max_age_seconds: int = 3600):
        """Räume alte Nachrichten auf.

        Args:
            max_age_seconds: Maximale Alter der Nachrichten
        """
        await self._ensure_connection()

        # In Redis Pub/Sub gibt es keine automatische Cleanup
        # Hier könnten wir z.B. Monitoring-Metriken sammeln
        logger.info(f"Message cleanup requested (max_age: {max_age_seconds}s)")
        # Implementierung je nach Bedarf

    # Context Manager Support
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


# Globale Message Queue Instance
message_queue = DockerAgentMessageQueue()
