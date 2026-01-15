"""Simple persistence for LangGraph checkpoints (Phase 1)."""

import logging
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver

from app.config import settings

logger = logging.getLogger(__name__)


class GraphPersistenceError(Exception):
    """Exception raised when graph persistence fails."""

    pass


async def setup_graph_persistence(db_url: Optional[str] = None) -> MemorySaver:
    """Initialisiere einfachen In-Memory Checkpointer für LangGraph (Phase 1).

    Args:
        db_url: Datenbank-URL (optional, wird für Phase 1 ignoriert)

    Returns:
        Konfigurierter MemorySaver

    Note:
        Phase 1 verwendet In-Memory Persistence. PostgreSQL wird in späteren Phasen hinzugefügt.
    """
    try:
        # Verwende einfachen Memory Checkpointer für Phase 1
        checkpointer = MemorySaver()

        logger.info("LangGraph Memory persistence initialized successfully (Phase 1)")
        return checkpointer

    except Exception as e:
        logger.error(f"Failed to initialize graph persistence: {e}")
        raise GraphPersistenceError(f"Graph persistence setup failed: {e}")


async def get_checkpoint_stats(checkpointer: MemorySaver) -> dict:
    """Hole Statistiken über gespeicherte Checkpoints.

    Args:
        checkpointer: Initialisierter Checkpointer

    Returns:
        Dictionary mit Checkpoint-Statistiken
    """
    try:
        # Für MemorySaver haben wir begrenzte Statistiken
        return {
            "status": "operational",
            "checkpointer_type": "MemorySaver",
            "note": "Phase 1: In-memory persistence",
        }
    except Exception as e:
        logger.warning(f"Failed to get checkpoint stats: {e}")
        return {"status": "error", "error": str(e)}


async def cleanup_old_checkpoints(
    checkpointer: MemorySaver, max_age_days: int = 30
) -> int:
    """Räume alte Checkpoints auf.

    Args:
        checkpointer: Checkpointer-Instanz
        max_age_days: Maximale Alter in Tagen

    Returns:
        Anzahl der gelöschten Checkpoints
    """
    try:
        # MemorySaver hat keine persistenten Checkpoints zum Aufräumen
        logger.info(f"Memory checkpoint cleanup requested (no-op for Phase 1)")
        return 0
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoints: {e}")
        return 0


class GraphPersistenceManager:
    """Manager für Graph-Persistenz und Lifecycle-Management."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.database_url
        self.checkpointer: Optional[MemorySaver] = None

    async def initialize(self) -> MemorySaver:
        """Initialisiere Persistence-System."""
        if self.checkpointer is None:
            self.checkpointer = await setup_graph_persistence(self.db_url)
        return self.checkpointer

    async def get_stats(self) -> dict:
        """Hole Persistence-Statistiken."""
        if self.checkpointer is None:
            return {"status": "not_initialized"}
        return await get_checkpoint_stats(self.checkpointer)

    async def cleanup(self, max_age_days: int = 30) -> int:
        """Räume alte Daten auf."""
        if self.checkpointer is None:
            return 0
        return await cleanup_old_checkpoints(self.checkpointer, max_age_days)

    async def health_check(self) -> dict:
        """Führe Health Check durch."""
        try:
            if self.checkpointer is None:
                return {"status": "not_initialized"}

            # Teste einfache Operation
            stats = await self.get_stats()
            return {
                "status": "healthy",
                "checkpointer": "MemorySaver",
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"Graph persistence health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    # Context Manager Support
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup falls nötig
        pass


# Globale Persistence Manager Instance
persistence_manager = GraphPersistenceManager()
