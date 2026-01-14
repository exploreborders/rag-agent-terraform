"""PostgreSQL-based persistence for LangGraph checkpoints."""

import logging
from typing import Optional

from langgraph.checkpoint.postgres import AsyncPostgresSaver

from app.config import settings

logger = logging.getLogger(__name__)


class GraphPersistenceError(Exception):
    """Exception raised when graph persistence fails."""

    pass


async def setup_graph_persistence(db_url: Optional[str] = None) -> AsyncPostgresSaver:
    """Initialisiere PostgreSQL Checkpointer für LangGraph.

    Args:
        db_url: Datenbank-URL (optional, verwendet settings.database_url)

    Returns:
        Konfigurierter AsyncPostgresSaver

    Raises:
        GraphPersistenceError: Bei Initialisierungsfehlern
    """
    db_url = db_url or settings.database_url

    try:
        # Erstelle Checkpointer-Instanz
        checkpointer = AsyncPostgresSaver.from_conn_string(db_url)

        # Initialisiere Datenbankschema für Checkpoints
        await checkpointer.setup()

        logger.info("LangGraph PostgreSQL persistence initialized successfully")
        return checkpointer

    except Exception as e:
        logger.error(f"Failed to initialize graph persistence: {e}")
        raise GraphPersistenceError(f"Graph persistence setup failed: {e}")


async def get_checkpoint_stats(checkpointer: AsyncPostgresSaver) -> dict:
    """Hole Statistiken über gespeicherte Checkpoints.

    Args:
        checkpointer: Initialisierter Checkpointer

    Returns:
        Dictionary mit Checkpoint-Statistiken
    """
    try:
        # Hier könnten wir SQL-Queries ausführen um Statistiken zu sammeln
        # Für Phase 1 reicht eine Basis-Implementierung
        return {
            "status": "operational",
            "checkpointer_type": "AsyncPostgresSaver",
            "database_url": checkpointer.conn_string
            if hasattr(checkpointer, "conn_string")
            else "configured",
        }
    except Exception as e:
        logger.warning(f"Failed to get checkpoint stats: {e}")
        return {"status": "error", "error": str(e)}


async def cleanup_old_checkpoints(
    checkpointer: AsyncPostgresSaver, max_age_days: int = 30
) -> int:
    """Räume alte Checkpoints auf.

    Args:
        checkpointer: Checkpointer-Instanz
        max_age_days: Maximale Alter in Tagen

    Returns:
        Anzahl der gelöschten Checkpoints
    """
    try:
        # Hier würde die Cleanup-Logik implementiert werden
        # Für Phase 1 als Platzhalter
        logger.info(f"Checkpoint cleanup requested (max_age: {max_age_days} days)")
        return 0  # Placeholder
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoints: {e}")
        return 0


class GraphPersistenceManager:
    """Manager für Graph-Persistenz und Lifecycle-Management."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.database_url
        self.checkpointer: Optional[AsyncPostgresSaver] = None

    async def initialize(self) -> AsyncPostgresSaver:
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

            # Versuche eine Operation durchzuführen
            stats = await self.get_stats()
            return {
                "status": "healthy",
                "checkpointer": "AsyncPostgresSaver",
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
