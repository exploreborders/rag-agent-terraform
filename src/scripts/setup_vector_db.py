#!/usr/bin/env python3
"""
Vector Database Setup Script

This script initializes the vector database schema and ensures all necessary
tables and extensions are created for the RAG system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to the path so we can import modules
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import logging

from app.vector_store import VectorStore
from app.config import Settings

logger = logging.getLogger(__name__)


async def main():
    """Initialize the vector database schema."""
    logger.info("Starting vector database setup...")

    try:
        # Load configuration
        settings = Settings()

        # Create vector store instance
        vector_store = VectorStore()

        # Initialize schema
        logger.info("Creating database schema and extensions...")
        await vector_store.initialize_schema()

        logger.info("Vector database setup completed successfully!")

    except Exception as e:
        logger.error(f"Vector database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
