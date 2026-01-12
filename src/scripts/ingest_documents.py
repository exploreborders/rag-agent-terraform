#!/usr/bin/env python3
"""
Document Ingestion Script

This script processes and ingests documents into the vector database for RAG.
Supports PDF, text, and image files with automatic text extraction and chunking.
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path
from typing import List
import mimetypes

# Add the app directory to the path so we can import modules
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from app.rag_agent import RAGAgent

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".jpg", ".jpeg", ".png"}


def find_documents(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all supported document files in the directory."""
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob(f"{pattern}{ext}"))

    return sorted(files)


def get_content_type(file_path: Path) -> str:
    """Get MIME content type for a file."""
    content_type, _ = mimetypes.guess_type(str(file_path))
    return content_type or "application/octet-stream"


async def ingest_document(file_path: Path, rag_agent: RAGAgent) -> bool:
    """Ingest a single document using the RAG agent."""
    try:
        logger.info(f"Processing document: {file_path}")

        # Determine content type
        content_type = get_content_type(file_path)

        # Copy file to upload directory first (RAG agent expects files there)
        upload_path = rag_agent.document_loader.upload_dir / file_path.name

        # Copy the file to upload directory
        import shutil

        shutil.copy2(file_path, upload_path)

        try:
            # Process the document using RAG agent (pass just the filename)
            result = await rag_agent.process_document(
                file_path=file_path.name,  # Just filename, RAG agent will look in upload_dir
                content_type=content_type,
            )

            if not result or not result.id:
                logger.warning(f"No content extracted from {file_path}")
                return False

            logger.info(f"Successfully ingested {file_path} (ID: {result.id})")
            return True

        finally:
            # Clean up the copied file
            if upload_path.exists():
                upload_path.unlink()

    except Exception as e:
        logger.error(f"Failed to ingest {file_path}: {e}")
        return False


async def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory containing documents",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=True,
        help="Recursively search subdirectories (default: True)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    logger.info("Starting document ingestion process...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Recursive search: {args.recursive}")

    try:
        # Initialize RAG agent (it will create all necessary components)
        rag_agent = RAGAgent()
        await rag_agent.initialize()

        # Find documents
        documents = find_documents(input_dir, args.recursive)
        if not documents:
            logger.warning(f"No supported documents found in {input_dir}")
            return

        logger.info(f"Found {len(documents)} documents to process")

        # Process each document
        success_count = 0
        for doc_path in documents:
            if await ingest_document(doc_path, rag_agent):
                success_count += 1

        logger.info(
            f"Document ingestion completed: {success_count}/{len(documents)} successful"
        )

    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
