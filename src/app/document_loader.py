"""Document processing pipeline for PDF, text, and image files."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""

    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Exception raised when file type is not supported."""

    pass


class DocumentCorruptionError(DocumentProcessingError):
    """Exception raised when document is corrupted."""

    pass


class DocumentLoader:
    """Multi-format document processing pipeline."""

    def __init__(
        self, upload_dir: Optional[str] = None, processed_dir: Optional[str] = None
    ):
        """Initialize document loader.

        Args:
            upload_dir: Directory for uploaded files
            processed_dir: Directory for processed files
        """
        self.upload_dir = Path(upload_dir or settings.upload_dir)
        self.processed_dir = Path(processed_dir or settings.processed_dir)

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Configure pytesseract for OCR
        try:
            # Try to find tesseract installation
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(
                f"Tesseract not found or not configured: {e}. OCR will be unavailable."
            )

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file.

        Args:
            file_path: Path to file

        Returns:
            SHA-256 checksum as hex string
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_file_metadata(self, file_path: Path, content_type: str) -> Dict[str, Any]:
        """Extract metadata from uploaded file.

        Args:
            file_path: Path to uploaded file
            content_type: MIME content type

        Returns:
            File metadata dictionary
        """
        stat = file_path.stat()

        metadata = {
            "filename": file_path.name,
            "content_type": content_type,
            "size": stat.st_size,
            "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "checksum": self._calculate_checksum(file_path),
        }

        return metadata

    def _process_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process PDF file and extract text.

        Args:
            file_path: Path to PDF file

        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            metadata = {"page_count": len(doc)}

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"

            doc.close()

            # Count words
            word_count = len(text.split())
            metadata["word_count"] = word_count

            return text.strip(), metadata

        except Exception as e:
            raise DocumentCorruptionError(f"Failed to process PDF {file_path}: {e}")

    def _process_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process text file.

        Args:
            file_path: Path to text file

        Returns:
            Tuple of (text_content, metadata)
        """
        try:
            # Try different encodings
            encodings = ["utf-8", "latin-1", "cp1252"]
            text = None

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                raise DocumentCorruptionError(f"Could not decode text file {file_path}")

            word_count = len(text.split())
            metadata = {"word_count": word_count}

            return text, metadata

        except Exception as e:
            raise DocumentCorruptionError(
                f"Failed to process text file {file_path}: {e}"
            )

    def _process_image(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Process image file with OCR.

        Args:
            file_path: Path to image file

        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            # Open image
            image = Image.open(file_path)

            # Convert to RGB if necessary
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB")

            # Perform OCR
            text = pytesseract.image_to_string(image)

            # Get image metadata
            metadata = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "word_count": len(text.split()) if text else 0,
            }

            image.close()

            return text.strip(), metadata

        except Exception as e:
            logger.warning(
                f"OCR processing failed for {file_path}: {e}. Returning empty text."
            )
            # Return empty text with basic metadata if OCR fails
            try:
                image = Image.open(file_path)
                metadata = {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "word_count": 0,
                    "ocr_error": str(e),
                }
                image.close()
                return "", metadata
            except Exception:
                return "", {"ocr_error": str(e)}

    def _chunk_text(
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[str]:
        """Split text into chunks with overlap.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                break_chars = [". ", "! ", "? ", "\n\n"]
                break_pos = -1

                for break_char in break_chars:
                    pos = text.rfind(break_char, start, end)
                    if pos > break_pos:
                        break_pos = pos + len(break_char)

                if break_pos > start:
                    end = break_pos

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap

            # Prevent infinite loop
            if start >= len(text) or len(chunks) > settings.max_chunks_per_document:
                break

        return chunks

    def process_document(self, file_path: Path, content_type: str) -> Dict[str, Any]:
        """Process a document and return chunks with metadata.

        Args:
            file_path: Path to the document file
            content_type: MIME content type

        Returns:
            Dictionary containing document metadata and chunks
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Extract basic metadata
        metadata = self._get_file_metadata(file_path, content_type)

        # Process based on content type
        if content_type == "application/pdf":
            text, processing_metadata = self._process_pdf(file_path)
        elif content_type in ["text/plain", "text/markdown", "text/x-python"]:
            text, processing_metadata = self._process_text(file_path)
        elif content_type in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
            text, processing_metadata = self._process_image(file_path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported content type: {content_type}")

        # Merge processing metadata
        metadata.update(processing_metadata)

        # Chunk the text
        chunks = self._chunk_text(text)

        # Create chunk objects
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{metadata['checksum']}_chunk_{i}"
            document_chunks.append(
                {
                    "id": chunk_id,
                    "document_id": metadata["checksum"],  # Use checksum as document ID
                    "content": chunk_text,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "metadata": {
                        "filename": metadata["filename"],
                        "content_type": metadata["content_type"],
                        "chunk_size": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                    },
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

        result = {
            "document_id": metadata["checksum"],
            "metadata": metadata,
            "text": text,
            "chunks": document_chunks,
            "total_chunks": len(document_chunks),
        }

        logger.info(
            f"Processed document {file_path.name}: {len(document_chunks)} chunks"
        )
        return result

    def save_uploaded_file(self, file: BinaryIO, filename: str) -> Path:
        """Save uploaded file to disk.

        Args:
            file: Uploaded file object
            filename: Original filename

        Returns:
            Path to saved file
        """
        # Generate unique filename to avoid conflicts
        file_hash = hashlib.md5(
            filename.encode() + str(datetime.utcnow().timestamp()).encode()
        ).hexdigest()[:8]
        safe_filename = f"{file_hash}_{filename}"
        file_path = self.upload_dir / safe_filename

        # Save file
        with open(file_path, "wb") as f:
            content = file.read()
            f.write(content)

        logger.info(f"Saved uploaded file: {file_path}")
        return file_path

    def cleanup_old_files(self, days_old: int = 7) -> int:
        """Clean up old uploaded files.

        Args:
            days_old: Remove files older than this many days

        Returns:
            Number of files removed
        """
        cutoff_time = datetime.utcnow().timestamp() - (days_old * 24 * 60 * 60)
        removed_count = 0

        for file_path in self.upload_dir.glob("*"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                removed_count += 1

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old files")

        return removed_count

    def get_supported_types(self) -> List[str]:
        """Get list of supported MIME content types.

        Returns:
            List of supported content types
        """
        return [
            "application/pdf",
            "text/plain",
            "text/markdown",
            "text/x-python",
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ]
