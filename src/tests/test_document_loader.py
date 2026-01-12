"""Unit tests for document loader."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.document_loader import (
    DocumentLoader,
    DocumentProcessingError,
    UnsupportedFileTypeError,
)


class TestDocumentLoader:
    """Test cases for document loader."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        upload_dir = Path(tempfile.mkdtemp())
        processed_dir = Path(tempfile.mkdtemp())

        yield upload_dir, processed_dir

        # Cleanup
        shutil.rmtree(upload_dir)
        shutil.rmtree(processed_dir)

    @pytest.fixture
    def document_loader(self, temp_dirs):
        """Document loader instance with temp directories."""
        upload_dir, processed_dir = temp_dirs
        return DocumentLoader(
            upload_dir=str(upload_dir), processed_dir=str(processed_dir)
        )

    def test_initialization(self, temp_dirs):
        """Test document loader initialization."""
        upload_dir, processed_dir = temp_dirs
        loader = DocumentLoader(
            upload_dir=str(upload_dir), processed_dir=str(processed_dir)
        )

        assert loader.upload_dir == upload_dir
        assert loader.processed_dir == processed_dir

    def test_supported_types(self, document_loader):
        """Test supported content types."""
        supported = document_loader.get_supported_types()
        expected = [
            "application/pdf",
            "text/plain",
            "text/markdown",
            "text/x-python",
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ]
        assert set(supported) == set(expected)

    def test_save_uploaded_file(self, document_loader):
        """Test saving uploaded file."""
        test_content = b"test file content"
        file_obj = MagicMock()
        file_obj.read.return_value = test_content

        saved_path = document_loader.save_uploaded_file(file_obj, "test.txt")

        assert saved_path.exists()
        assert saved_path.read_bytes() == test_content
        assert "test.txt" in str(saved_path)

    def test_chunk_text_simple(self, document_loader):
        """Test simple text chunking."""
        text = "This is a simple test text for chunking."
        chunks = document_loader._chunk_text(text, chunk_size=10)

        assert len(chunks) > 1
        assert all(len(chunk) <= 10 + 200 for chunk in chunks)  # chunk_size + overlap

    def test_chunk_text_with_overlap(self, document_loader):
        """Test text chunking with overlap."""
        text = "This is a longer text that should be split into multiple chunks with overlap."
        chunks = document_loader._chunk_text(text, chunk_size=20, overlap=5)

        assert len(chunks) > 1
        # Check that chunks have some overlap
        for i in range(len(chunks) - 1):
            current_end = chunks[i][-10:]
            next_start = chunks[i + 1][:10]
            # Some overlap should exist
            assert len(current_end) > 0 and len(next_start) > 0

    def test_chunk_text_small(self, document_loader):
        """Test chunking text smaller than chunk size."""
        text = "Short text"
        chunks = document_loader._chunk_text(text, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    @patch("app.document_loader.fitz")
    def test_process_pdf_success(self, mock_fitz, document_loader, temp_dirs):
        """Test successful PDF processing."""
        # Mock PDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page content"
        mock_doc.__len__.return_value = 2
        mock_doc.load_page.return_value = mock_page
        mock_fitz.open.return_value = mock_doc

        upload_dir, _ = temp_dirs
        pdf_path = upload_dir / "test.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        text, metadata = document_loader._process_pdf(pdf_path)

        assert "Page content" in text
        assert metadata["page_count"] == 2
        assert "word_count" in metadata

    @patch("app.document_loader.fitz")
    def test_process_pdf_corruption_error(self, mock_fitz, document_loader, temp_dirs):
        """Test PDF processing with corruption error."""
        mock_fitz.open.side_effect = Exception("PDF corrupted")

        upload_dir, _ = temp_dirs
        pdf_path = upload_dir / "corrupted.pdf"
        pdf_path.write_bytes(b"corrupted content")

        with pytest.raises(DocumentProcessingError):
            document_loader._process_pdf(pdf_path)

    def test_process_text_success(self, document_loader, temp_dirs):
        """Test successful text processing."""
        upload_dir, _ = temp_dirs
        text_path = upload_dir / "test.txt"
        test_content = (
            "This is a test document.\nIt has multiple lines.\nAnd some words."
        )
        text_path.write_text(test_content)

        text, metadata = document_loader._process_text(text_path)

        assert text == test_content
        assert metadata["word_count"] == len(test_content.split())

    def test_process_text_encoding_fallback(self, document_loader, temp_dirs):
        """Test text processing with encoding fallback."""
        upload_dir, _ = temp_dirs
        text_path = upload_dir / "test.txt"
        test_content = "Test content with special chars: àáâãäå"
        # Write with specific encoding
        text_path.write_text(test_content, encoding="utf-8")

        text, metadata = document_loader._process_text(text_path)

        assert text == test_content
        assert metadata["word_count"] > 0

    @patch("app.document_loader.Image")
    def test_process_image_success(self, mock_image_class, document_loader, temp_dirs):
        """Test successful image processing."""
        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.width = 100
        mock_image.height = 200
        mock_image.mode = "RGB"
        mock_image_class.open.return_value = mock_image

        # Mock pytesseract
        with patch(
            "app.document_loader.pytesseract.image_to_string",
            return_value="Extracted text",
        ):
            upload_dir, _ = temp_dirs
            image_path = upload_dir / "test.jpg"
            image_path.write_bytes(b"fake image content")

            text, metadata = document_loader._process_image(image_path)

            assert text == "Extracted text"
            assert metadata["width"] == 100
            assert metadata["height"] == 200
            assert metadata["mode"] == "RGB"

    @patch("app.document_loader.Image")
    def test_process_image_ocr_failure(
        self, mock_image_class, document_loader, temp_dirs
    ):
        """Test image processing when OCR fails."""
        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.width = 100
        mock_image.height = 200
        mock_image.mode = "RGB"
        mock_image_class.open.return_value = mock_image

        # Mock pytesseract failure
        with patch(
            "app.document_loader.pytesseract.image_to_string",
            side_effect=Exception("OCR failed"),
        ):
            upload_dir, _ = temp_dirs
            image_path = upload_dir / "test.jpg"
            image_path.write_bytes(b"fake image content")

            text, metadata = document_loader._process_image(image_path)

            assert text == ""  # Empty text on OCR failure
            assert "ocr_error" in metadata

    def test_process_document_pdf(self, document_loader, temp_dirs):
        """Test full document processing for PDF."""
        with patch.object(document_loader, "_process_pdf") as mock_process:
            mock_process.return_value = (
                "PDF content",
                {"page_count": 5, "word_count": 100},
            )

            upload_dir, _ = temp_dirs
            pdf_path = upload_dir / "test.pdf"
            pdf_path.write_bytes(b"pdf content")

            result = document_loader.process_document(pdf_path, "application/pdf")

            assert result["document_id"] == document_loader._calculate_checksum(
                pdf_path
            )
            assert result["text"] == "PDF content"
            assert len(result["chunks"]) > 0
            assert result["chunks"][0]["content"] == "PDF content"

    def test_process_document_text(self, document_loader, temp_dirs):
        """Test full document processing for text."""
        with patch.object(document_loader, "_process_text") as mock_process:
            mock_process.return_value = ("Text content", {"word_count": 50})

            upload_dir, _ = temp_dirs
            text_path = upload_dir / "test.txt"
            text_path.write_text("Text content")

            result = document_loader.process_document(text_path, "text/plain")

            assert result["text"] == "Text content"
            assert len(result["chunks"]) > 0

    def test_process_document_unsupported_type(self, document_loader, temp_dirs):
        """Test processing document with unsupported type."""
        upload_dir, _ = temp_dirs
        file_path = upload_dir / "test.xyz"
        file_path.write_bytes(b"unsupported content")

        with pytest.raises(UnsupportedFileTypeError):
            document_loader.process_document(file_path, "application/xyz")

    def test_calculate_checksum(self, document_loader, temp_dirs):
        """Test checksum calculation."""
        upload_dir, _ = temp_dirs
        file_path = upload_dir / "checksum_test.txt"
        file_path.write_text("test content")

        checksum1 = document_loader._calculate_checksum(file_path)
        checksum2 = document_loader._calculate_checksum(file_path)

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 hex length

    def test_cleanup_old_files(self, document_loader, temp_dirs):
        """Test cleanup of old files."""
        import time
        from unittest.mock import patch

        upload_dir, _ = temp_dirs

        # Create a mock old file
        old_file = upload_dir / "old_file.txt"
        old_file.write_text("old content")

        # Mock the file modification time to be old
        old_time = time.time() - (8 * 24 * 60 * 60)  # 8 days ago

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_mtime=old_time)

            removed = document_loader.cleanup_old_files(days_old=7)
            assert removed == 1
