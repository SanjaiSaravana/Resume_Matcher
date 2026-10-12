"""
tests/test_pdf_utils.py
=======================
Automated tests for PyMuPDF text extraction in pdf_utils.py.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from pdf_utils import PDFProcessingError, extract_text_from_pdf


@pytest.fixture
def sample_valid_pdf(tmp_path: Path) -> Path:
    """Create a temporary valid PDF document with known text."""
    pdf_path = tmp_path / "valid_resume.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 100),
        "Jane Doe\nSenior Python Engineer\nSkills: Python, SQL, Docker, AWS",
        fontsize=12,
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_blank_pdf(tmp_path: Path) -> Path:
    """Create a temporary PDF with a blank page (no extractable text)."""
    pdf_path = tmp_path / "blank_document.pdf"
    doc = fitz.open()
    doc.new_page()  # Blank page without any text
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_extract_text_valid_file(sample_valid_pdf: Path):
    """Test text extraction from a valid PDF file path."""
    extracted = extract_text_from_pdf(sample_valid_pdf)
    assert "Jane Doe" in extracted
    assert "Senior Python Engineer" in extracted
    assert "Skills: Python, SQL, Docker, AWS" in extracted


def test_extract_text_valid_bytes(sample_valid_pdf: Path):
    """Test text extraction from raw bytes."""
    pdf_bytes = sample_valid_pdf.read_bytes()
    extracted = extract_text_from_pdf(pdf_bytes)
    assert "Jane Doe" in extracted


def test_extract_text_valid_stream(sample_valid_pdf: Path):
    """Test text extraction from an in-memory BytesIO stream."""
    with open(sample_valid_pdf, "rb") as f:
        stream = io.BytesIO(f.read())
    extracted = extract_text_from_pdf(stream)
    assert "Jane Doe" in extracted


def test_extract_text_none_input():
    """Providing None must raise ValueError."""
    with pytest.raises(ValueError, match="No PDF file provided"):
        extract_text_from_pdf(None)


def test_extract_text_nonexistent_file(tmp_path: Path):
    """Missing file path must raise ValueError."""
    fake_path = tmp_path / "does_not_exist.pdf"
    with pytest.raises(ValueError, match="was not found"):
        extract_text_from_pdf(fake_path)


def test_extract_text_empty_file(tmp_path: Path):
    """0-byte file must raise PDFProcessingError."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.touch()
    with pytest.raises(PDFProcessingError, match="empty"):
        extract_text_from_pdf(empty_file)


def test_extract_text_corrupted_file(tmp_path: Path):
    """Arbitrary non-PDF bytes must raise PDFProcessingError."""
    corrupt_file = tmp_path / "corrupted.pdf"
    corrupt_file.write_bytes(b"NOT A REAL PDF FILE CONTENT AT ALL 123456789")
    with pytest.raises(PDFProcessingError):
        extract_text_from_pdf(corrupt_file)


def test_extract_text_blank_pdf(sample_blank_pdf: Path):
    """PDF with no extractable text must raise PDFProcessingError with useful guidance."""
    with pytest.raises(PDFProcessingError, match="Scanned or image-only PDFs are not supported without OCR"):
        extract_text_from_pdf(sample_blank_pdf)
