"""
pdf_utils.py
============
Robust PDF text extraction utility using PyMuPDF (fitz).

Handles:
- File paths, byte streams, and file objects
- Empty PDFs (0 pages)
- Scanned / image-only PDFs with zero extractable text
- Corrupted or invalid PDF files
- Graceful error reporting without crashing the host application
- Safe cleanup of resources
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None

try:
    import pypdf
except ImportError:
    pypdf = None


class PDFProcessingError(Exception):
    """Custom exception raised when PDF extraction fails due to formatting or content issues."""
    pass


def _extract_with_pypdf(pdf_source: str | Path | bytes | BinaryIO) -> str:
    """Extract text using pure-Python pypdf (WASM / browser friendly)."""
    if pypdf is None:
        raise PDFProcessingError("pypdf is not available.")

    try:
        if isinstance(pdf_source, (str, Path)):
            path_obj = Path(pdf_source)
            if not path_obj.exists():
                raise ValueError(f"The specified PDF file was not found: {path_obj.name}")
            if path_obj.stat().st_size == 0:
                raise PDFProcessingError("The uploaded PDF file is empty (0 bytes).")
            reader = pypdf.PdfReader(str(path_obj))
        elif isinstance(pdf_source, (bytes, bytearray)):
            if len(pdf_source) == 0:
                raise PDFProcessingError("The provided byte stream is empty (0 bytes).")
            reader = pypdf.PdfReader(io.BytesIO(pdf_source))
        elif hasattr(pdf_source, "read"):
            data = pdf_source.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            if len(data) == 0:
                raise PDFProcessingError("The uploaded file stream is empty (0 bytes).")
            reader = pypdf.PdfReader(io.BytesIO(data))
        else:
            raise ValueError(f"Unsupported PDF input type: {type(pdf_source).__name__}")

        if len(reader.pages) == 0:
            raise PDFProcessingError("The PDF document contains 0 pages and cannot be processed.")

        extracted_chunks: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                extracted_chunks.append(t)

        full_text = "\n".join(extracted_chunks).strip()
        if not full_text:
            raise PDFProcessingError(
                "The PDF does not contain extractable text. Scanned or image-only PDFs are not supported without OCR."
            )
        return full_text
    except (ValueError, PDFProcessingError):
        raise
    except Exception as exc:
        raise PDFProcessingError(f"PDF extraction error: {exc}") from exc


def extract_text_from_pdf(pdf_source: str | Path | bytes | BinaryIO | None) -> str:
    """
    Extract readable text from a PDF source using PyMuPDF or pypdf fallback.

    Args:
        pdf_source: A path to a PDF file, raw bytes, or an open binary stream.

    Returns:
        The extracted and stripped plain text.

    Raises:
        ValueError: If the input is missing, empty, or unreadable.
        PDFProcessingError: If the PDF is corrupted, password-protected, or has no extractable text.
    """
    if pdf_source is None:
        raise ValueError("No PDF file provided. Please upload a valid PDF document.")

    # 1. Try PyMuPDF if installed
    if fitz is not None:
        doc: fitz.Document | None = None
        extracted_text_chunks: list[str] = []

        try:
            # Handle file path
            if isinstance(pdf_source, (str, Path)):
                path_obj = Path(pdf_source)
                if not path_obj.exists():
                    raise ValueError(f"The specified PDF file was not found: {path_obj.name}")
                if path_obj.stat().st_size == 0:
                    raise PDFProcessingError(
                        "The uploaded PDF file is empty (0 bytes). Please upload a valid document."
                    )
                doc = fitz.open(str(path_obj))

            # Handle raw bytes
            elif isinstance(pdf_source, (bytes, bytearray)):
                if len(pdf_source) == 0:
                    raise PDFProcessingError("The provided byte stream is empty (0 bytes).")
                doc = fitz.open(stream=pdf_source, filetype="pdf")

            # Handle file-like objects
            elif hasattr(pdf_source, "read"):
                data = pdf_source.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
                if len(data) == 0:
                    raise PDFProcessingError("The uploaded file stream is empty (0 bytes).")
                doc = fitz.open(stream=data, filetype="pdf")

            else:
                raise ValueError(
                    f"Unsupported PDF input type: {type(pdf_source).__name__}. "
                    "Expected file path, bytes, or file stream."
                )

            # Check page count
            if doc.page_count == 0:
                raise PDFProcessingError(
                    "The PDF document contains 0 pages and cannot be processed."
                )

            # Extract text page by page
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                page_text = page.get_text("text")
                if page_text:
                    extracted_text_chunks.append(page_text)

            full_text = "\n".join(extracted_text_chunks).strip()

            # Verify that readable text was actually extracted
            if not full_text:
                raise PDFProcessingError(
                    "The PDF does not contain extractable text. "
                    "Scanned or image-only PDFs are not supported without OCR."
                )

            return full_text

        except (ValueError, PDFProcessingError):
            raise
        except Exception:
            # If PyMuPDF fails or errors, try fallback to pypdf if available
            if pypdf is not None:
                return _extract_with_pypdf(pdf_source)
            raise
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    # 2. Fallback to pure-Python pypdf (WASM / Pyodide)
    if pypdf is not None:
        return _extract_with_pypdf(pdf_source)

    raise PDFProcessingError("Neither PyMuPDF nor pypdf is available for PDF processing.")
