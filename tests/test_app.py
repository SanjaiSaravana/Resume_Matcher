"""
tests/test_app.py
=================
Integration and validation tests for the Gradio interface logic in app.py.
"""

from __future__ import annotations

from pathlib import Path
import pytest
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from app import build_app, compute_match, load_sample_content


@pytest.fixture
def temp_pdf_resume(tmp_path: Path) -> Path:
    """Create a temporary valid PDF resume for UI testing."""
    pdf_path = tmp_path / "alex_resume.pdf"
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Alex Morgan\n"
        "Full Stack Python Engineer\n"
        "Skills: Python, FastAPI, Docker, SQL, AWS, Git, scikit-learn, pandas\n"
        "Experience building high scale web apps and ML pipelines."
    )
    page.insert_text((50, 100), text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_app_build():
    """Verify that the Gradio app blocks construct without errors."""
    demo = build_app()
    assert demo is not None


def test_compute_match_missing_resume():
    """UI validation: compute_match with None resume returns error banner."""
    score_html, text_gauge, skill_gauge, matching_html, missing_html, suggestions, status = compute_match(
        resume_file=None,
        job_description="Looking for a Python developer with Docker experience.",
    )
    assert text_gauge == 0.0
    assert skill_gauge == 0.0
    assert "Missing Resume" in status
    assert "alert-box error" in status


def test_compute_match_missing_job_description(temp_pdf_resume: Path):
    """UI validation: compute_match with empty JD returns error banner."""
    score_html, text_gauge, skill_gauge, matching_html, missing_html, suggestions, status = compute_match(
        resume_file=str(temp_pdf_resume),
        job_description="",
    )
    assert text_gauge == 0.0
    assert skill_gauge == 0.0
    assert "Missing Job Description" in status
    assert "alert-box error" in status


def test_compute_match_successful(temp_pdf_resume: Path):
    """UI integration: compute_match with valid inputs produces scores and badges."""
    job_desc = (
        "Looking for a Senior Python Developer with expertise in FastAPI, Docker, SQL, and AWS. "
        "Experience with Machine Learning or scikit-learn is a plus."
    )
    score_html, text_gauge, skill_gauge, matching_html, missing_html, suggestions, status = compute_match(
        resume_file=str(temp_pdf_resume),
        job_description=job_desc,
        skill_weight=0.4,
    )

    # Status should be success
    assert "Analysis Completed" in status
    assert "alert-box success" in status

    # Scores should be in valid range
    assert 0.0 <= text_gauge <= 100.0
    assert 0.0 <= skill_gauge <= 100.0
    assert "score-card" in score_html

    # Badges should contain matching skills
    assert "Python" in matching_html
    assert "Docker" in matching_html

    # Suggestions should be provided
    assert len(suggestions) > 0


def test_compute_match_corrupt_pdf(tmp_path: Path):
    """UI validation: uploading corrupt file displays error banner without crashing."""
    corrupt_file = tmp_path / "bad.pdf"
    corrupt_file.write_bytes(b"garbage not a pdf")

    score_html, text_gauge, skill_gauge, matching_html, missing_html, suggestions, status = compute_match(
        resume_file=str(corrupt_file),
        job_description="Seeking Python Developer",
    )
    assert text_gauge == 0.0
    assert "PDF Processing Error" in status


def test_load_sample_content():
    """Verify that sample loader function returns strings/paths."""
    pdf_path, job_text = load_sample_content()
    # If sample files exist, job_text should have content
    assert isinstance(job_text, str)
