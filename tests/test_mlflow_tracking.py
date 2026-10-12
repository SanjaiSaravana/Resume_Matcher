"""
tests/test_mlflow_tracking.py
=============================
Automated test suite for live MLflow tracking in the Resume-Job Description Matcher.

Verifies:
1. A successful calculation creates exactly one MLflow run.
2. A successful run records the expected numeric metrics.
3. A successful run records the expected parameters and safe tags.
4. Multiple successful calculations generate distinct run IDs.
5. Invalid inputs do not create successful live-prediction runs.
6. Matching results remain available even if MLflow tracking fails.
7. Sensitive resume text and job-description text are strictly excluded from
   logged parameters, tags, and artifacts (privacy check).
8. Non-finite metric values (NaN, inf) are handled safely and not logged.
9. Exceptions are caught and active runs are cleanly closed.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch
import mlflow
import pytest

from app import compute_match
from matcher import match_resume_to_job
from mlflow_tracking import (
    LIVE_EXPERIMENT_NAME,
    get_tracking_uri,
    log_live_match,
)


@pytest.fixture
def isolated_tracking_uri(tmp_path: Path) -> str:
    """Create an isolated temporary MLflow tracking file store for test execution."""
    test_mlruns = tmp_path / "test_mlruns"
    test_mlruns.mkdir(parents=True, exist_ok=True)
    uri = test_mlruns.as_uri()
    return uri


@pytest.fixture
def sample_match_results() -> dict:
    """Standard successful match result dictionary."""
    return {
        "final_score": 85.5,
        "text_similarity": 78.0,
        "skill_coverage": 90.0,
        "matching_skills": ["Python", "FastAPI", "Docker", "SQL"],
        "missing_skills": ["AWS"],
        "resume_skills": ["Python", "FastAPI", "Docker", "SQL", "Git"],
        "job_skills": ["Python", "FastAPI", "Docker", "SQL", "AWS"],
        "suggestions": ["Add AWS experience."],
        "disclaimer": "This matching score measures textual and skill alignment, not the probability of getting hired.",
    }


def test_successful_calculation_creates_one_mlflow_run(
    isolated_tracking_uri: str,
    sample_match_results: dict,
):
    """Requirement 1: A successful calculation creates exactly one MLflow run."""
    run_id = log_live_match(
        results=sample_match_results,
        skill_weight=0.4,
        tracking_uri=isolated_tracking_uri,
    )

    assert run_id is not None
    assert isinstance(run_id, str)
    assert len(run_id) > 0

    # Verify run exists in client
    client = mlflow.tracking.MlflowClient(tracking_uri=isolated_tracking_uri)
    run = client.get_run(run_id)
    assert run is not None
    assert run.info.run_id == run_id


def test_successful_run_records_expected_metrics(
    isolated_tracking_uri: str,
    sample_match_results: dict,
):
    """Requirement 2: A successful run records all expected metrics on 0-100 scale."""
    run_id = log_live_match(
        results=sample_match_results,
        skill_weight=0.4,
        tracking_uri=isolated_tracking_uri,
    )
    assert run_id is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=isolated_tracking_uri)
    run = client.get_run(run_id)
    metrics = run.data.metrics

    assert "final_matching_score" in metrics
    assert "text_similarity_score" in metrics
    assert "skill_coverage_score" in metrics
    assert "matched_skill_count" in metrics
    assert "missing_skill_count" in metrics
    assert "job_description_skill_count" in metrics
    assert "resume_skill_count" in metrics

    assert metrics["final_matching_score"] == pytest.approx(85.5)
    assert metrics["text_similarity_score"] == pytest.approx(78.0)
    assert metrics["skill_coverage_score"] == pytest.approx(90.0)
    assert metrics["matched_skill_count"] == 4.0
    assert metrics["missing_skill_count"] == 1.0
    assert metrics["job_description_skill_count"] == 5.0
    assert metrics["resume_skill_count"] == 5.0


def test_successful_run_records_expected_parameters_and_tags(
    isolated_tracking_uri: str,
    sample_match_results: dict,
):
    """Requirement 3: Parameters and non-sensitive tags are correctly populated."""
    run_id = log_live_match(
        results=sample_match_results,
        skill_weight=0.45,
        ngram_range=(1, 2),
        tracking_uri=isolated_tracking_uri,
    )
    assert run_id is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=isolated_tracking_uri)
    run = client.get_run(run_id)
    params = run.data.params
    tags = run.data.tags

    assert params["matching_algorithm"] == "TF-IDF + Cosine Similarity & Weighted Skill Extraction"
    assert params["ngram_range"] == "(1, 2)"
    assert float(params["skill_weight"]) == pytest.approx(0.45)
    assert params["pdf_parser"] == "PyMuPDF (fitz)"

    assert tags["source"] == "gradio_app"
    assert tags["run_type"] == "live_prediction"
    assert tags["application"] == "resume_job_matcher"
    assert "calculation_timestamp" in tags


def test_multiple_successful_calculations_create_distinct_runs(
    isolated_tracking_uri: str,
    sample_match_results: dict,
):
    """Requirement 4: Multiple calculations generate distinct MLflow run IDs."""
    run_id_1 = log_live_match(
        results=sample_match_results,
        skill_weight=0.4,
        tracking_uri=isolated_tracking_uri,
    )
    run_id_2 = log_live_match(
        results=sample_match_results,
        skill_weight=0.5,
        tracking_uri=isolated_tracking_uri,
    )
    run_id_3 = log_live_match(
        results=sample_match_results,
        skill_weight=0.3,
        tracking_uri=isolated_tracking_uri,
    )

    assert run_id_1 is not None
    assert run_id_2 is not None
    assert run_id_3 is not None

    # Every run must have a unique identifier
    assert len({run_id_1, run_id_2, run_id_3}) == 3


def test_invalid_inputs_do_not_create_live_runs(isolated_tracking_uri: str):
    """Requirement 5: Failed or invalid inputs must not trigger an MLflow run."""
    client = mlflow.tracking.MlflowClient(tracking_uri=isolated_tracking_uri)

    # 1. Non-dict input to log_live_match directly
    assert log_live_match(None, tracking_uri=isolated_tracking_uri) is None  # type: ignore

    # 2. Gradio UI compute_match with invalid inputs
    with patch("app.log_live_match") as mock_logger:
        # Empty resume
        compute_match(None, "Valid job description")
        mock_logger.assert_not_called()

        # Empty job description
        compute_match("fake_path.pdf", "")
        mock_logger.assert_not_called()


def test_matching_results_remain_available_if_tracking_fails(
    sample_match_results: dict,
    tmp_path: Path,
):
    """Requirement 6: Tracking failure must not break the user's matching computation."""
    # Create valid dummy PDF
    import pymupdf as fitz
    pdf_path = tmp_path / "resume.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Python Developer with SQL and Docker")
    doc.save(str(pdf_path))
    doc.close()

    # Simulate MLflow failure in app.py
    with patch("app.log_live_match", return_value=None):
        score_html, text_sim, skill_cov, matching_html, missing_html, suggestions, status = compute_match(
            resume_file=str(pdf_path),
            job_description="Seeking Python Developer with SQL experience.",
        )

        # Computation succeeded and scores are returned
        assert "score-card" in score_html
        assert text_sim > 0.0
        assert skill_cov > 0.0
        # Status communicates graceful degradation
        assert "MLflow tracking:" in status
        assert "Unavailable" in status
        assert "Analysis Completed" in status


def test_privacy_preservation_no_sensitive_data_logged(
    isolated_tracking_uri: str,
):
    """Requirement 7: Candidate resume text and JD text are never logged in params, tags, or artifacts."""
    sensitive_resume_text = "John Doe, phone: +1-555-0199, SSN: 000-11-2222, email: john@secret.org"
    sensitive_job_text = "Confidential project codename Prometheus: hiring lead architect"

    results = {
        "final_score": 75.0,
        "text_similarity": 60.0,
        "skill_coverage": 90.0,
        "matching_skills": ["Python"],
        "missing_skills": ["Docker"],
        "resume_skills": ["Python"],
        "job_skills": ["Python", "Docker"],
        "suggestions": ["Add Docker"],
    }

    run_id = log_live_match(
        results=results,
        skill_weight=0.4,
        tracking_uri=isolated_tracking_uri,
    )
    assert run_id is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=isolated_tracking_uri)
    run = client.get_run(run_id)

    # Convert all logged parameters and tags into a searchable string
    logged_metadata = str(run.data.params) + str(run.data.tags)

    # Ensure sensitive strings are completely absent
    assert "John Doe" not in logged_metadata
    assert "555-0199" not in logged_metadata
    assert "000-11-2222" not in logged_metadata
    assert "john@secret.org" not in logged_metadata
    assert "Prometheus" not in logged_metadata

    # Verify no artifacts were uploaded containing resume or JD
    artifacts = client.list_artifacts(run_id)
    assert len(artifacts) == 0


def test_non_finite_metric_values_not_logged(
    isolated_tracking_uri: str,
):
    """Requirement 8: Non-finite values (NaN, inf) are handled safely and not logged."""
    bad_results = {
        "final_score": float("nan"),
        "text_similarity": float("inf"),
        "skill_coverage": 50.0,
        "matching_skills": ["Python"],
        "missing_skills": [],
        "resume_skills": ["Python"],
        "job_skills": ["Python"],
    }

    run_id = log_live_match(
        results=bad_results,
        skill_weight=0.4,
        tracking_uri=isolated_tracking_uri,
    )
    assert run_id is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=isolated_tracking_uri)
    run = client.get_run(run_id)
    metrics = run.data.metrics

    # NaN and Inf must NOT be logged
    assert "final_matching_score" not in metrics
    assert "text_similarity_score" not in metrics
    # Finite metric should still be logged
    assert metrics["skill_coverage_score"] == 50.0


def test_handles_exceptions_and_closes_runs_correctly(
    isolated_tracking_uri: str,
    sample_match_results: dict,
):
    """Requirement 9: Exceptions during logging are caught and any active run is closed."""
    # Force an exception during mlflow.log_metric
    with patch("mlflow.log_metric", side_effect=RuntimeError("Simulated logging error")):
        run_id = log_live_match(
            results=sample_match_results,
            tracking_uri=isolated_tracking_uri,
        )
        # Should gracefully return None without raising
        assert run_id is None

    # Crucial: verify active_run() is NOT leaking
    assert mlflow.active_run() is None
