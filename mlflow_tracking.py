"""
mlflow_tracking.py
==================
Reusable live tracking module for the Resume-Job Description Matcher.
Logs metrics, parameters, and metadata for live Gradio matching calculations
to the 'Resume-Job-Description-Matcher-Live' MLflow experiment.

Privacy Guarantees:
- Never logs resume text, candidate names, or contact details.
- Never logs job description text.
- Never creates artifacts containing sensitive or PII data.
- Only logs numerical scores, skill counts, configuration, and execution metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any

# Ensure MLflow file store flags are active
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

try:
    import mlflow
except ImportError:
    mlflow = None

LIVE_EXPERIMENT_NAME = "Resume-Job-Description-Matcher-Live"
DEFAULT_TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"
APP_VERSION = "1.0.0"


def get_tracking_uri() -> str:
    """
    Resolve the MLflow tracking URI.
    Uses MLFLOW_TRACKING_URI environment variable if set, otherwise defaults
    to the local project 'mlruns' directory.
    """
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri and env_uri.strip():
        return env_uri.strip()
    return DEFAULT_TRACKING_DIR.as_uri()


def log_live_match(
    results: dict[str, Any],
    skill_weight: float = 0.4,
    ngram_range: tuple[int, int] = (1, 2),
    tracking_uri: str | None = None,
    experiment_name: str = LIVE_EXPERIMENT_NAME,
) -> str | None:
    """
    Log non-sensitive results and configuration of a successful matching calculation
    as an individual MLflow run.

    Args:
        results: Dictionary returned by match_resume_to_job containing final_score,
                 text_similarity, skill_coverage, matching_skills, missing_skills,
                 resume_skills, job_skills.
        skill_weight: The weighting factor applied between text similarity and skill coverage.
        ngram_range: The TF-IDF n-gram range used.
        tracking_uri: Optional override for tracking URI (defaults to get_tracking_uri()).
        experiment_name: Experiment name (defaults to LIVE_EXPERIMENT_NAME).

    Returns:
        The actual MLflow run ID string if tracking succeeded, or None if tracking failed.
    """
    if mlflow is None or not isinstance(results, dict):
        return None

    # Resolve tracking URI and configure MLflow
    target_uri = tracking_uri or get_tracking_uri()

    try:
        mlflow.set_tracking_uri(target_uri)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        print(f"[MLflow Tracking Warning] Failed to configure tracking URI/experiment: {exc}")
        return None

    # Safely close any active run from previous interruptions in the current thread/process
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass

    run_id: str | None = None

    try:
        with mlflow.start_run(run_name="live_match_prediction") as run:
            run_id = run.info.run_id

            # 1. Log Non-Sensitive Parameters
            params = {
                "matching_algorithm": "TF-IDF + Cosine Similarity & Weighted Skill Extraction",
                "ngram_range": str(ngram_range),
                "skill_weight": float(skill_weight),
                "pdf_parser": "PyMuPDF (fitz)",
                "app_version": APP_VERSION,
                "input_processing": "lowercase=True, clean_punctuation=True, stop_words=english",
                "scoring_formula": "(1 - skill_weight) * text_similarity + skill_weight * skill_coverage",
            }
            for param_key, param_val in params.items():
                mlflow.log_param(param_key, param_val)

            # 2. Extract and Validate Numerical Metrics
            raw_metrics = {
                "final_matching_score": results.get("final_score"),
                "text_similarity_score": results.get("text_similarity"),
                "skill_coverage_score": results.get("skill_coverage"),
                "matched_skill_count": len(results.get("matching_skills", [])),
                "missing_skill_count": len(results.get("missing_skills", [])),
                "job_description_skill_count": len(results.get("job_skills", [])),
                "resume_skill_count": len(results.get("resume_skills", [])),
            }

            # Filter out non-finite (NaN, inf) values safely
            valid_metrics = {}
            for metric_key, metric_val in raw_metrics.items():
                if metric_val is not None:
                    try:
                        num_val = float(metric_val)
                        if math.isfinite(num_val):
                            valid_metrics[metric_key] = num_val
                    except (ValueError, TypeError):
                        pass

            for metric_key, metric_val in valid_metrics.items():
                mlflow.log_metric(metric_key, metric_val)

            # 3. Log Safe Metadata Tags
            tags = {
                "source": "gradio_app",
                "run_type": "live_prediction",
                "application": "resume_job_matcher",
                "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for tag_key, tag_val in tags.items():
                mlflow.set_tag(tag_key, tag_val)

            return run_id

    except Exception as exc:
        print(f"[MLflow Tracking Warning] Failed to log live run: {exc}")
        return None
    finally:
        # Guarantee run termination even if unexpected exceptions arise
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception:
            pass
