"""
tests/test_matcher.py
=====================
Unit tests for the ML/NLP matching engine in matcher.py.
"""

from __future__ import annotations

import pytest
from matcher import (
    DEFAULT_SKILLS,
    calculate_final_score,
    calculate_skill_coverage,
    calculate_text_similarity,
    clean_text,
    extract_skills,
    generate_suggestions,
    match_resume_to_job,
)


def test_clean_text_basic():
    """Test standard text cleaning and lowercasing."""
    raw = "  Hello, World! Python & Machine-Learning... \n\t 123 "
    cleaned = clean_text(raw)
    assert cleaned == "hello world python machine-learning 123"


def test_clean_text_empty_and_non_string():
    """Test cleaning on empty or non-string inputs."""
    assert clean_text("") == ""
    assert clean_text("     ") == ""
    assert clean_text(None) == ""  # type: ignore


def test_calculate_text_similarity_identical():
    """Identical texts should yield high similarity close to 100%."""
    sample = "Software engineer proficient in Python, SQL, Docker, and AWS microservices."
    sim = calculate_text_similarity(sample, sample)
    assert 99.0 <= sim <= 100.0


def test_calculate_text_similarity_completely_different():
    """Completely disjoint texts should have low similarity."""
    text_a = "Cooking delicious Italian pizza and pasta with tomatoes."
    text_b = "Quantum mechanics and gravitational particle physics."
    sim = calculate_text_similarity(text_a, text_b)
    assert sim < 10.0


def test_calculate_text_similarity_empty():
    """Empty strings should return 0.0 similarity."""
    assert calculate_text_similarity("", "Python developer") == 0.0
    assert calculate_text_similarity("Python developer", "") == 0.0
    assert calculate_text_similarity("", "") == 0.0


def test_extract_skills_precision():
    """
    Test that skills are extracted accurately.
    Crucially: Java should not trigger from JavaScript.
    """
    sample_text = (
        "Proficient in JavaScript, React, Python, and SQL. "
        "Built models with scikit-learn and pandas. Deployed via Docker to AWS."
    )
    skills = extract_skills(sample_text)

    # Should match these
    assert "JavaScript" in skills
    assert "React" in skills
    assert "Python" in skills
    assert "SQL" in skills
    assert "scikit-learn" in skills
    assert "pandas" in skills
    assert "Docker" in skills
    assert "AWS" in skills

    # Should NOT match Java (since only JavaScript was mentioned)
    assert "Java" not in skills


def test_extract_skills_multi_word():
    """Verify multi-word skill extraction like Machine Learning and Power BI."""
    sample_text = "Experience with Machine Learning algorithms and Power BI reporting."
    skills = extract_skills(sample_text)
    assert "Machine Learning" in skills
    assert "Power BI" in skills


def test_calculate_skill_coverage_full_match():
    """When all job skills are in the resume, coverage must be 100%."""
    resume_skills = {"Python", "SQL", "Docker", "AWS", "Git"}
    job_skills = {"Python", "SQL", "Docker"}
    coverage, matching, missing = calculate_skill_coverage(resume_skills, job_skills)

    assert coverage == 100.0
    assert matching == ["Docker", "Python", "SQL"]
    assert missing == []


def test_calculate_skill_coverage_partial_and_missing():
    """Verify calculation with some missing skills."""
    resume_skills = {"Python", "Flask"}
    job_skills = {"Python", "Django", "React", "Docker"}
    coverage, matching, missing = calculate_skill_coverage(resume_skills, job_skills)

    assert coverage == 25.0  # 1 out of 4 = 25%
    assert matching == ["Python"]
    assert missing == ["Django", "Docker", "React"]


def test_calculate_skill_coverage_no_recognized_skills():
    """Zero-division test: JD with zero recognized skills must return 0.0 without errors."""
    resume_skills = {"Python", "SQL"}
    job_skills = set()
    coverage, matching, missing = calculate_skill_coverage(resume_skills, job_skills)

    assert coverage == 0.0
    assert matching == []
    assert missing == []


def test_calculate_final_score_weighting():
    """Test the weighted formula: (1 - w)*Text + w*Skill."""
    # (1 - 0.4)*80 + 0.4*50 = 0.6*80 + 20 = 48 + 20 = 68.0
    score = calculate_final_score(text_similarity=80.0, skill_coverage=50.0, skill_weight=0.4)
    assert score == 68.0


def test_calculate_final_score_invalid_weights():
    """Weights outside [0.0, 1.0] must raise ValueError."""
    with pytest.raises(ValueError, match="skill_weight must be between 0.0 and 1.0"):
        calculate_final_score(50.0, 50.0, skill_weight=-0.1)

    with pytest.raises(ValueError, match="skill_weight must be between 0.0 and 1.0"):
        calculate_final_score(50.0, 50.0, skill_weight=1.2)


def test_match_resume_to_job_normal():
    """End-to-end matching test with normal valid inputs."""
    resume = (
        "Experienced Python software engineer. Skilled in FastAPI, Docker, SQL, and AWS. "
        "Developed cloud backend microservices using Git workflows."
    )
    job = (
        "Seeking a Python Developer with hands-on expertise in FastAPI, SQL, Docker, and AWS. "
        "Familiarity with Git and cloud architectures is required."
    )

    res = match_resume_to_job(resume, job, skill_weight=0.4)

    assert 0.0 <= res["final_score"] <= 100.0
    assert 0.0 <= res["text_similarity"] <= 100.0
    assert res["skill_coverage"] == 100.0  # All job skills matched
    assert "FastAPI" in res["matching_skills"]
    assert "Docker" in res["matching_skills"]
    assert len(res["missing_skills"]) == 0
    assert len(res["suggestions"]) > 0
    assert "probability of getting hired" in res["disclaimer"]


def test_match_resume_to_job_empty_inputs():
    """Validation test: empty resume or job description must raise ValueError."""
    with pytest.raises(ValueError, match="Resume text is empty or invalid"):
        match_resume_to_job("", "Valid job description")

    with pytest.raises(ValueError, match="Job description text is empty or invalid"):
        match_resume_to_job("Valid resume text", "   ")


def test_generate_suggestions_content():
    """Verify suggestions are helpful and reference missing skills."""
    suggestions = generate_suggestions(
        final_score=45.0,
        text_similarity=30.0,
        skill_coverage=20.0,
        matching_skills=["Python"],
        missing_skills=["Docker", "AWS", "FastAPI"],
        resume_skills=["Python"],
        job_skills=["Python", "Docker", "AWS", "FastAPI"],
    )
    combined = " ".join(suggestions)
    assert "Docker" in combined
    assert "AWS" in combined
    assert "FastAPI" in combined
