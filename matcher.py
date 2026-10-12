"""
matcher.py
==========
Machine Learning and Natural Language Processing matching engine for comparing
resumes with job descriptions.

Key Features:
- Text normalization and cleaning
- TF-IDF vectorization with cosine similarity calculation
- Exact and boundary-aware skill extraction
- Configurable weighted combination of text similarity and skill coverage
- Actionable suggestion generation for resume improvement
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default list of recognized technical skills as required by project specification
DEFAULT_SKILLS: list[str] = [
    "Python",
    "Java",
    "SQL",
    "pandas",
    "NumPy",
    "Flask",
    "Django",
    "React",
    "JavaScript",
    "HTML",
    "CSS",
    "Docker",
    "AWS",
    "Git",
    "Machine Learning",
    "scikit-learn",
    "FastAPI",
    "Excel",
    "Power BI",
]

# Aliases and regex patterns for skill normalization to ensure high precision
SKILL_PATTERNS: dict[str, str] = {
    "Python": r"\bpython\b",
    "Java": r"\bjava\b(?!\s*script)",  # Avoid matching JavaScript as Java
    "SQL": r"\bsql\b",
    "pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "Flask": r"\bflask\b",
    "Django": r"\bdjango\b",
    "React": r"\breact(?:\.js|js)?\b",
    "JavaScript": r"\b(?:javascript|js)\b",
    "HTML": r"\bhtml(?:5)?\b",
    "CSS": r"\bcss(?:3)?\b",
    "Docker": r"\bdocker\b",
    "AWS": r"\b(?:aws|amazon web services)\b",
    "Git": r"\bgit\b(?!\s*(?:hub|lab))|\bgithub\b|\bgitlab\b",
    "Machine Learning": r"\b(?:machine learning|ml)\b",
    "scikit-learn": r"\b(?:scikit-learn|sklearn)\b",
    "FastAPI": r"\bfastapi\b",
    "Excel": r"\b(?:excel|ms excel|microsoft excel)\b",
    "Power BI": r"\b(?:power bi|powerbi)\b",
}


def clean_text(text: str) -> str:
    """
    Clean and normalize input text:
    1. Convert to lowercase.
    2. Replace punctuation with spaces while preserving word boundaries.
    3. Collapse consecutive whitespaces into a single space.
    4. Strip leading and trailing whitespace.

    Args:
        text: Raw text string.

    Returns:
        Cleaned, normalized text string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Replace punctuation characters with space, preserving alphanumeric
    # Keep hyphens between letters for terms like scikit-learn
    text = re.sub(r"[^a-z0-9\s\-+]", " ", text)
    # Replace multiple hyphens or isolated hyphens with space
    text = re.sub(r"(?<![a-z0-9])\-+|\-+(?![a-z0-9])", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_text_similarity(
    resume_text: str,
    job_text: str,
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: str | list[str] | None = "english",
) -> float:
    """
    Calculate TF-IDF Cosine Similarity between resume text and job description text.

    Args:
        resume_text: Raw or cleaned resume text.
        job_text: Raw or cleaned job description text.
        ngram_range: Lower and upper boundary of n-values for n-grams.
        stop_words: Stop words list or 'english'.

    Returns:
        Similarity score as a percentage between 0.0 and 100.0.
    """
    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_text)

    # If either text has no content, similarity is 0.0
    if not cleaned_resume or not cleaned_job:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words,
        )
        tfidf_matrix = vectorizer.fit_transform([cleaned_resume, cleaned_job])

        # If vectorizer produces an empty vocabulary (e.g., only stop words)
        if tfidf_matrix.shape[1] == 0:
            return 0.0

        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        score = float(similarity_matrix[0][0])
        # Bound score between 0.0 and 1.0, convert to percentage
        score = max(0.0, min(1.0, score))
        return round(score * 100.0, 2)
    except ValueError:
        # Occurs if all words are stop words or vocabulary is empty
        return 0.0


def extract_skills(text: str, skill_list: Sequence[str] | None = None) -> set[str]:
    """
    Identify which recognized skills appear in the given text.
    Uses regex patterns for known skills to avoid false positives (e.g., 'Java' in 'JavaScript').

    Args:
        text: Input text (resume or job description).
        skill_list: Optional custom list of skill names. If None, defaults to DEFAULT_SKILLS.

    Returns:
        Set of detected canonical skill names.
    """
    if not text or not isinstance(text, str):
        return set()

    normalized_text = text.lower()
    target_skills = list(skill_list) if skill_list is not None else DEFAULT_SKILLS
    found_skills: set[str] = set()

    for skill in target_skills:
        pattern = SKILL_PATTERNS.get(skill)
        if pattern:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                found_skills.add(skill)
        else:
            # Fallback regex with word boundary for skills not in pattern dictionary
            escaped_skill = re.escape(skill.lower())
            custom_pattern = rf"\b{escaped_skill}\b"
            if re.search(custom_pattern, normalized_text, re.IGNORECASE):
                found_skills.add(skill)

    return found_skills


def calculate_skill_coverage(
    resume_skills: set[str] | Iterable[str],
    job_skills: set[str] | Iterable[str],
) -> tuple[float, list[str], list[str]]:
    """
    Calculate skill coverage based on the job-description skills present in the resume.
    Handles division-by-zero safely if no recognized skills are present in the job description.

    Args:
        resume_skills: Set or list of skills detected in resume.
        job_skills: Set or list of skills detected in job description.

    Returns:
        tuple of:
        - coverage_score (0.0 to 100.0)
        - matching_skills (sorted list)
        - missing_skills (sorted list)
    """
    r_skills = set(resume_skills)
    j_skills = set(job_skills)

    matching = sorted(list(j_skills.intersection(r_skills)))
    missing = sorted(list(j_skills.difference(r_skills)))

    if len(j_skills) == 0:
        # Zero-division safety: if JD has no recognized skills, coverage is 0.0
        return 0.0, matching, missing

    coverage = (len(matching) / len(j_skills)) * 100.0
    return round(coverage, 2), matching, missing


def calculate_final_score(
    text_similarity: float,
    skill_coverage: float,
    skill_weight: float = 0.4,
) -> float:
    """
    Combine text similarity and skill coverage into a weighted matching score.
    Formula:
        Final Score = (1 - skill_weight) * Text Similarity + skill_weight * Skill Coverage

    Args:
        text_similarity: Score between 0.0 and 100.0.
        skill_coverage: Score between 0.0 and 100.0.
        skill_weight: Float between 0.0 and 1.0 (default 0.4).

    Returns:
        Final combined score between 0.0 and 100.0.

    Raises:
        ValueError: If skill_weight is outside [0.0, 1.0].
    """
    if not (0.0 <= skill_weight <= 1.0):
        raise ValueError(
            f"skill_weight must be between 0.0 and 1.0, got: {skill_weight}"
        )

    score = ((1.0 - skill_weight) * text_similarity) + (skill_weight * skill_coverage)
    score = max(0.0, min(100.0, score))
    return round(score, 2)


def generate_suggestions(
    final_score: float,
    text_similarity: float,
    skill_coverage: float,
    matching_skills: list[str],
    missing_skills: list[str],
    resume_skills: list[str] | None = None,
    job_skills: list[str] | None = None,
) -> list[str]:
    """
    Generate practical recommendations to improve resume alignment with the job description.

    Args:
        final_score: Overall matching score.
        text_similarity: TF-IDF similarity score.
        skill_coverage: Skill coverage score.
        matching_skills: List of skills found in both.
        missing_skills: List of skills requested in JD but missing in resume.
        resume_skills: All skills detected in resume.
        job_skills: All skills detected in JD.

    Returns:
        List of actionable suggestion strings.
    """
    suggestions: list[str] = []

    # 1. Missing skills advice
    if missing_skills:
        missing_formatted = ", ".join(missing_skills)
        suggestions.append(
            f"Add Missing Skills: The job description emphasizes {missing_formatted}. "
            "If you possess hands-on experience or academic projects using these tools, "
            "explicitly feature them in your Skills and Experience sections."
        )

    # 2. Skill coverage advice
    if skill_coverage < 50.0 and job_skills:
        suggestions.append(
            "Low Technical Skill Coverage: Your resume covers less than 50% of the recognized technical skills "
            "demanded by the role. Consider highlighting relevant transferrable technologies or framing past "
            "responsibilities around the required tools."
        )
    elif skill_coverage >= 80.0:
        suggestions.append(
            "Strong Skill Match: Your technical skill alignment is solid. Ensure each matching skill "
            f"({', '.join(matching_skills[:5])}{'...' if len(matching_skills) > 5 else ''}) "
            "is backed by concrete project outcomes or quantifiable metrics (e.g., 'reduced latency by 25%')."
        )

    # 3. Text similarity & phrasing advice
    if text_similarity < 40.0:
        suggestions.append(
            "Align Terminology and Keywords: The overall textual similarity is relatively low. "
            "Mirror the role's language, industry terms, and responsibilities in your bullet points "
            "to enhance Applicant Tracking System (ATS) readability."
        )
    elif text_similarity >= 70.0:
        suggestions.append(
            "Good Contextual Alignment: Your resume vocabulary naturally mirrors the job description."
        )

    # 4. Overall score recommendations
    if final_score < 40.0:
        suggestions.append(
            "Customization Needed: Your resume currently shows low alignment with this specific profile. "
            "Consider tailoring your summary, project descriptions, and key achievements specifically to this opportunity."
        )
    elif final_score >= 75.0:
        suggestions.append(
            "High Overall Alignment: Your profile aligns well with this posting. Review formatting, proofread "
            "for grammar, and ensure your contact details and portfolio links are up to date."
        )

    # If no recognized skills were detected in the JD
    if not job_skills:
        suggestions.append(
            "General Job Description: The provided job description contains few standard technical keywords. "
            "Check for domain-specific or niche requirements and weave them into your experience."
        )

    return suggestions


def match_resume_to_job(
    resume_text: str,
    job_text: str,
    skill_weight: float = 0.4,
    ngram_range: tuple[int, int] = (1, 1),
    skill_list: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Complete end-to-end matching pipeline between a resume and a job description.

    Args:
        resume_text: Extracted text from resume.
        job_text: Raw job description text.
        skill_weight: Importance of skill coverage relative to text similarity (0.0 to 1.0).
        ngram_range: TF-IDF n-gram range (e.g., (1, 1) or (1, 2)).
        skill_list: Configurable list of skills to recognize.

    Returns:
        dict containing:
        - final_score: float (0.0 - 100.0)
        - text_similarity: float (0.0 - 100.0)
        - skill_coverage: float (0.0 - 100.0)
        - matching_skills: list[str]
        - missing_skills: list[str]
        - resume_skills: list[str]
        - job_skills: list[str]
        - suggestions: list[str]
        - disclaimer: str
    """
    # Input validation
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text is empty or invalid.")
    if not isinstance(job_text, str) or not job_text.strip():
        raise ValueError("Job description text is empty or invalid.")

    # 1. Text Similarity
    text_sim = calculate_text_similarity(
        resume_text=resume_text,
        job_text=job_text,
        ngram_range=ngram_range,
    )

    # 2. Skill Extraction
    r_skills = extract_skills(resume_text, skill_list=skill_list)
    j_skills = extract_skills(job_text, skill_list=skill_list)

    # 3. Skill Coverage
    skill_cov, matching, missing = calculate_skill_coverage(r_skills, j_skills)

    # 4. Final Score
    final_score = calculate_final_score(
        text_similarity=text_sim,
        skill_coverage=skill_cov,
        skill_weight=skill_weight,
    )

    # 5. Suggestions
    suggestions = generate_suggestions(
        final_score=final_score,
        text_similarity=text_sim,
        skill_coverage=skill_cov,
        matching_skills=matching,
        missing_skills=missing,
        resume_skills=sorted(list(r_skills)),
        job_skills=sorted(list(j_skills)),
    )

    return {
        "final_score": final_score,
        "text_similarity": text_sim,
        "skill_coverage": skill_cov,
        "matching_skills": matching,
        "missing_skills": missing,
        "resume_skills": sorted(list(r_skills)),
        "job_skills": sorted(list(j_skills)),
        "suggestions": suggestions,
        "disclaimer": (
            "This matching score measures textual and skill alignment, "
            "not the probability of getting hired."
        ),
    }
