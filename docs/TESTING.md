# Automated Testing Report & Test Suite Specification

## 1. Test Suite Architecture

The automated test suite is implemented using **pytest** and organized into four modular test files located in the `tests/` directory:

- `tests/test_matcher.py`: Validates text normalization, TF-IDF cosine similarity, regex-based skill extraction, weighted score math, boundary constraints, and suggestion generation (15 tests).
- `tests/test_pdf_utils.py`: Validates PyMuPDF extraction across valid PDFs, in-memory bytes, streams, zero-byte files, corrupted files, and blank/image-only PDFs (8 tests).
- `tests/test_app.py`: Validates the Gradio user interface logic, input validation errors (missing PDF, missing job description), corrupt file handling in UI, and full end-to-end component outputs (6 tests).
- `tests/test_mlflow_tracking.py`: Validates live MLflow tracking creation, metric recording, parameter & tag logging, distinct run IDs, input validation protection, graceful failure degradation, privacy preservation, and exception handling (9 tests).

---

## 2. Test Execution Command

To execute all automated tests with verbose output:

```bash
pytest -v
```

---

## 3. Actual Test Execution Results

Executed in Python 3.11 environment on Windows:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\raghu\Videos\CPPE Project\resume-job-matcher
plugins: anyio-4.14.2
collected 38 items

tests/test_app.py::test_app_build PASSED                                 [  2%]
tests/test_app.py::test_compute_match_missing_resume PASSED              [  5%]
tests/test_app.py::test_compute_match_missing_job_description PASSED     [  7%]
tests/test_app.py::test_compute_match_successful PASSED                  [ 10%]
tests/test_app.py::test_compute_match_corrupt_pdf PASSED                 [ 13%]
tests/test_app.py::test_load_sample_content PASSED                       [ 15%]
tests/test_matcher.py::test_clean_text_basic PASSED                      [ 18%]
tests/test_matcher.py::test_clean_text_empty_and_non_string PASSED       [ 21%]
tests/test_matcher.py::test_calculate_text_similarity_identical PASSED   [ 23%]
tests/test_matcher.py::test_calculate_text_similarity_completely_different PASSED [ 26%]
tests/test_matcher.py::test_calculate_text_similarity_empty PASSED       [ 28%]
tests/test_matcher.py::test_extract_skills_precision PASSED              [ 31%]
tests/test_matcher.py::test_extract_skills_multi_word PASSED             [ 34%]
tests/test_matcher.py::test_calculate_skill_coverage_full_match PASSED   [ 36%]
tests/test_matcher.py::test_calculate_skill_coverage_partial_and_missing PASSED [ 39%]
tests/test_matcher.py::test_calculate_skill_coverage_no_recognized_skills PASSED [ 42%]
tests/test_matcher.py::test_calculate_final_score_weighting PASSED       [ 44%]
tests/test_matcher.py::test_calculate_final_score_invalid_weights PASSED [ 47%]
tests/test_matcher.py::test_match_resume_to_job_normal PASSED            [ 50%]
tests/test_matcher.py::test_match_resume_to_job_empty_inputs PASSED      [ 52%]
tests/test_matcher.py::test_generate_suggestions_content PASSED          [ 55%]
tests/test_mlflow_tracking.py::test_successful_calculation_creates_one_mlflow_run PASSED [ 57%]
tests/test_mlflow_tracking.py::test_successful_run_records_expected_metrics PASSED [ 60%]
tests/test_mlflow_tracking.py::test_successful_run_records_expected_parameters_and_tags PASSED [ 63%]
tests/test_mlflow_tracking.py::test_multiple_successful_calculations_create_distinct_runs PASSED [ 65%]
tests/test_mlflow_tracking.py::test_invalid_inputs_do_not_create_live_runs PASSED [ 68%]
tests/test_mlflow_tracking.py::test_matching_results_remain_available_if_tracking_fails PASSED [ 71%]
tests/test_mlflow_tracking.py::test_privacy_preservation_no_sensitive_data_logged PASSED [ 73%]
tests/test_mlflow_tracking.py::test_non_finite_metric_values_not_logged PASSED [ 76%]
tests/test_mlflow_tracking.py::test_handles_exceptions_and_closes_runs_correctly PASSED [ 78%]
tests/test_pdf_utils.py::test_extract_text_valid_file PASSED             [ 81%]
tests/test_pdf_utils.py::test_extract_text_valid_bytes PASSED            [ 84%]
tests/test_pdf_utils.py::test_extract_text_valid_stream PASSED           [ 86%]
tests/test_pdf_utils.py::test_extract_text_none_input PASSED             [ 89%]
tests/test_pdf_utils.py::test_extract_text_nonexistent_file PASSED       [ 92%]
tests/test_pdf_utils.py::test_extract_text_empty_file PASSED             [ 94%]
tests/test_pdf_utils.py::test_extract_text_corrupted_file PASSED         [ 97%]
tests/test_pdf_utils.py::test_extract_text_blank_pdf PASSED              [100%]

============================= 38 passed in 22.57s =============================
```

**Status:** `38/38 PASSED` (100% Pass Rate).tests/test_matcher.py::test_generate_suggestions_content PASSED          [ 72%]
tests/test_pdf_utils.py::test_extract_text_valid_file PASSED             [ 75%]
tests/test_pdf_utils.py::test_extract_text_valid_bytes PASSED            [ 79%]
tests/test_pdf_utils.py::test_extract_text_valid_stream PASSED           [ 82%]
tests/test_pdf_utils.py::test_extract_text_none_input PASSED             [ 86%]
tests/test_pdf_utils.py::test_extract_text_nonexistent_file PASSED       [ 89%]
tests/test_pdf_utils.py::test_extract_text_empty_file PASSED             [ 93%]
tests/test_pdf_utils.py::test_extract_text_corrupted_file PASSED         [ 96%]
tests/test_pdf_utils.py::test_extract_text_blank_pdf PASSED              [100%]

============================= 29 passed in 9.10s ==============================
```

**Status:** `29/29 PASSED` (100% Pass Rate).

---

## 4. Test Case Catalog

### A. Matching Engine Tests (`tests/test_matcher.py`)

| Test Function | Target Behavior Verified | Result |
| :--- | :--- | :---: |
| `test_clean_text_basic` | Lowercasing, punctuation stripping, preserving hyphens in terms like `machine-learning`. | `PASSED` |
| `test_clean_text_empty_and_non_string` | Safe return of empty string on `""`, whitespace, and non-string inputs. | `PASSED` |
| `test_calculate_text_similarity_identical` | Cosine similarity of identical text approaches ~100.0%. | `PASSED` |
| `test_calculate_text_similarity_completely_different` | Unrelated domains yield low cosine similarity (< 10%). | `PASSED` |
| `test_calculate_text_similarity_empty` | Empty or whitespace texts yield exactly 0.0% similarity. | `PASSED` |
| `test_extract_skills_precision` | Boundary precision: matches JavaScript without false-triggering Java. | `PASSED` |
| `test_extract_skills_multi_word` | Multi-word skill patterns (Machine Learning, Power BI) are recognized. | `PASSED` |
| `test_calculate_skill_coverage_full_match` | When resume has all JD skills, coverage equals 100.0% and missing list is empty. | `PASSED` |
| `test_calculate_skill_coverage_partial_and_missing` | Correct subset calculation and missing skills enumeration. | `PASSED` |
| `test_calculate_skill_coverage_no_recognized_skills` | Division-by-zero protection: zero JD skills returns 0.0% coverage safely. | `PASSED` |
| `test_calculate_final_score_weighting` | Verification of $(1 - \lambda)\times \text{Text} + \lambda \times \text{Skill}$ math. | `PASSED` |
| `test_calculate_final_score_invalid_weights` | Raises `ValueError` for weights $< 0.0$ or $> 1.0$. | `PASSED` |
| `test_match_resume_to_job_normal` | Full end-to-end matching produces valid score, gauges, and disclaimer. | `PASSED` |
| `test_match_resume_to_job_empty_inputs` | Empty resume or JD raises `ValueError` with clear message. | `PASSED` |
| `test_generate_suggestions_content` | Dynamic suggestions explicitly list missing skills. | `PASSED` |

### B. PDF Processing Tests (`tests/test_pdf_utils.py`)

| Test Function | Target Behavior Verified | Result |
| :--- | :--- | :---: |
| `test_extract_text_valid_file` | Extracts plain text from valid PDF file path. | `PASSED` |
| `test_extract_text_valid_bytes` | Extracts text from raw PDF `bytes`. | `PASSED` |
| `test_extract_text_valid_stream` | Extracts text from an in-memory `io.BytesIO` binary stream. | `PASSED` |
| `test_extract_text_none_input` | Raises `ValueError` if input is `None`. | `PASSED` |
| `test_extract_text_nonexistent_file` | Raises `ValueError` if file path does not exist. | `PASSED` |
| `test_extract_text_empty_file` | Raises `PDFProcessingError` for 0-byte PDF files. | `PASSED` |
| `test_extract_text_corrupted_file` | Raises `PDFProcessingError` for non-PDF corrupted bytes. | `PASSED` |
| `test_extract_text_blank_pdf` | Raises `PDFProcessingError` explaining OCR requirement for blank/image PDFs. | `PASSED` |

### C. Application UI Tests (`tests/test_app.py`)

| Test Function | Target Behavior Verified | Result |
| :--- | :--- | :---: |
| `test_app_build` | Gradio `gr.Blocks` builds cleanly without layout errors. | `PASSED` |
| `test_compute_match_missing_resume` | UI displays friendly error banner when resume is not uploaded. | `PASSED` |
| `test_compute_match_missing_job_description` | UI displays friendly error banner when JD text is omitted. | `PASSED` |
| `test_compute_match_successful` | Returns complete score card HTML, numbers, badges, and advice. | `PASSED` |
| `test_compute_match_corrupt_pdf` | Displays PDF processing error banner without server crashes. | `PASSED` |
| `test_load_sample_content` | Verifies sample content loader loads bundled example data. | `PASSED` |

### D. Live MLflow Tracking Tests (`tests/test_mlflow_tracking.py`)

| Test Function | Target Behavior Verified | Result |
| :--- | :--- | :---: |
| `test_successful_calculation_creates_one_mlflow_run` | A valid matching calculation creates exactly one MLflow run with a valid run ID. | `PASSED` |
| `test_successful_run_records_expected_metrics` | Verifies that all 7 numerical metrics are correctly logged on a 0–100 scale. | `PASSED` |
| `test_successful_run_records_expected_parameters_and_tags` | Verifies non-sensitive parameters and execution tags (`source`, `run_type`, timestamp). | `PASSED` |
| `test_multiple_successful_calculations_create_distinct_runs` | Multiple calculations generate unique run IDs and do not reuse active runs. | `PASSED` |
| `test_invalid_inputs_do_not_create_live_runs` | Validation failures (empty resume or JD) do not trigger MLflow logging. | `PASSED` |
| `test_matching_results_remain_available_if_tracking_fails` | Simulates MLflow failure and verifies graceful UI degradation without calculation interruption. | `PASSED` |
| `test_privacy_preservation_no_sensitive_data_logged` | Ensures candidate PII, resume text, and JD text are absent from params, tags, and artifacts. | `PASSED` |
| `test_non_finite_metric_values_not_logged` | Non-finite values (NaN, inf) are handled safely and excluded from logged metrics. | `PASSED` |
| `test_handles_exceptions_and_closes_runs_correctly` | Exception handling ensures active runs are cleanly closed without leaks. | `PASSED` |
