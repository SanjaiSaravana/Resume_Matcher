---
title: Resume Job Description Matcher
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
---

# Resume–Job Description Matcher

An end-to-end Machine Learning web application that calculates the alignment between a candidate's PDF resume and a target job description. The system combines **TF-IDF Vectorization**, **Cosine Similarity**, and a **Boundary-Aware Technical Skill Extraction Engine** to calculate an interpretable match score (0–100), identify matching and missing skills, and provide actionable recommendations.

Developed with experiment tracking via **MLflow**, automated unit/integration tests with **pytest**, and configured for direct deployment on **Hugging Face Spaces** with **Gradio**.

> **Important Disclosure:** This system evaluates textual similarity and technical keyword correspondence. It **does not predict hiring probability**, candidate culture fit, or overall job competence.

---

## 🌟 Key Features

- **PDF Resume Ingestion:** Robust PDF text extraction using PyMuPDF (`fitz`), handling empty pages, non-extractable text, and corrupted files gracefully.
- **TF-IDF & Cosine Similarity:** Normalizes text, computes n-gram term frequencies, and evaluates vector cosine similarity.
- **Configurable Technical Skill Extraction:** Recognizes core industry technologies (Python, Java, SQL, pandas, NumPy, Flask, Django, React, JavaScript, HTML, CSS, Docker, AWS, Git, Machine Learning, scikit-learn, FastAPI, Excel, Power BI) using regex boundaries to avoid false positives (e.g., distinguishing Java from JavaScript).
- **Weighted Blended Scoring:**
  $$\text{Final Score} = (1 - \lambda) \times \text{Text Similarity} + \lambda \times \text{Skill Coverage}$$
  where $\lambda = \text{skill\_weight}$ (default $0.4$). Zero-division safe when no recognized skills appear in the job description.
- **Actionable Feedback:** Produces targeted suggestions for closing technical skill gaps and enhancing ATS readability.
- **Experiment Tracking with MLflow:** Pre-configured suite of 3 hyperparameter experiments with parameter, metric, and artifact logging.
- **Modern Interactive UI:** Responsive Gradio Blocks interface with score cards, color-coded badges, and 1-click example loading.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Language & Runtime** | Python 3.11 / 3.12 |
| **NLP & Machine Learning** | scikit-learn (`TfidfVectorizer`, `cosine_similarity`), NumPy, pandas, SciPy |
| **PDF Extraction** | PyMuPDF (`pymupdf`) |
| **Experiment Tracking** | MLflow |
| **Web Interface** | Gradio (Blocks API) |
| **Deployment Target** | Hugging Face Spaces (Gradio SDK, CPU Basic) |
| **Testing** | pytest |

---

## 📂 Project Structure

```text
resume-job-matcher/
├── app.py                      # Interactive Gradio web application
├── matcher.py                  # ML & NLP matching engine (TF-IDF, skills, scoring)
├── mlflow_tracking.py          # Privacy-preserving live MLflow tracking module
├── pdf_utils.py                # PyMuPDF PDF extraction and validation
├── experiments.py              # MLflow experiment pipeline (Runs A, B, C)
├── requirements.txt            # Runtime and testing dependencies
├── README.md                   # Project documentation & HF Spaces metadata
├── evaluation_pairs.csv        # Synthetic benchmark dataset (12 test pairs)
├── .gitignore                  # Git exclusions for venv, caches, and MLflow
├── tests/
│   ├── conftest.py             # Test harness environment setup
│   ├── test_matcher.py         # 15 tests for scoring, cleaning, and skills
│   ├── test_pdf_utils.py       # 8 tests for PDF extraction and edge cases
│   ├── test_app.py             # 6 tests for Gradio UI logic & input validation
│   └── test_mlflow_tracking.py # 9 tests for live MLflow tracking & privacy
├── sample_data/
│   ├── sample_resume.txt       # Sample fullstack / ML engineer resume text
│   ├── sample_job.txt          # Sample senior Python & ML engineer job description
│   └── sample_resume.pdf       # Ready-to-upload PDF resume for UI testing
└── docs/
    ├── MLflow_EXPERIMENTS.md   # Detailed MLflow experiment logs and UI guide
    ├── DEPLOYMENT.md           # Step-by-step local setup and HF Spaces guide
    └── TESTING.md              # Test catalog and actual 38/38 execution output
```

---

## 🚀 Quickstart: Local Setup

### 1. Clone & Setup Virtual Environment

```bash
git clone <repository-url>
cd resume-job-matcher

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Web Application

```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser. Click **Load Example** to populate the form instantly with the sample resume and job description.

---

## 🧪 Automated Testing

The project includes 38 automated pytest tests covering text normalization, skill boundary precision, zero-division safety, corrupted PDFs, blank pages, UI validation, and live MLflow tracking with privacy guarantees.

Run the test suite:
```bash
pytest -v
```

**Actual Test Result:** `38 passed in 22.57s (100% pass rate)` — see [docs/TESTING.md](docs/TESTING.md) for full catalog.

---

## 📊 MLflow Experiment Tracking

To execute the 3 hyperparameter experiments and log metrics:
```bash
python experiments.py
```

### Experiment Summary Table

| Experiment | TF-IDF N-grams | Skill Weight | MAE | Spearman Rank Corr | Mean Text Sim | Mean Skill Cov |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment A** | `(1, 1)` | 0.4 | 19.45 | 0.8757 | 18.43% | 54.58% |
| **Experiment B** | `(1, 2)` | 0.4 | 24.31 | 0.8616 | 10.32% | 54.58% |
| **Experiment C** | `(1, 2)` | 0.5 | 19.88 | 0.8616 | 10.32% | 54.58% |

### View the Interactive MLflow Dashboard

```bash
# Windows PowerShell
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --port 5000

# macOS / Linux
export MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --port 5000
```
Open [http://localhost:5000](http://localhost:5000) to compare runs, visualize parameter correlations, and download artifact files. Detailed instructions are available in [docs/MLflow_EXPERIMENTS.md](docs/MLflow_EXPERIMENTS.md).

---

## 📡 Live MLflow Tracking

The application automatically records an individual MLflow run for **every successful resume-matching calculation** performed via the Gradio web interface.

### How Live Tracking Works
1. When a user uploads a valid PDF resume and enters a job description, the matching engine calculates textual similarity, skill coverage, and final score.
2. Upon successful computation, [`mlflow_tracking.py`](file:///c:/Users/raghu/Videos/CPPE%20Project/resume-job-matcher/mlflow_tracking.py) creates a new MLflow run under the experiment:
   ```text
   Resume-Job-Description-Matcher-Live
   ```
3. Each calculation generates a unique MLflow **Run ID** which is immediately displayed in the Gradio UI:
   `📊 MLflow tracking: Successful | Run ID: <actual-run-id>`
4. Calculations with invalid inputs (empty resume, missing job description, corrupt PDF) or matching errors **never** create runs.
5. If MLflow tracking is unavailable or encounters an error, the matching results are still delivered safely to the user with a graceful degradation status (`MLflow tracking: Unavailable`).

### Information Logged for Each Calculation
- **Parameters:**
  - `matching_algorithm`: `TF-IDF + Cosine Similarity & Weighted Skill Extraction`
  - `ngram_range`: `(1, 2)`
  - `skill_weight`: Configured weight (e.g., `0.4` or slider value)
  - `pdf_parser`: `PyMuPDF (fitz)`
  - `app_version`: `1.0.0`
  - `input_processing`: `lowercase=True, clean_punctuation=True, stop_words=english`
  - `scoring_formula`: `(1 - skill_weight) * text_similarity + skill_weight * skill_coverage`
- **Metrics (Numeric, 0–100 scale):**
  - `final_matching_score`
  - `text_similarity_score`
  - `skill_coverage_score`
  - `matched_skill_count`
  - `missing_skill_count`
  - `job_description_skill_count`
  - `resume_skill_count`
- **Tags & Metadata:**
  - `source`: `gradio_app`
  - `run_type`: `live_prediction`
  - `application`: `resume_job_matcher`
  - `calculation_timestamp`: UTC ISO timestamp
  - Standard MLflow run metadata (`start_time`, `status = FINISHED`)

### 🔒 Privacy and Security Guarantees
Candidate and company data are strictly protected. The tracking pipeline **never** logs or uploads:
- Raw resume text or original PDF files
- Candidate names, emails, telephone numbers, addresses, or any PII
- Full job-description text
- Secrets, credentials, or API tokens
- Artifacts containing identifying text

Only sanitized numeric scores, skill counts, and technical parameters are logged.

### How to Monitor Live Runs in the Dashboard
1. Run the application in one terminal:
   ```bash
   python app.py
   ```
2. In a second terminal, start the MLflow UI:
   ```bash
   # Windows PowerShell
   $env:MLFLOW_ALLOW_FILE_STORE="true"
   mlflow ui --port 5000
   ```
3. Open [http://localhost:5000](http://localhost:5000) and select **`Resume-Job-Description-Matcher-Live`** from the left-hand experiment list.
4. Perform a calculation in Gradio and refresh the MLflow dashboard (a browser refresh displays newly added runs). Each submission produces a separate run with its own metrics and parameters.

### Configuring a Remote Tracking Server
By default, the application writes to the local `./mlruns` directory. To stream runs to a remote or team MLflow tracking server:
```bash
# Windows PowerShell
$env:MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
python app.py

# Linux / macOS
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
python app.py
```

### Local vs. Deployed Tracking Considerations
- **Local Execution:** Runs are persisted directly to the local filesystem (`./mlruns`) and can be explored immediately via `mlflow ui --port 5000`.
- **Hugging Face Spaces Deployment:** Spaces operate with ephemeral storage, meaning local run history resets if the Space container restarts. If persistent experiment tracking is needed in production, configure a hosted remote MLflow tracking URI via Hugging Face Space repository secrets (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`). If no remote server is configured, the local fallback operates seamlessly without throwing errors.

---

## ☁️ Hugging Face Spaces Deployment (Static SDK & Gradio-Lite)

This project supports deployment on Hugging Face Spaces using the **Static** SDK and the **Gradio-Lite** (Pyodide / WebAssembly) template:

1. **Create Space on Hugging Face:**
   - Select **Static** as the Space SDK.
   - Choose the **Gradio-Lite** static template.
2. **Deploy Files:**
   - Push or upload this repository to your Space.
   - The root `index.html` file is automatically served by Hugging Face Spaces as a client-side WebAssembly application running entirely in the visitor's browser (zero backend server cost, 100% private client-side execution).
   - If modifying python source files, regenerate `index.html` using:
     ```bash
     python build_index_html.py
     ```
3. **Local Preview / Testing:**
   - Preview the static Gradio-Lite template locally:
     ```bash
     python -m http.server 8000
     ```
     and open [http://localhost:8000/index.html](http://localhost:8000/index.html).
   - Or run the local full-featured Python server:
     ```bash
     python app.py
     ```
     and open [http://localhost:7860](http://localhost:7860).

---

## ⚠️ Known Limitations

1. **Scanned / Image-Only PDFs:** Text extraction requires standard digital PDF text streams. Scanned PDFs containing pure bitmap images require an external OCR pipeline (such as Tesseract), which is not bundled.
2. **Deterministic Skill Vocabulary:** The technical skill extractor checks against standard technical keywords. Specialized or proprietary internal tool names are evaluated via TF-IDF textual similarity rather than explicit skill tags.
3. **Synthetic Evaluation Ground Truth:** The MLflow evaluation benchmark uses synthetic test pairs designed to test algorithmic robustness across varying overlap levels; scores do not reflect real hiring outcomes.
