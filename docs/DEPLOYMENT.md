# Deployment Guide: Local Execution and Hugging Face Spaces

This guide provides end-to-end instructions for running the **Resume–Job Description Matcher** locally and deploying it to **Hugging Face Spaces** using the native Gradio SDK.

---

## 1. Local Setup and Execution

### Prerequisites
- Python 3.11 or 3.12 installed on your system.
- Git installed.

### Step-by-Step Local Setup

1. **Navigate to the Project Directory:**
   ```bash
   cd resume-job-matcher
   ```

2. **Create and Activate a Virtual Environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Gradio Web Application:**
   ```bash
   python app.py
   ```
   By default, Gradio starts a local web server at:
   ```text
   http://localhost:7860
   ```
   Open this URL in your web browser to test the interactive interface.

---

## 2. Hugging Face Spaces Deployment

The application is architected specifically for frictionless deployment on **Hugging Face Spaces** without requiring Docker or custom containerization:
- Uses standard Gradio SDK (`sdk: gradio`).
- Relies solely on portable Python packages specified in `requirements.txt`.
- Operates independently from local filesystem paths and MLflow servers.

### Step 1: Create a Hugging Face Account
1. Visit [huggingface.co](https://huggingface.co/) and click **Sign Up** (or log in if you already have an account).

### Step 2: Create a New Space
1. In your Hugging Face dashboard, click **Spaces** and then **Create new Space**.
2. Fill in the following details:
   - **Space Name:** `resume-job-matcher` (or your preferred name).
   - **License:** `mit` or `apache-2.0`.
   - **Select the Space SDK:** Choose **Gradio**.
   - **Space Hardware:** Select the default free tier (`CPU basic • 2 vCPU • 16GB RAM`).
   - **Visibility:** Public.
3. Click **Create Space**.

### Step 3: Upload Project Files to the Space

You can push files using Git or upload them directly via the web UI.

#### Option A: Using Git (Recommended)

1. Clone your empty Hugging Face Space repository:
   ```bash
   git clone https://huggingface.co/spaces/<YOUR-USERNAME>/<YOUR-SPACE-NAME> hf-space
   ```
2. Copy the contents of `resume-job-matcher` into the cloned `hf-space` directory:
   - `app.py` (Must be at the root of the space)
   - `matcher.py`
   - `pdf_utils.py`
   - `requirements.txt`
   - `README.md` (Contains Hugging Face frontmatter metadata)
   - `sample_data/` (Directory with sample resume and job description)
3. Commit and push the files:
   ```bash
   cd hf-space
   git add app.py matcher.py pdf_utils.py requirements.txt README.md sample_data/
   git commit -m "Deploy Resume-Job Description Matcher Gradio App"
   git push
   ```

#### Option B: Direct Web Interface Upload
1. In your Hugging Face Space, click the **Files** tab.
2. Click **Add file** $\rightarrow$ **Upload files**.
3. Upload `app.py`, `matcher.py`, `pdf_utils.py`, `requirements.txt`, `README.md`, and the `sample_data/` directory.
4. Click **Commit changes to main**.

### Step 4: Monitor the Build
1. Hugging Face Spaces will automatically detect `requirements.txt` and install all dependencies.
2. In the **App** tab or **Logs** tab, observe the container building and Gradio launching.
3. Once the build finishes, the status badge will change to **Running**.

### Step 5: Test the Live Application
1. In the Hugging Face Space interface, upload `sample_data/sample_resume.pdf` (or click **Load Example**).
2. Paste a job description into the text area.
3. Click **Calculate Match**.
4. Verify that:
   - The prominent Match Score is displayed.
   - Text Similarity and Skill Coverage scores are calculated.
   - Matching skills are shown with green badges and missing skills with red badges.
   - Actionable recommendations are rendered below the score.

---

## 3. Architecture & Portability Checklist

| Requirement | Implementation Verification |
| :--- | :--- |
| **No Docker Requirement** | Gradio SDK auto-installs from `requirements.txt` on the free tier. |
| **No Local Paths** | All asset lookups use relative `Path(__file__).resolve().parent`. |
| **Independent of MLflow** | `app.py` imports only `matcher.py` and `pdf_utils.py`; does not require an active MLflow server to compute scores. |
| **No Private Credentials** | The application requires no external API keys or secret tokens. |
| **Metadata Compliant** | `README.md` includes valid YAML frontmatter specifying `title`, `emoji`, `sdk: gradio`, and `app_file: app.py`. |
