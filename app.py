"""
app.py
======
Resume–Job Description Matcher
Interactive Web Application built with Gradio and scikit-learn.

Deployable on Hugging Face Spaces using the Gradio SDK.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import gradio as gr

from matcher import match_resume_to_job
from mlflow_tracking import log_live_match
from pdf_utils import PDFProcessingError, extract_text_from_pdf

import base64

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_RESUME_TXT = BASE_DIR / "sample_data" / "sample_resume.txt"
SAMPLE_JOB_TXT = BASE_DIR / "sample_data" / "sample_job.txt"
SAMPLE_RESUME_PDF = BASE_DIR / "sample_data" / "sample_resume.pdf"

DEFAULT_SAMPLE_JOB = """Role: Senior Python & Machine Learning Engineer
Location: Remote (US) / New York, NY
Company: InnovateTech AI Solutions

About the Role:
We are seeking a talented Senior Python Engineer with a strong interest in Machine Learning to join our core intelligence team. In this role, you will architect resilient backend services, productionize machine learning workflows, and collaborate on cloud-native deployments.

Key Responsibilities:
- Design, build, and optimize scalable RESTful backend services using Python and FastAPI.
- Develop predictive and machine learning pipelines using scikit-learn, pandas, and NumPy.
- Containerize services with Docker and manage automated deployments on AWS.
- Design relational schemas and optimize database performance using SQL (PostgreSQL).
- Collaborate with frontend and mobile teams, practicing code reviews and Git version control.
- Ensure high test coverage and reliability with automated testing suites.

Required Qualifications & Skills:
- 3+ years of professional experience with Python software engineering.
- Hands-on experience building backend APIs with FastAPI, Flask, or Django.
- Proficiency in data manipulation and model building with pandas, NumPy, and scikit-learn.
- Strong knowledge of Machine Learning concepts and algorithmic problem solving.
- Practical experience with Docker containerization and AWS cloud services.
- Solid understanding of SQL database querying and optimization.
- Familiarity with version control using Git.

Preferred Qualifications:
- Experience with frontend technologies such as React or JavaScript.
- Familiarity with data visualization tools such as Power BI or Excel dashboards.
- Bachelor's degree in Computer Science, Engineering, or equivalent practical experience.
"""

SAMPLE_RESUME_B64 = (
    "JVBERi0xLjcKJcK1wrYKJSBXcml0dGVuIGJ5IE11UERGIDEuMjguMgoKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFIv"
    "SW5mbzw8L1Byb2R1Y2VyKE11UERGIDEuMjguMik+Pj4+CmVuZG9iagoKMiAwIG9iago8PC9UeXBlL1BhZ2VzL0NvdW50IDEvS2lkc1s0"
    "IDAgUl0+PgplbmRvYmoKCjMgMCBvYmoKPDwvRm9udDw8L2hlbHYgNSAwIFI+Pj4+CmVuZG9iagoKNCAwIG9iago8PC9UeXBlL1BhZ2Uv"
    "TWVkaWFCb3hbMCAwIDU5NSA4NDJdL1JvdGF0ZSAwL1Jlc291cmNlcyAzIDAgUi9QYXJlbnQgMiAwIFIvQ29udGVudHNbNiAwIFJdPj4K"
    "ZW5kb2JqCgo1IDAgb2JqCjw8L1R5cGUvRm9udC9TdWJ0eXBlL1R5cGUxL0Jhc2VGb250L0hlbHZldGljYS9FbmNvZGluZy9XaW5BbnNp"
    "RW5jb2Rpbmc+PgplbmRvYmoKCjYgMCBvYmoKPDwvTGVuZ3RoIDE0ODUvRmlsdGVyL0ZsYXRlRGVjb2RlPj4Kc3RyZWFtCnjalVe7ruU2"
    "DOzPV5w6QDayREkWEKQIkiZdgNMFKe71AymSYpv9/gyHki0/brKLxT1ry3qQw+GQenx+/Px6DE+Hf8Mzumce/fP1z+OHv5a/vzzLp/h8"
    "rc8/fpRBJolx9E5mWaOXjJHlpz9fvz3c8/shfHIxPF+/PDAz5ZimNHmMZElDCundO590Jd/GVNKSIt6nFNOQPd50JGMk6v/2PcXsvcvY"
    "R/CW8TWWtGL2u+dYLDz99Z1aF5PuXdIU3rzDE3aGHwtGdUXGyCJOx/A0Z6ff8TWkNc31DJyaRUbY7rkH3xPft5lr27nft66faPU77BbB"
    "E/foxvSp20ffbnfbfdqfogPeqyTgH2KQgudF46EIx4h4zDJE3+MRPfYNGocsOfg5CfYvOcGSRVfBioQvmejHW9yr/94F8YhfLhYr7OZ0"
    "NUYc1uKJJwWsXbF60LirT1l/caaeCK9xBjCInJe3eOtcDysRN0OojgGHJHWForji9MoQ7BCI25A8o6iWaoywChYZ5piRxSyivft+Oj6c"
    "8YD9JStXwSLMX4y59I5n7v7ortm8KUQSuChKOpOsTDgBaxEPJ4Ujgm9BOQBUO98wI0e1LdC/SPtgDWwq9eSZ72IWyQ2mXEM7iJR6o3u5"
    "hlU65Vu65JvO1V0rU6OilS1KO9oF1sNLoLatV09y9UU9Pni2wJtZkfGGkLGkcCd61LMLDCaTW2wQ1UERtj0tKzE/Ef3KMqK2Ehvdezpw"
    "BrlENDznJcOSseV3Pikmi67En7KeftNnRCPHjYtO3mANYoqMCWQM1tOuMQqy7nCyBM1Of5vBEqiZiAHWRbyNyOCC/2sWyzveJpli2Nf4"
    "eUMP4qe6hX/NromnZtg6wAPYr3pzYeMH1lfMT36Y/cfzxZt+N/So4qlaA002PbbTb7ifNE9zqIot6c2sTiut8NQzTFKdYkZgXT5b8L81"
    "gzZV9qgd38LZjqvT15xF5TGvDen5Ym+Xz9UyZKYgKytKJz1oOuDHGILZFDUS7Sl4X7pMYJyA7opfucvfkzWXrDKewELVO+VV7Ngw57K/"
    "aQ1J1Ltvq0naJUT99RiNGA0Su3WBXGIlv/Lrg/pfveixZQ1DrvY678fgg8Mf1m3Zo2pmqlNO2GhkGW3TBP7KB9UnVm0pzKSVWPpsqnZR"
    "zUMe7vpwzA+XxsrFqa0LQ0AMtNoqQ7UqgpeuzacvQ46st6wKF+aZ0g2VC8rRt2r5CeeWuVTAIcnZgwNLd0TWiphZ4BmVeMFqr1vOEGgc"
    "35mdya0c03bi3Q6sWEPV7qBYoFvALxIF7c7F/9I6u7a+xZMsWbUadvZaJ2BdTKy9TNh6mZbt5neiDsBe65bYD9B/sqbWpNtautW5g/Lc"
    "1sqz8lpflK1mDeyTFA1fe4ahWpqO/U5nNWuZ9TXsGmrFJ39F/zLtPdn5kXqqFWuNU82XehaRpddbHa0R1m5wpmUzO7WWNZbZQqwYLUXz"
    "BgPmeO311p5vu70tH9nlaiVZ9phYN1pjS7bl1gEM7PSOnN8VtrsLcH29z1wysO87blhG5M2a2k1pTG30vk/4qs6cVYORY/906F13DXSm"
    "gZsilktl6DvGo/LZXarTEnpjdwpnamvdMzssQ2fO3W3vmDH/qY/XPuHSkxy6rLue5sQbvSyS45Ga5SoDdzXkLcy0Inc9aI2t35h8uje0"
    "KlorQa6M3O8Z9U5wX10T9YmWpU71dMdR+c2Iy26FKh3xjk3DqX2CyN5pn+ldPmSJsLrpruqt3t6EKjpYbjAKWtHC3n3tXrNuBotgu50g"
    "ijT+pen7noeo/5YBY/B9tKD8e3aMVZePirXcdHwX9oVCv/Umyv4hd/llETVdsX74qLy5dDOHTjFsnWm63xga67d2OwjtVkGmf5i7oq29"
    "9vYDHtgPdd987Swt33y7Q8fQbkWpu1VK5ScxY84f53V6YVhst6MaX9v7ctPRb34Uu02oNxtTFHOyq2mHqcWvr8fvj38BmdEkQAplbmRz"
    "dHJlYW0KZW5kb2JqCgp4cmVmCjAgNwowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwNDIgMDAwMDAgbiAKMDAwMDAwMDEyMCAwMDAw"
    "MCBuIAowMDAwMDAwMTcyIDAwMDAwIG4gCjAwMDAwMDAyMTMgMDAwMDAgbiAKMDAwMDAwMDMyMCAwMDAwMCBuIAowMDAwMDAwNDA5IDAw"
    "MDAwIG4gCgp0cmFpbGVyCjw8L1NpemUgNy9Sb290IDEgMCBSL0lEWzwzNzU1N0JDMkFFMjlDMzk3QzNBQUMyOUJDMkEzQzNBOT48OEM1"
    "Njc5NzJFQkRFRUFBQTYzRjIyQkQxMUNFRTZDRUI+XT4+CnN0YXJ0eHJlZgoxOTY0CiUlRU9GCg=="
)


def load_sample_content() -> tuple[str | None, str]:
    """Load default sample resume PDF path and sample job description text."""
    if SAMPLE_RESUME_PDF.exists():
        pdf_path = str(SAMPLE_RESUME_PDF)
    else:
        # Fallback for Gradio-Lite / WebAssembly environments
        temp_dir = Path("/tmp") if Path("/tmp").exists() else BASE_DIR / "sample_data"
        temp_dir.mkdir(parents=True, exist_ok=True)
        sample_pdf_file = temp_dir / "sample_resume.pdf"
        try:
            if not sample_pdf_file.exists():
                sample_pdf_file.write_bytes(base64.b64decode(SAMPLE_RESUME_B64))
            pdf_path = str(sample_pdf_file)
        except Exception:
            pdf_path = None

    if SAMPLE_JOB_TXT.exists():
        job_text = SAMPLE_JOB_TXT.read_text(encoding="utf-8")
    else:
        job_text = DEFAULT_SAMPLE_JOB

    return pdf_path, job_text


def compute_match(
    resume_file: Any,
    job_description: str,
    skill_weight: float = 0.4,
) -> tuple[
    str,  # Score Display (HTML)
    float,  # Text Similarity Progress/Gauge
    float,  # Skill Coverage Progress/Gauge
    str,  # Matching Skills badges (HTML)
    str,  # Missing Skills badges (HTML)
    str,  # Suggestions (Markdown)
    str,  # Status/Error banner (HTML)
]:
    """
    Process uploaded resume and job description, calculate similarity,
    and return formatted presentation components for the UI.
    """
    empty_result = (
        '<div class="score-card empty"><h2>--</h2><span>Match Score</span></div>',
        0.0,
        0.0,
        '<p class="empty-hint">No matching skills detected</p>',
        '<p class="empty-hint">No missing skills detected</p>',
        "Submit a resume and job description to view actionable suggestions.",
        "",
    )

    # 1. Validate inputs
    if resume_file is None:
        error_html = (
            '<div class="alert-box error">'
            '<strong>⚠️ Missing Resume:</strong> Please upload a PDF resume before calculating match.'
            '</div>'
        )
        return (
            empty_result[0],
            0.0,
            0.0,
            empty_result[3],
            empty_result[4],
            empty_result[5],
            error_html,
        )

    if not job_description or not job_description.strip():
        error_html = (
            '<div class="alert-box error">'
            '<strong>⚠️ Missing Job Description:</strong> Please paste the target job description text.'
            '</div>'
        )
        return (
            empty_result[0],
            0.0,
            0.0,
            empty_result[3],
            empty_result[4],
            empty_result[5],
            error_html,
        )

    # 2. Extract PDF text
    try:
        # Gradio gr.File may return a str path or an object with a .name attribute
        file_path = getattr(resume_file, "name", resume_file)
        resume_text = extract_text_from_pdf(file_path)
    except PDFProcessingError as err:
        error_html = f'<div class="alert-box error"><strong>📄 PDF Processing Error:</strong> {err}</div>'
        return (
            empty_result[0],
            0.0,
            0.0,
            empty_result[3],
            empty_result[4],
            empty_result[5],
            error_html,
        )
    except ValueError as err:
        error_html = f'<div class="alert-box error"><strong>⚠️ Input Validation Error:</strong> {err}</div>'
        return (
            empty_result[0],
            0.0,
            0.0,
            empty_result[3],
            empty_result[4],
            empty_result[5],
            error_html,
        )
    except Exception as err:
        error_html = f'<div class="alert-box error"><strong>❌ Unexpected Error:</strong> {err}</div>'
        return (
            empty_result[0],
            0.0,
            0.0,
            empty_result[3],
            empty_result[4],
            empty_result[5],
            error_html,
        )

    # 3. Calculate match
    try:
        # Constrain skill_weight safely between 0.0 and 1.0
        weight = float(max(0.0, min(1.0, skill_weight)))
        results = match_resume_to_job(
            resume_text=resume_text,
            job_text=job_description,
            skill_weight=weight,
            ngram_range=(1, 2),
        )
    except Exception as err:
        error_html = f'<div class="alert-box error"><strong>⚙️ Matching Engine Error:</strong> {err}</div>'
        return (
            empty_result[0],
            0.0,
            0.0,
            empty_result[3],
            empty_result[4],
            empty_result[5],
            error_html,
        )

    final_score = results["final_score"]
    text_sim = results["text_similarity"]
    skill_cov = results["skill_coverage"]
    matching_skills = results["matching_skills"]
    missing_skills = results["missing_skills"]
    suggestions = results["suggestions"]

    # 4. Determine score badge color scheme
    if final_score >= 75.0:
        color_class = "score-high"
        verdict = "Strong Alignment"
    elif final_score >= 50.0:
        color_class = "score-mid"
        verdict = "Moderate Alignment"
    else:
        color_class = "score-low"
        verdict = "Low Alignment"

    score_html = f"""
    <div class="score-card {color_class}">
        <div class="score-number">{final_score:.1f}%</div>
        <div class="score-label">{verdict}</div>
        <div class="score-subtext">Weighted Blend: {int((1-weight)*100)}% TF-IDF & {int(weight*100)}% Skills</div>
    </div>
    """

    # 5. Format skill badges
    if matching_skills:
        matching_badges = "".join([f'<span class="badge badge-match">✓ {s}</span>' for s in matching_skills])
        matching_html = f'<div class="badge-container">{matching_badges}</div>'
    else:
        matching_html = '<p class="empty-hint">None of the recognized skills were matched.</p>'

    if missing_skills:
        missing_badges = "".join([f'<span class="badge badge-missing">✗ {s}</span>' for s in missing_skills])
        missing_html = f'<div class="badge-container">{missing_badges}</div>'
    else:
        missing_html = '<p class="empty-hint success-text">No missing skills! Resume covers all recognized skills from the job description.</p>'

    # 6. Format suggestions
    if suggestions:
        suggestions_md = "\n\n".join([f"- **{i+1}.** {sug}" for i, sug in enumerate(suggestions)])
    else:
        suggestions_md = "Great match! No immediate keyword improvements detected."

    # 7. MLflow Live Tracking (Privacy-Preserving)
    run_id = log_live_match(
        results=results,
        skill_weight=weight,
        ngram_range=(1, 2),
    )

    if run_id:
        tracking_status_html = (
            f'<div class="tracking-badge tracking-success">'
            f'📊 <strong>MLflow tracking:</strong> Successful &nbsp;|&nbsp; '
            f'<strong>Run ID:</strong> <code>{run_id}</code>'
            f'</div>'
        )
    else:
        tracking_status_html = (
            '<div class="tracking-badge tracking-warning">'
            '⚠️ <strong>MLflow tracking:</strong> Unavailable — matching results were calculated, '
            'but the run could not be recorded.'
            '</div>'
        )

    success_banner = (
        '<div class="alert-box success">'
        '<strong>✅ Analysis Completed:</strong> Resume parsed successfully and alignment metrics computed.'
        f'{tracking_status_html}'
        '</div>'
    )

    return (
        score_html,
        text_sim,
        skill_cov,
        matching_html,
        missing_html,
        suggestions_md,
        success_banner,
    )


# Custom CSS styling for a clean, professional aesthetic
CUSTOM_CSS = """
.container {
    max-width: 1200px;
    margin: 0 auto;
}
.header-box {
    text-align: center;
    padding: 1.5rem 1rem 0.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-subtitle {
    font-size: 1.05rem;
    color: #4b5563;
    margin-bottom: 0.75rem;
}
.disclaimer-banner {
    background: #f0fdf4;
    border: 1px solid #86efac;
    color: #166534;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: 0.9rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.score-card {
    text-align: center;
    padding: 1.5rem 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.score-high {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
}
.score-mid {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
}
.score-low {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: white;
}
.score-card.empty {
    background: #f3f4f6;
    color: #6b7280;
}
.score-number {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
}
.score-label {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.score-subtext {
    font-size: 0.85rem;
    opacity: 0.9;
    margin-top: 0.25rem;
}
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.5rem 0;
}
.badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.badge-match {
    background-color: #d1fae5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}
.badge-missing {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.empty-hint {
    color: #6b7280;
    font-style: italic;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}
.success-text {
    color: #059669;
    font-weight: 500;
}
.alert-box {
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: 0.9rem;
    margin-bottom: 0.75rem;
}
.alert-box.error {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    color: #991b1b;
}
.alert-box.success {
    background: #f0fdf4;
    border: 1px solid #86efac;
    color: #166534;
}
.tracking-badge {
    margin-top: 0.5rem;
    padding: 0.35rem 0.6rem;
    border-radius: 6px;
    font-size: 0.85rem;
    display: inline-block;
}
.tracking-success {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}
.tracking-warning {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
}
.tracking-badge code {
    background: rgba(0, 0, 0, 0.05);
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
    font-family: monospace;
}
"""


def build_app() -> gr.Blocks:
    """Build and configure the Gradio interface."""
    with gr.Blocks(title="Resume - Job Description Matcher") as demo:
        with gr.Column(elem_classes=["container"]):
            # Include custom CSS directly for full compatibility across Gradio versions
            gr.HTML(f"<style>{CUSTOM_CSS}</style>")

            # Header
            gr.HTML(
                """
                <div class="header-box">
                    <div class="header-title">📄 Resume–Job Description Matcher</div>
                    <div class="header-subtitle">
                        Machine Learning & Natural Language Processing Alignment Engine
                    </div>
                </div>
                """
            )

            # Disclaimer banner
            gr.HTML(
                """
                <div class="disclaimer-banner">
                    <span>💡</span>
                    <span>
                        <strong>Note on Methodology:</strong> The matching score evaluates textual and technical skill alignment
                        using TF-IDF vectorization and cosine similarity. It measures keyword and vocabulary correspondence,
                        <strong>not the probability of getting hired</strong>.
                    </span>
                </div>
                """
            )

            # Status Banner
            status_banner = gr.HTML("", visible=True)

            with gr.Row():
                # Left Column: Inputs
                with gr.Column(scale=5):
                    gr.Markdown("### 1. Upload & Job Description")
                    resume_input = gr.File(
                        label="Upload Resume (PDF)",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    job_input = gr.Textbox(
                        label="Job Description",
                        placeholder="Paste the target job description here...",
                        lines=10,
                    )

                    with gr.Accordion("⚙️ Advanced Settings (Skill Weight)", open=False):
                        skill_weight_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.4,
                            step=0.05,
                            label="Skill Weight Factor",
                            info="0.0 = Pure TF-IDF Text Similarity | 1.0 = Pure Skill Coverage (Default: 0.4)",
                        )

                    with gr.Row():
                        calculate_btn = gr.Button("🔍 Calculate Match", variant="primary", scale=2)
                        load_sample_btn = gr.Button("📋 Load Example", variant="secondary", scale=1)
                        clear_btn = gr.Button("🔄 Clear", variant="stop", scale=1)

                # Right Column: Results
                with gr.Column(scale=5):
                    gr.Markdown("### 2. Match Analysis & Score")
                    score_display = gr.HTML(
                        '<div class="score-card empty"><h2>--</h2><span>Upload a resume and job description</span></div>'
                    )

                    with gr.Row():
                        text_sim_gauge = gr.Number(
                            label="Text Similarity Score (%)",
                            value=0.0,
                            precision=2,
                        )
                        skill_cov_gauge = gr.Number(
                            label="Skill Coverage Score (%)",
                            value=0.0,
                            precision=2,
                        )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Matching Skills Found")
                            matching_skills_output = gr.HTML('<p class="empty-hint">Awaiting submission...</p>')
                        with gr.Column():
                            gr.Markdown("#### Missing Skills")
                            missing_skills_output = gr.HTML('<p class="empty-hint">Awaiting submission...</p>')

                    gr.Markdown("### 3. Actionable Suggestions")
                    suggestions_output = gr.Markdown(
                        "Submit a resume and job description to view actionable suggestions."
                    )

            # Event Handlers
            calculate_btn.click(
                fn=compute_match,
                inputs=[resume_input, job_input, skill_weight_slider],
                outputs=[
                    score_display,
                    text_sim_gauge,
                    skill_cov_gauge,
                    matching_skills_output,
                    missing_skills_output,
                    suggestions_output,
                    status_banner,
                ],
            )

            load_sample_btn.click(
                fn=load_sample_content,
                inputs=[],
                outputs=[resume_input, job_input],
            )

            clear_btn.click(
                fn=lambda: (
                    None,
                    "",
                    0.4,
                    '<div class="score-card empty"><h2>--</h2><span>Upload a resume and job description</span></div>',
                    0.0,
                    0.0,
                    '<p class="empty-hint">Awaiting submission...</p>',
                    '<p class="empty-hint">Awaiting submission...</p>',
                    "Submit a resume and job description to view actionable suggestions.",
                    "",
                ),
                inputs=[],
                outputs=[
                    resume_input,
                    job_input,
                    skill_weight_slider,
                    score_display,
                    text_sim_gauge,
                    skill_cov_gauge,
                    matching_skills_output,
                    missing_skills_output,
                    suggestions_output,
                    status_banner,
                ],
            )

    return demo


# Create app instance for Gradio / Hugging Face Spaces
app = build_app()

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 7860))
        app.launch(server_name="0.0.0.0", server_port=port)
    except Exception:
        app.launch()
