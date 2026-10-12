"""
build_index_html.py
===================
Generates a standalone, client-side WebAssembly (Pyodide) application for Hugging Face Spaces
using the 'static' SDK. Eliminates the '@gradio/lite' micropip wheel bug while providing the
identical Gradio look-and-feel, layout, and functionality.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

matcher_code = (BASE_DIR / "matcher.py").read_text(encoding="utf-8")
pdf_utils_code = (BASE_DIR / "pdf_utils.py").read_text(encoding="utf-8")
sample_job_text = (BASE_DIR / "sample_data" / "sample_job.txt").read_text(encoding="utf-8")
sample_resume_text = (BASE_DIR / "sample_data" / "sample_resume.txt").read_text(encoding="utf-8")

import base64
sample_pdf_bytes = (BASE_DIR / "sample_data" / "sample_resume.pdf").read_bytes()
sample_pdf_b64 = base64.b64encode(sample_pdf_bytes).decode("ascii")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Resume–Job Description Matcher</title>
    <!-- Pyodide: Full WebAssembly Python 3.12 runtime with scikit-learn & numpy -->
    <script src="https://cdn.jsdelivr.net/pyodide/v0.27.3/full/pyodide.js"></script>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            padding: 1.5rem 1rem 3rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header-box {{
            text-align: center;
            padding: 1.5rem 1rem 0.5rem;
            margin-bottom: 1rem;
        }}
        .header-title {{
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header-subtitle {{
            font-size: 1.05rem;
            color: #4b5563;
            margin-bottom: 0.75rem;
        }}
        .badge-client {{
            display: inline-block;
            background: #ede9fe;
            color: #6d28d9;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid #ddd6fe;
        }}
        .disclaimer-banner {{
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
        }}
        #loader-overlay {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255, 255, 255, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            transition: opacity 0.4s ease;
        }}
        .spinner {{
            width: 50px;
            height: 50px;
            border: 5px solid #e2e8f0;
            border-top: 5px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .loader-title {{
            margin-top: 1.2rem;
            font-size: 1.2rem;
            font-weight: 700;
            color: #1e293b;
        }}
        .loader-status {{
            margin-top: 0.4rem;
            font-size: 0.9rem;
            color: #64748b;
        }}
        .main-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}
        @media (max-width: 900px) {{
            .main-layout {{
                grid-template-columns: 1fr;
            }}
        }}
        .panel {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
        .panel h3 {{
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #1e293b;
        }}
        .form-group {{
            margin-bottom: 1.25rem;
        }}
        label {{
            display: block;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.4rem;
            color: #334155;
        }}
        .help-text {{
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.25rem;
        }}
        .file-dropzone {{
            border: 2px dashed #cbd5e1;
            border-radius: 8px;
            padding: 1.5rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            background: #f8fafc;
        }}
        .file-dropzone:hover {{
            border-color: var(--primary);
            background: #eff6ff;
        }}
        .file-dropzone input[type="file"] {{
            display: none;
        }}
        .file-selected {{
            font-weight: 600;
            color: #166534;
            margin-top: 0.5rem;
            font-size: 0.85rem;
        }}
        textarea {{
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem;
            font-family: inherit;
            font-size: 0.9rem;
            resize: vertical;
            min-height: 140px;
        }}
        textarea:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }}
        .slider-wrap {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .slider-wrap input[type="range"] {{
            flex: 1;
            height: 6px;
            accent-color: var(--primary);
        }}
        .slider-value {{
            font-weight: 700;
            min-width: 45px;
            text-align: right;
            font-size: 0.95rem;
        }}
        .btn-row {{
            display: flex;
            gap: 0.75rem;
            margin-top: 1rem;
        }}
        .btn {{
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.15s;
        }}
        .btn-primary {{
            background: var(--primary);
            color: white;
            flex: 2;
        }}
        .btn-primary:hover {{
            background: var(--primary-hover);
        }}
        .btn-secondary {{
            background: #f1f5f9;
            color: #334155;
            border-color: #cbd5e1;
            flex: 1;
        }}
        .btn-secondary:hover {{
            background: #e2e8f0;
        }}
        .btn-clear {{
            background: transparent;
            color: #64748b;
            border-color: #cbd5e1;
        }}
        .btn-clear:hover {{
            background: #f1f5f9;
            color: #334155;
        }}
        /* Score Card */
        .score-card {{
            text-align: center;
            padding: 1.5rem 1rem;
            border-radius: 12px;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
            transition: all 0.3s;
        }}
        .score-card.empty {{
            background: #f1f5f9;
            color: #64748b;
        }}
        .score-card.score-high {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }}
        .score-card.score-mid {{
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
        }}
        .score-card.score-low {{
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
        }}
        .score-number {{
            font-size: 3.5rem;
            font-weight: 800;
            line-height: 1;
        }}
        .score-label {{
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }}
        .score-subtext {{
            font-size: 0.85rem;
            opacity: 0.9;
            margin-top: 0.25rem;
        }}
        /* Progress metrics */
        .metric-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }}
        .metric-card {{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
        }}
        .metric-card-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
        }}
        .metric-card-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e293b;
            margin-top: 0.2rem;
        }}
        /* Badges */
        .badge-section {{
            margin-bottom: 1.25rem;
        }}
        .badge-section-title {{
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .badge-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .badge-match {{
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }}
        .badge-missing {{
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }}
        .empty-hint {{
            color: #64748b;
            font-style: italic;
            font-size: 0.9rem;
        }}
        .alert-box {{
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            display: none;
        }}
        .alert-box.error {{
            background: #fef2f2;
            border: 1px solid #fca5a5;
            color: #991b1b;
        }}
        .alert-box.success {{
            background: #f0fdf4;
            border: 1px solid #86efac;
            color: #166534;
        }}
        .suggestions-box {{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            font-size: 0.9rem;
            white-space: pre-line;
        }}
    </style>
</head>
<body>
    <div id="loader-overlay">
        <div class="spinner"></div>
        <div class="loader-title">Loading WebAssembly Engine</div>
        <div class="loader-status" id="loader-status">Fetching Pyodide runtime...</div>
    </div>

    <div class="container">
        <!-- Header -->
        <div class="header-box">
            <div class="header-title">📄 Resume–Job Description Matcher</div>
            <div class="header-subtitle">
                Machine Learning & Natural Language Processing Alignment Engine
            </div>
            <div>
                <span class="badge-client">⚡ 100% In-Browser WebAssembly (Pyodide & scikit-learn)</span>
            </div>
        </div>

        <!-- Disclaimer banner -->
        <div class="disclaimer-banner">
            <span>💡</span>
            <span>
                <strong>Note on Methodology:</strong> Evaluates textual and technical skill alignment using TF-IDF vectorization and cosine similarity. Measures keyword and vocabulary correspondence, <strong>not the probability of getting hired</strong>.
            </span>
        </div>

        <div id="status-banner" class="alert-box"></div>

        <div class="main-layout">
            <!-- Left Column: Inputs -->
            <div class="panel">
                <h3>1. Candidate Resume & Job Description</h3>

                <div class="form-group">
                    <label>Upload Resume (PDF)</label>
                    <div class="file-dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
                        <div style="font-size: 1.8rem;">📎</div>
                        <div style="font-weight: 600; margin-top: 0.25rem;">Click to upload or drag & drop PDF</div>
                        <div class="help-text">Standard PDF format (.pdf)</div>
                        <div id="file-name" class="file-selected" style="display: none;"></div>
                        <input type="file" id="file-input" accept=".pdf" />
                    </div>
                </div>

                <div class="form-group">
                    <label for="resume-text">Or Paste Resume Text directly:</label>
                    <textarea id="resume-text" placeholder="Alternatively, paste resume plain text here..."></textarea>
                </div>

                <div class="form-group">
                    <label for="job-text">Target Job Description</label>
                    <textarea id="job-text" placeholder="Paste the target job description here..."></textarea>
                </div>

                <div class="form-group">
                    <label>Skill Weight Factor: <span id="weight-display">0.40</span></label>
                    <div class="slider-wrap">
                        <input type="range" id="skill-weight" min="0" max="1" step="0.05" value="0.4" />
                    </div>
                    <div class="help-text">0.0 = Pure TF-IDF Text Similarity | 1.0 = Pure Skill Coverage (Default: 0.4)</div>
                </div>

                <div class="btn-row">
                    <button class="btn btn-primary" id="btn-calculate">✨ Calculate Match</button>
                    <button class="btn btn-secondary" id="btn-example">📄 Load Example</button>
                    <button class="btn btn-clear" id="btn-clear">Clear</button>
                </div>
            </div>

            <!-- Right Column: Results -->
            <div class="panel">
                <h3>2. Match Results & Breakdown</h3>

                <!-- Score Card -->
                <div id="score-card" class="score-card empty">
                    <div class="score-number" id="score-number">--</div>
                    <div class="score-label" id="score-label">Awaiting Submission</div>
                    <div class="score-subtext" id="score-subtext">Submit resume and job description to evaluate</div>
                </div>

                <!-- Metrics -->
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-card-label">Text Similarity</div>
                        <div class="metric-card-value" id="text-sim-val">0.0%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-card-label">Skill Coverage</div>
                        <div class="metric-card-value" id="skill-cov-val">0.0%</div>
                    </div>
                </div>

                <!-- Matching Skills -->
                <div class="badge-section">
                    <div class="badge-section-title">Matching Skills Found</div>
                    <div id="matching-badges" class="badge-container">
                        <p class="empty-hint">Awaiting submission...</p>
                    </div>
                </div>

                <!-- Missing Skills -->
                <div class="badge-section">
                    <div class="badge-section-title">Missing Skills</div>
                    <div id="missing-badges" class="badge-container">
                        <p class="empty-hint">Awaiting submission...</p>
                    </div>
                </div>

                <!-- Suggestions -->
                <div class="badge-section">
                    <div class="badge-section-title">3. Actionable Suggestions</div>
                    <div class="suggestions-box" id="suggestions-box">Submit a resume and job description to view actionable suggestions.</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Application Logic -->
    <script>
        const MATCHER_PY = {json.dumps(matcher_code)};
        const PDF_UTILS_PY = {json.dumps(pdf_utils_code)};
        const SAMPLE_JOB = {json.dumps(sample_job_text)};
        const SAMPLE_RESUME_TXT = {json.dumps(sample_resume_text)};
        const SAMPLE_PDF_B64 = "{sample_pdf_b64}";

        let pyodideInstance = null;
        let selectedPdfBytes = null;

        const updateStatus = (text) => {{
            document.getElementById('loader-status').innerText = text;
        }};

        async function initPyodide() {{
            try {{
                updateStatus("Loading WebAssembly core...");
                pyodideInstance = await loadPyodide({{
                    stdout: (msg) => console.log("[Python Stdout]:", msg),
                    stderr: (msg) => console.warn("[Python Stderr]:", msg),
                }});

                updateStatus("Loading scikit-learn & numpy packages...");
                await pyodideInstance.loadPackage(["numpy", "scikit-learn", "scipy"]);

                updateStatus("Loading pypdf for PDF extraction...");
                await pyodideInstance.loadPackage("micropip");
                const micropip = pyodideInstance.pyimport("micropip");
                await micropip.install("pypdf");

                updateStatus("Compiling NLP matcher engine...");
                // Mount python files
                pyodideInstance.FS.writeFile("/home/pyodide/matcher.py", MATCHER_PY);
                pyodideInstance.FS.writeFile("/home/pyodide/pdf_utils.py", PDF_UTILS_PY);

                // Run python engine wrapper
                await pyodideInstance.runPythonAsync(`
import sys
import io
import base64
import json
from matcher import match_resume_to_job
from pdf_utils import extract_text_from_pdf

def evaluate_match_json(pdf_b64, raw_text, job_desc, skill_weight):
    resume_text = ""
    if pdf_b64 and len(pdf_b64) > 0:
        pdf_bytes = base64.b64decode(pdf_b64)
        resume_text = extract_text_from_pdf(pdf_bytes)
    elif raw_text and raw_text.strip():
        resume_text = raw_text.strip()
    else:
        raise ValueError("Missing Resume: Please upload a valid PDF document or paste resume text.")

    if not job_desc or not job_desc.strip():
        raise ValueError("Missing Job Description: Please paste a target job description.")

    results = match_resume_to_job(
        resume_text=resume_text,
        job_description=job_desc.strip(),
        skill_weight=float(skill_weight),
        ngram_range=(1, 2),
    )
    return json.dumps(results)
                `);

                // Hide loader
                const overlay = document.getElementById("loader-overlay");
                overlay.style.opacity = "0";
                setTimeout(() => {{
                    overlay.style.display = "none";
                }}, 400);

            }} catch (err) {{
                console.error("Initialization Error:", err);
                document.getElementById('loader-status').innerHTML =
                    '<span style="color:#ef4444; font-weight:700;">Error: ' + err.message + '</span>';
            }}
        }}

        // File upload listeners
        const fileInput = document.getElementById('file-input');
        const dropzone = document.getElementById('dropzone');
        const fileNameEl = document.getElementById('file-name');

        fileInput.addEventListener('change', (e) => {{
            const file = e.target.files[0];
            if (file) {{
                handleSelectedFile(file);
            }}
        }});

        dropzone.addEventListener('dragover', (e) => {{
            e.preventDefault();
            dropzone.style.borderColor = '#2563eb';
        }});
        dropzone.addEventListener('dragleave', () => {{
            dropzone.style.borderColor = '#cbd5e1';
        }});
        dropzone.addEventListener('drop', (e) => {{
            e.preventDefault();
            dropzone.style.borderColor = '#cbd5e1';
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {{
                handleSelectedFile(e.dataTransfer.files[0]);
            }}
        }});

        function handleSelectedFile(file) {{
            if (!file.name.toLowerCase().endsWith('.pdf')) {{
                showBanner("Please select a valid .pdf file.", "error");
                return;
            }}
            const reader = new FileReader();
            reader.onload = () => {{
                const arrayBuffer = reader.result;
                const bytes = new Uint8Array(arrayBuffer);
                let binary = '';
                for (let i = 0; i < bytes.byteLength; i++) {{
                    binary += String.fromCharCode(bytes[i]);
                }}
                selectedPdfBytes = btoa(binary);
                fileNameEl.innerText = "✓ Attached: " + file.name + " (" + Math.round(file.size / 1024) + " KB)";
                fileNameEl.style.display = 'block';
                hideBanner();
            }};
            reader.readAsArrayBuffer(file);
        }}

        // Slider listener
        const weightSlider = document.getElementById('skill-weight');
        const weightDisplay = document.getElementById('weight-display');
        weightSlider.addEventListener('input', (e) => {{
            weightDisplay.innerText = Number(e.target.value).toFixed(2);
        }});

        // Banner functions
        function showBanner(msg, type) {{
            const b = document.getElementById('status-banner');
            b.className = 'alert-box ' + type;
            b.innerHTML = (type === 'error' ? '⚠️ ' : '✅ ') + msg;
            b.style.display = 'block';
        }}
        function hideBanner() {{
            document.getElementById('status-banner').style.display = 'none';
        }}

        // Calculate Match
        document.getElementById('btn-calculate').addEventListener('click', async () => {{
            if (!pyodideInstance) return;

            hideBanner();
            const btn = document.getElementById('btn-calculate');
            btn.innerText = "⏳ Computing Alignment...";
            btn.disabled = true;

            try {{
                const rawText = document.getElementById('resume-text').value;
                const jobDesc = document.getElementById('job-text').value;
                const weight = parseFloat(weightSlider.value);
                const pdfB64 = selectedPdfBytes || "";

                const pyCall = `evaluate_match_json(${{JSON.stringify(pdfB64)}}, ${{JSON.stringify(rawText)}}, ${{JSON.stringify(jobDesc)}}, ${{weight}})`;
                const jsonStr = await pyodideInstance.runPythonAsync(pyCall);
                const res = JSON.parse(jsonStr);

                renderResults(res, weight);
                showBanner("Analysis Completed: Match score and breakdown calculated.", "success");

            }} catch (err) {{
                console.error("Match error:", err);
                showBanner(err.message || String(err), "error");
            }} finally {{
                btn.innerText = "✨ Calculate Match";
                btn.disabled = false;
            }}
        }});

        function renderResults(res, weight) {{
            const score = res.final_score;
            const textSim = res.text_similarity;
            const skillCov = res.skill_coverage;

            // Score Card
            const scoreCard = document.getElementById('score-card');
            const scoreNumber = document.getElementById('score-number');
            const scoreLabel = document.getElementById('score-label');
            const scoreSubtext = document.getElementById('score-subtext');

            scoreNumber.innerText = score.toFixed(1) + "%";
            let colorClass = "score-low";
            let label = "Low Alignment";
            if (score >= 75) {{
                colorClass = "score-high";
                label = "Strong Alignment";
            }} else if (score >= 50) {{
                colorClass = "score-mid";
                label = "Moderate Alignment";
            }}
            scoreCard.className = "score-card " + colorClass;
            scoreLabel.innerText = label;
            scoreSubtext.innerText = "Weighted Blend: " + Math.round((1 - weight) * 100) + "% TF-IDF & " + Math.round(weight * 100) + "% Skills";

            // Metric numbers
            document.getElementById('text-sim-val').innerText = textSim.toFixed(1) + "%";
            document.getElementById('skill-cov-val').innerText = skillCov.toFixed(1) + "%";

            // Matching Skills
            const matchContainer = document.getElementById('matching-badges');
            if (res.matching_skills && res.matching_skills.length > 0) {{
                matchContainer.innerHTML = res.matching_skills.map(s => `<span class="badge badge-match">✓ ${{s}}</span>`).join('');
            }} else {{
                matchContainer.innerHTML = '<p class="empty-hint">None of the recognized skills were matched.</p>';
            }}

            // Missing Skills
            const missingContainer = document.getElementById('missing-badges');
            if (res.missing_skills && res.missing_skills.length > 0) {{
                missingContainer.innerHTML = res.missing_skills.map(s => `<span class="badge badge-missing">✗ ${{s}}</span>`).join('');
            }} else {{
                missingContainer.innerHTML = '<p class="empty-hint" style="color:#059669; font-weight:600;">No missing skills! Resume covers all recognized skills.</p>';
            }}

            // Suggestions
            const suggBox = document.getElementById('suggestions-box');
            if (res.suggestions && res.suggestions.length > 0) {{
                suggBox.innerHTML = res.suggestions.map((sug, idx) => `<strong>${{idx + 1}}.</strong> ${{sug}}`).join('<br><br>');
            }} else {{
                suggBox.innerText = "Great match! No immediate keyword improvements detected.";
            }}
        }}

        // Example loader
        document.getElementById('btn-example').addEventListener('click', () => {{
            selectedPdfBytes = SAMPLE_PDF_B64;
            fileNameEl.innerText = "✓ Attached: sample_resume.pdf (Sample Data)";
            fileNameEl.style.display = 'block';
            document.getElementById('resume-text').value = SAMPLE_RESUME_TXT;
            document.getElementById('job-text').value = SAMPLE_JOB;
            hideBanner();
        }});

        // Clear
        document.getElementById('btn-clear').addEventListener('click', () => {{
            selectedPdfBytes = null;
            fileInput.value = "";
            fileNameEl.style.display = 'none';
            document.getElementById('resume-text').value = "";
            document.getElementById('job-text').value = "";
            weightSlider.value = 0.4;
            weightDisplay.innerText = "0.40";

            document.getElementById('score-card').className = "score-card empty";
            document.getElementById('score-number').innerText = "--";
            document.getElementById('score-label').innerText = "Awaiting Submission";
            document.getElementById('score-subtext').innerText = "Submit resume and job description to evaluate";
            document.getElementById('text-sim-val').innerText = "0.0%";
            document.getElementById('skill-cov-val').innerText = "0.0%";
            document.getElementById('matching-badges').innerHTML = '<p class="empty-hint">Awaiting submission...</p>';
            document.getElementById('missing-badges').innerHTML = '<p class="empty-hint">Awaiting submission...</p>';
            document.getElementById('suggestions-box').innerText = "Submit a resume and job description to view actionable suggestions.";
            hideBanner();
        }});

        // Start initialization
        window.addEventListener('DOMContentLoaded', initPyodide);
    </script>
</body>
</html>
"""

output_path = BASE_DIR / "index.html"
output_path.write_text(html_content, encoding="utf-8")
print(f"Generated standalone WebAssembly index.html ({output_path.stat().st_size} bytes)")
