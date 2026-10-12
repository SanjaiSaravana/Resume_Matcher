"""
build_index_html.py
===================
Compiles python modules into index.html for Hugging Face Spaces
using the 'static' SDK with the 'gradio-lite' (WebAssembly/Pyodide) template.
"""

from __future__ import annotations

import html
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

matcher_code = (BASE_DIR / "matcher.py").read_text(encoding="utf-8")
pdf_utils_code = (BASE_DIR / "pdf_utils.py").read_text(encoding="utf-8")
mlflow_tracking_code = (BASE_DIR / "mlflow_tracking.py").read_text(encoding="utf-8")
app_code = (BASE_DIR / "app.py").read_text(encoding="utf-8")

# In app.py inside Gradio-Lite, ensure app.launch() executes
if "app.launch()" not in app_code:
    app_code += "\n\napp.launch()\n"
else:
    # Ensure app.launch() runs even if not __name__ == '__main__'
    app_code = app_code.replace(
        'if __name__ == "__main__":\n    try:\n        port = int(os.environ.get("PORT", 7860))\n        app.launch(server_name="0.0.0.0", server_port=port)\n    except Exception:\n        app.launch()',
        'try:\n    app.launch()\nexcept Exception:\n    pass'
    )

index_html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Resume–Job Description Matcher | Gradio-Lite</title>
    <!-- Gradio-Lite: Client-side WebAssembly execution via Pyodide -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.css" />
    <script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite/dist/lite.js"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            min-height: 100vh;
            background-color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
    </style>
</head>
<body>
    <gradio-lite theme="light">
        <gradio-requirements>
https://files.pythonhosted.org/packages/33/d5/d9e9b75d8dc9cf125fff16fb0cd51d864a29e8b46b6880d8808940989405/huggingface_hub-0.33.5-py3-none-any.whl
pypdf
        </gradio-requirements>

        <gradio-file name="matcher.py">
{html.escape(matcher_code, quote=False)}
        </gradio-file>

        <gradio-file name="pdf_utils.py">
{html.escape(pdf_utils_code, quote=False)}
        </gradio-file>

        <gradio-file name="mlflow_tracking.py">
{html.escape(mlflow_tracking_code, quote=False)}
        </gradio-file>

        <gradio-file name="app.py" entrypoint>
{html.escape(app_code, quote=False)}
        </gradio-file>
    </gradio-lite>
</body>
</html>
"""

output_path = BASE_DIR / "index.html"
output_path.write_text(index_html_content, encoding="utf-8")
print(f"Generated {output_path} ({output_path.stat().st_size} bytes)")
