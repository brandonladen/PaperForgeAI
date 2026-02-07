"""Flask API code generation - creates runnable web APIs from paper analysis."""
import json
from pathlib import Path
from typing import Literal
from .analyzer import PaperAnalysis
from .templates import generate_project_artifacts, inject_safety_banner, PYTHON_SAFETY_BANNER
from .utils import sanitize_name, extract_main_class, write_file, clean_code_markdown
from .config import get_openai_model, get_gemini_model


def _get_input_type(type_str: str) -> str:
    """Map paper input types to HTML input types."""
    type_lower = type_str.lower() if type_str else "text"

    if any(t in type_lower for t in ["int", "float", "number", "numeric", "rating", "score"]):
        return "number"
    elif any(t in type_lower for t in ["bool", "boolean", "flag"]):
        return "checkbox"
    elif any(t in type_lower for t in ["array", "list", "vector", "matrix", "sequence"]):
        return "array"
    elif any(t in type_lower for t in ["image", "picture", "photo", "img"]):
        return "file"
    elif any(t in type_lower for t in ["file", "document", "upload"]):
        return "file"
    elif any(t in type_lower for t in ["text", "string", "str", "name", "id", "identifier"]):
        return "text"
    elif any(t in type_lower for t in ["object", "dict", "json", "map"]):
        return "json"
    else:
        return "text"


def _generate_form_fields(inputs: list[dict]) -> str:
    """Generate HTML form fields based on paper inputs."""
    if not inputs:
        return ""

    fields_html = []
    for inp in inputs:
        name = inp.get("name", "input").replace(" ", "_").lower()
        label = inp.get("name", "Input")
        desc = inp.get("description", "")
        inp_type = _get_input_type(inp.get("type", ""))

        # Escape for HTML
        label_safe = label.replace("<", "&lt;").replace(">", "&gt;")
        desc_safe = desc.replace("<", "&lt;").replace(">", "&gt;")[:100]

        if inp_type == "number":
            fields_html.append(f'''
            <div class="form-group">
                <label for="{name}">{label_safe}</label>
                <input type="number" id="{name}" name="{name}" step="any" placeholder="{desc_safe}">
                <small>{desc_safe}</small>
            </div>''')
        elif inp_type == "checkbox":
            fields_html.append(f'''
            <div class="form-group checkbox-group">
                <label><input type="checkbox" id="{name}" name="{name}"> {label_safe}</label>
                <small>{desc_safe}</small>
            </div>''')
        elif inp_type == "array":
            fields_html.append(f'''
            <div class="form-group">
                <label for="{name}">{label_safe} (comma-separated or JSON array)</label>
                <textarea id="{name}" name="{name}" rows="2" placeholder="1, 2, 3, 4, 5 or [1,2,3,4,5]">{desc_safe}</textarea>
                <small>{desc_safe}</small>
            </div>''')
        elif inp_type == "file":
            fields_html.append(f'''
            <div class="form-group">
                <label for="{name}">{label_safe}</label>
                <input type="file" id="{name}" name="{name}" class="file-input">
                <small>{desc_safe}</small>
            </div>''')
        elif inp_type == "json":
            fields_html.append(f'''
            <div class="form-group">
                <label for="{name}">{label_safe} (JSON object)</label>
                <textarea id="{name}" name="{name}" rows="3" placeholder='{{"key": "value"}}'></textarea>
                <small>{desc_safe}</small>
            </div>''')
        else:  # text
            fields_html.append(f'''
            <div class="form-group">
                <label for="{name}">{label_safe}</label>
                <input type="text" id="{name}" name="{name}" placeholder="{desc_safe}">
                <small>{desc_safe}</small>
            </div>''')

    return "\n".join(fields_html)


def _generate_input_info(inputs: list[dict]) -> str:
    """Generate readable input documentation."""
    if not inputs:
        return '"data": [...] // Your input data'

    lines = []
    for inp in inputs:
        name = inp.get("name", "input").replace(" ", "_").lower()
        inp_type = inp.get("type", "any")
        desc = inp.get("description", "")[:50]
        lines.append(f'  "{name}": <{inp_type}>,  // {desc}')

    return "\n".join(lines)


def _generate_output_info(outputs: list[dict]) -> str:
    """Generate readable output documentation."""
    if not outputs:
        return '"result": {...} // Algorithm output'

    lines = []
    for out in outputs:
        name = out.get("name", "result").replace(" ", "_").lower()
        out_type = out.get("type", "any")
        desc = out.get("description", "")[:50]
        lines.append(f'  "{name}": <{out_type}>,  // {desc}')

    return "\n".join(lines)


def _generate_sample_input(inputs: list[dict]) -> str:
    """Generate sample JSON input based on paper inputs."""
    if not inputs:
        return '{"data": [5, 2, 8, 1, 9], "parameters": {}}'

    sample = {}
    for inp in inputs:
        name = inp.get("name", "input").replace(" ", "_").lower()
        inp_type = _get_input_type(inp.get("type", ""))

        if inp_type == "number":
            sample[name] = 0
        elif inp_type == "checkbox":
            sample[name] = True
        elif inp_type == "array":
            sample[name] = [1, 2, 3, 4, 5]
        elif inp_type == "json":
            sample[name] = {"key": "value"}
        else:
            sample[name] = "example"

    return json.dumps(sample, indent=2)


FLASK_CODE_GEN_PROMPT = """You are an expert Python developer who converts research paper algorithms into Flask web APIs.

Your code should:
1. Create a Flask REST API that exposes the algorithm
2. Have clear endpoints: POST /run (run algorithm), GET /health, GET /info
3. Accept JSON input, return JSON output
4. Include proper error handling
5. Be immediately runnable
6. Store results in JSON files

The API should have these endpoints:
- GET / - Simple HTML page showing API info and a form to test
- GET /health - Health check
- GET /info - Algorithm information
- POST /run - Execute the algorithm with input data
- GET /results - List previous results
- GET /results/<id> - Get specific result

IMPORTANT: Generate ONLY the algorithm class code for algorithm.py.
- The class MUST have a run(self, input_data) method
- Do NOT include Flask app code (that's in app.py separately)
- Do NOT include any Express/Node.js code

Generate complete, working Python code for the algorithm class."""


FLASK_USER_PROMPT = """Generate a Flask web API that implements this algorithm:

Title: {title}
Summary: {summary}
Core Algorithm: {core_algorithm}

Algorithm Steps:
{algorithm_steps}

Data Structures:
{data_structures}

Inputs: {inputs}
Outputs: {outputs}

Pseudocode:
{pseudocode}

Create a complete Flask API with:
1. Core algorithm in a separate class/module
2. Flask routes for /run, /health, /info, /results
3. A simple HTML landing page at / with a test form
4. JSON file storage for results
5. Proper error handling

Return ONLY Python code, no markdown."""


FLASK_APP_TEMPLATE = '''"""
{project_name} - Flask API
{summary}

Run with: python app.py
Access at: http://localhost:{{port}}
"""
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

from algorithm import {main_class}

app = Flask(__name__)

# Storage
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Algorithm instance
algorithm = {main_class}()

# HTML Template for landing page
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{project_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
        .subtitle {{ color: #888; margin-bottom: 2rem; font-size: 1.1rem; line-height: 1.6; }}
        .card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .card h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #4CAF50; }}
        .card h3 {{ font-size: 1rem; margin: 1rem 0 0.5rem 0; color: #64B5F6; }}
        .endpoint {{
            background: rgba(0,0,0,0.2);
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            font-family: monospace;
        }}
        .method {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-right: 0.5rem;
        }}
        .get {{ background: #2196F3; }}
        .post {{ background: #4CAF50; }}
        textarea {{
            width: 100%;
            height: 150px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 1rem;
            color: #fff;
            font-family: monospace;
            margin-bottom: 1rem;
        }}
        button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
        }}
        button:hover {{ background: #45a049; }}
        #result {{
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            font-family: monospace;
            white-space: pre-wrap;
            display: none;
        }}
        .status {{
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #4CAF50;
            border-radius: 20px;
            font-size: 0.9rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: rgba(255,152,0,0.2);
            border: 1px solid #FF9800;
            border-radius: 20px;
            font-size: 0.8rem;
            color: #FF9800;
            margin-left: 0.5rem;
        }}
        .about-section {{ color: #bbb; line-height: 1.7; }}
        .about-section p {{ margin-bottom: 0.75rem; }}
        .step-list {{
            list-style: none;
            counter-reset: step-counter;
            padding-left: 0;
        }}
        .step-list li {{
            counter-increment: step-counter;
            padding: 0.5rem 0 0.5rem 2.5rem;
            position: relative;
            color: #bbb;
        }}
        .step-list li::before {{
            content: counter(step-counter);
            position: absolute;
            left: 0;
            top: 0.5rem;
            background: #4CAF50;
            color: white;
            width: 1.5rem;
            height: 1.5rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .code-block {{
            background: rgba(0,0,0,0.3);
            border-radius: 6px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            margin: 0.5rem 0;
        }}
        .warning {{
            background: rgba(255,152,0,0.1);
            border-left: 4px solid #FF9800;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }}
        /* Dynamic form styles */
        .form-group {{
            margin-bottom: 1rem;
        }}
        .form-group label {{
            display: block;
            color: #64B5F6;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }}
        .form-group input[type="text"],
        .form-group input[type="number"],
        .form-group textarea {{
            width: 100%;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            padding: 0.75rem;
            color: #fff;
            font-size: 1rem;
        }}
        .form-group input:focus,
        .form-group textarea:focus {{
            outline: none;
            border-color: #4CAF50;
        }}
        .form-group small {{
            display: block;
            color: #666;
            margin-top: 0.25rem;
            font-size: 0.8rem;
        }}
        .form-group.checkbox-group label {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
        }}
        .form-group input[type="checkbox"] {{
            width: 1.2rem;
            height: 1.2rem;
            accent-color: #4CAF50;
        }}
        .form-group input[type="file"] {{
            padding: 0.5rem;
            background: rgba(0,0,0,0.2);
        }}
        .form-group input[type="file"]::file-selector-button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 1rem;
        }}
        .tab-buttons {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .tab-btn {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #888;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
        }}
        .tab-btn.active {{
            background: #4CAF50;
            border-color: #4CAF50;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .dynamic-form {{
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{project_name_display} <span class="badge">MVP</span></h1>
        <p class="subtitle">{summary}</p>
        <p><span class="status">🟢 Running</span></p>

        <div class="card">
            <h2>📄 About This MVP</h2>
            <div class="about-section">
                <p><strong>What is this?</strong> This is an automatically generated Minimum Viable Product (MVP) created from a research paper using AI-powered code generation.</p>
                <p><strong>Algorithm:</strong> {algorithm_name}</p>
                <p><strong>Purpose:</strong> {summary}</p>

                <div class="warning">
                    ⚠️ <strong>MVP Notice:</strong> This is a demonstration implementation for testing and exploration purposes. It may contain simplifications and is not intended for production use.
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🧪 Try It - Interactive Test</h2>

            <div class="tab-buttons">
                <button class="tab-btn active" onclick="switchTab('form')">📝 Form Input</button>
                <button class="tab-btn" onclick="switchTab('json')">🔧 Raw JSON</button>
            </div>

            <!-- Dynamic Form Tab -->
            <div id="form-tab" class="tab-content active">
                <p style="margin-bottom: 1rem; color: #888;">Fill in the fields below and click "Run Algorithm":</p>
                <div class="dynamic-form">
                    {input_form_html}
                </div>
                <button onclick="runFromForm()">Run Algorithm</button>
            </div>

            <!-- Raw JSON Tab -->
            <div id="json-tab" class="tab-content">
                <p style="margin-bottom: 1rem; color: #888;">Enter JSON input data directly:</p>
                <textarea id="json-input">{sample_input}</textarea>
                <button onclick="runFromJson()">Run Algorithm</button>
            </div>

            <div id="result"></div>

            <h3>Expected Input Format</h3>
            <div class="code-block">
{{
{input_info}
}}
            </div>

            <h3>Expected Output Format</h3>
            <div class="code-block">
{{
  "success": true,
  "result_id": "abc123",
{output_info}
}}
            </div>
        </div>

        <div class="card">
            <h2>📡 API Endpoints</h2>
            <div class="endpoint"><span class="method get">GET</span> /health - Health check</div>
            <div class="endpoint"><span class="method get">GET</span> /info - Algorithm information</div>
            <div class="endpoint"><span class="method post">POST</span> /run - Execute algorithm</div>
            <div class="endpoint"><span class="method get">GET</span> /results - List all results</div>
            <div class="endpoint"><span class="method get">GET</span> /results/&lt;id&gt; - Get specific result</div>

            <h3>Example cURL Request</h3>
            <div class="code-block">curl -X POST http://localhost:{{port}}/run \\
  -H "Content-Type: application/json" \\
  -d '{sample_curl}'</div>
        </div>

        <div class="card">
            <h2>📖 How It Works</h2>
            <p style="color: #888; margin-bottom: 1rem;">This MVP implements the <strong>{algorithm_name}</strong> algorithm:</p>
            <ol class="step-list">
                <li>Send your input data via POST /run endpoint</li>
                <li>The algorithm processes your data using the implementation</li>
                <li>Results are returned immediately and saved for later retrieval</li>
                <li>Access previous results via GET /results endpoint</li>
            </ol>
        </div>

        <div class="card" style="background: rgba(33,150,243,0.1); border-color: rgba(33,150,243,0.3);">
            <h2 style="color: #64B5F6;">🚀 Generated by PaperForge AI</h2>
            <p style="color: #888;">This MVP was automatically generated from a research paper. The code converts academic algorithms into runnable web services.</p>
        </div>
    </div>

    <script>
        function switchTab(tab) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            if (tab === 'form') {{
                document.querySelector('.tab-btn:first-child').classList.add('active');
                document.getElementById('form-tab').classList.add('active');
            }} else {{
                document.querySelector('.tab-btn:last-child').classList.add('active');
                document.getElementById('json-tab').classList.add('active');
            }}
        }}

        function collectFormData() {{
            const formData = {{}};
            const dynamicForm = document.querySelector('.dynamic-form');
            if (!dynamicForm) return {{}};

            // Collect all inputs
            dynamicForm.querySelectorAll('input, textarea').forEach(el => {{
                const name = el.name || el.id;
                if (!name) return;

                if (el.type === 'checkbox') {{
                    formData[name] = el.checked;
                }} else if (el.type === 'number') {{
                    formData[name] = el.value ? parseFloat(el.value) : 0;
                }} else if (el.type === 'file') {{
                    // File handling - will be base64 encoded
                    if (el.files && el.files[0]) {{
                        formData[name] = el.dataset.base64 || null;
                    }}
                }} else {{
                    // Try to parse as JSON array or use as string
                    const val = el.value.trim();
                    if (val.startsWith('[') || val.startsWith('{{')) {{
                        try {{
                            formData[name] = JSON.parse(val);
                        }} catch {{
                            formData[name] = val;
                        }}
                    }} else if (val.includes(',')) {{
                        // Comma-separated values - convert to array
                        formData[name] = val.split(',').map(v => {{
                            const trimmed = v.trim();
                            const num = parseFloat(trimmed);
                            return isNaN(num) ? trimmed : num;
                        }});
                    }} else {{
                        formData[name] = val;
                    }}
                }}
            }});

            return formData;
        }}

        async function runFromForm() {{
            const inputData = collectFormData();
            await sendRequest(inputData);
        }}

        async function runFromJson() {{
            const jsonInput = document.getElementById('json-input').value;
            try {{
                const inputData = JSON.parse(jsonInput);
                await sendRequest(inputData);
            }} catch (err) {{
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.textContent = 'Invalid JSON: ' + err.message;
            }}
        }}

        async function sendRequest(inputData) {{
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Running...';

            try {{
                const response = await fetch('/run', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(inputData)
                }});
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
            }} catch (err) {{
                resultDiv.textContent = 'Error: ' + err.message;
            }}
        }}

        // Handle file inputs - convert to base64
        document.querySelectorAll('input[type="file"]').forEach(input => {{
            input.addEventListener('change', function() {{
                const file = this.files[0];
                if (file) {{
                    const reader = new FileReader();
                    reader.onload = () => {{
                        this.dataset.base64 = reader.result;
                    }};
                    reader.readAsDataURL(file);
                }}
            }});
        }});
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Landing page with API info and test form."""
    return render_template_string(INDEX_HTML)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({{"status": "healthy", "timestamp": datetime.now().isoformat()}})


@app.route('/info')
def info():
    """Algorithm information."""
    return jsonify({{
        "name": "{project_name}",
        "algorithm": "{algorithm_name}",
        "summary": "{summary}",
        "version": "1.0.0",
        "endpoints": ["/health", "/info", "/run", "/results"]
    }})


@app.route('/run', methods=['POST'])
def run():
    """Execute the algorithm with provided input."""
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({{"error": "No input data provided"}}), 400

        # Check if algorithm has run method
        if not hasattr(algorithm, 'run'):
            return jsonify({{
                "error": "Algorithm does not have a 'run' method. Please check algorithm.py implementation.",
                "hint": "The algorithm class must have a 'run(self, input_data)' method."
            }}), 500

        # Run algorithm
        result = algorithm.run(input_data)

        # Save result
        result_id = str(uuid.uuid4())[:8]
        result_entry = {{
            "id": result_id,
            "input": input_data,
            "output": result,
            "timestamp": datetime.now().isoformat()
        }}

        result_file = RESULTS_DIR / f"{{result_id}}.json"
        with open(result_file, 'w') as f:
            json.dump(result_entry, f, indent=2, default=str)

        return jsonify({{
            "success": True,
            "result_id": result_id,
            "result": result
        }})

    except AttributeError as e:
        return jsonify({{
            "error": f"Algorithm method error: {{str(e)}}",
            "hint": "Make sure algorithm.py has a class with 'run(self, input_data)' method."
        }}), 500
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500


@app.route('/results')
def list_results():
    """List all saved results."""
    results = []
    for f in RESULTS_DIR.glob("*.json"):
        with open(f) as file:
            data = json.load(file)
            results.append({{
                "id": data["id"],
                "timestamp": data["timestamp"]
            }})
    return jsonify(sorted(results, key=lambda x: x["timestamp"], reverse=True))


@app.route('/results/<result_id>')
def get_result(result_id):
    """Get a specific result."""
    result_file = RESULTS_DIR / f"{{result_id}}.json"
    if not result_file.exists():
        return jsonify({{"error": "Result not found"}}), 404

    with open(result_file) as f:
        return jsonify(json.load(f))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', {port}))
    print(f"\\n{{'='*50}}")
    print(f"  {project_name}")
    print(f"  Running at: http://localhost:{{port}}")
    print(f"{{'='*50}}\\n")
    app.run(host='0.0.0.0', port=port, debug=False)
'''


class FlaskGenerator:
    """Generates Flask web API from paper analysis."""

    def __init__(
        self,
        api_key: str,
        model: str = None,
        provider: Literal["openai", "gemini"] = "openai"
    ):
        self.provider = provider
        self.api_key = api_key

        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model if model else get_openai_model()
        elif provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.model = model if model else get_gemini_model()

    def generate(self, analysis: PaperAnalysis, output_dir: str | Path, port: int = 5001) -> Path:
        """Generate a Flask API project."""
        output_dir = Path(output_dir)
        project_name = sanitize_name(analysis.title)
        project_dir = output_dir / project_name

        # Create structure
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "data").mkdir(exist_ok=True)
        (project_dir / "results").mkdir(exist_ok=True)

        # Generate algorithm code
        algorithm_code = self._generate_algorithm_code(analysis)
        main_class = extract_main_class(algorithm_code)

        # Write files
        write_file(project_dir / "algorithm.py", algorithm_code)

        # Generate dynamic form content based on paper inputs/outputs
        input_form_html = _generate_form_fields(analysis.inputs)
        if not input_form_html:
            # Fallback to generic input if no specific inputs detected
            input_form_html = '''
            <div class="form-group">
                <label for="data">Input Data (comma-separated or JSON array)</label>
                <textarea id="data" name="data" rows="2" placeholder="1, 2, 3, 4, 5 or [1,2,3,4,5]"></textarea>
                <small>Enter your data as comma-separated values or a JSON array</small>
            </div>'''

        input_info = _generate_input_info(analysis.inputs)
        output_info = _generate_output_info(analysis.outputs)
        sample_input = _generate_sample_input(analysis.inputs)

        # Generate compact sample for cURL (single line, escaped for shell)
        sample_curl = sample_input.replace('\n', '').replace('  ', '')

        # Generate Flask app
        app_code = FLASK_APP_TEMPLATE.format(
            project_name=project_name,
            project_name_display=analysis.title,
            summary=analysis.summary[:200] if analysis.summary else "",
            algorithm_name=analysis.core_algorithm,
            main_class=main_class,
            port=port,
            input_form_html=input_form_html,
            input_info=input_info,
            output_info=output_info,
            sample_input=sample_input,
            sample_curl=sample_curl
        )
        write_file(project_dir / "app.py", app_code)

        # Requirements
        write_file(project_dir / "requirements.txt", "flask>=3.0.0\n")

        # Generate all supporting artifacts (Dockerfile, run scripts, etc.)
        generate_project_artifacts(
            project_dir=project_dir,
            project_name=project_name,
            language="python",
            algorithm_name=analysis.core_algorithm,
            summary=analysis.summary or "",
            port=port
        )

        return project_dir

    def _generate_algorithm_code(self, analysis: PaperAnalysis) -> str:
        """Generate core algorithm using configured AI provider."""
        algorithm_steps = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(analysis.algorithm_steps))

        user_prompt = FLASK_USER_PROMPT.format(
            title=analysis.title,
            summary=analysis.summary,
            core_algorithm=analysis.core_algorithm,
            algorithm_steps=algorithm_steps,
            data_structures=json.dumps(analysis.data_structures, indent=2),
            inputs=json.dumps(analysis.inputs, indent=2),
            outputs=json.dumps(analysis.outputs, indent=2),
            pseudocode=analysis.pseudocode
        )

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": FLASK_CODE_GEN_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=16000
            )
            code = response.choices[0].message.content
        else:  # gemini
            from google import genai
            full_prompt = f"{FLASK_CODE_GEN_PROMPT}\n\n{user_prompt}"
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=16000
                )
            )
            code = response.text

        # Clean markdown (using shared utility)
        return clean_code_markdown(code, "python")

    def _write_readme(self, project_dir: Path, analysis: PaperAnalysis, port: int):
        readme = f"""# {analysis.title}

> Generated Flask API by PaperForge AI

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Then open: **http://localhost:{port}**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Landing page with test form |
| GET | /health | Health check |
| GET | /info | Algorithm info |
| POST | /run | Execute algorithm |
| GET | /results | List results |
| GET | /results/<id> | Get result |

## Example Usage

```bash
curl -X POST http://localhost:{port}/run \\
  -H "Content-Type: application/json" \\
  -d '{{"data": [5, 2, 8, 1, 9]}}'
```

## Algorithm: {analysis.core_algorithm}

{analysis.summary}

---
*Generated by PaperForge AI*
"""
        write_file(project_dir / "README.md", readme)
