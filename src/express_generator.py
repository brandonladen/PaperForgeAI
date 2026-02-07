"""Express.js API code generation - creates runnable Node.js web APIs."""
import json
import re
from pathlib import Path
from typing import Literal
from .analyzer import PaperAnalysis
from .templates import generate_project_artifacts, inject_safety_banner
from .utils import sanitize_name, extract_main_class, write_file, clean_code_markdown, ensure_module_exports
from .config import get_openai_model, get_gemini_model


def _generate_sample_input_js(inputs: list) -> str:
    """Generate sample JSON input based on paper inputs for Express apps."""
    if not inputs:
        return '{"data": [5, 2, 8, 1, 9], "parameters": {}}'

    sample = {}
    for inp in inputs:
        name = inp.get("name", "input").replace(" ", "_").lower()
        inp_type = inp.get("type", "").lower()

        if "int" in inp_type or "float" in inp_type or "number" in inp_type:
            sample[name] = 0
        elif "bool" in inp_type:
            sample[name] = True
        elif "list" in inp_type or "array" in inp_type:
            sample[name] = [1, 2, 3, 4, 5]
        elif "dict" in inp_type or "object" in inp_type:
            sample[name] = {"key": "value"}
        else:
            sample[name] = "example"

    return json.dumps(sample)


EXPRESS_CODE_GEN_PROMPT = """You are an expert Node.js developer who converts research paper algorithms into Express.js web APIs.

Your code should:
1. Create an Express REST API exposing the algorithm
2. Have endpoints: POST /run, GET /health, GET /info
3. Accept JSON input, return JSON output
4. Include error handling
5. Store results in JSON files
6. Be immediately runnable

CRITICAL REQUIREMENT FOR algorithm.js:
- The algorithm file MUST export the class using module.exports
- Without this export, the code WILL FAIL with "TypeError: X is not a constructor"

Generate complete, working JavaScript code with a class implementing the algorithm."""


EXPRESS_USER_PROMPT = """Generate a Node.js implementation of this algorithm:

Title: {title}
Summary: {summary}
Core Algorithm: {core_algorithm}

Algorithm Steps:
{algorithm_steps}

Data Structures: {data_structures}
Inputs: {inputs}
Outputs: {outputs}
Pseudocode: {pseudocode}

Create a complete algorithm class that:
1. Has a run(input) method that takes input data and returns results
2. Implements all the algorithm steps
3. Has proper error handling

⚠️ CRITICAL - YOU MUST INCLUDE THIS EXPORT AT THE END OF THE FILE:
```
module.exports = {{ YourClassName }};
```

Without this export statement, the app.js file cannot import the class and will crash!

Return ONLY JavaScript code for the algorithm class, no markdown.
The LAST LINE of your code MUST be the module.exports statement."""


EXPRESS_APP_TEMPLATE = '''/**
 * {project_name} - Express API
 * {summary}
 *
 * Run: npm start
 * Access: http://localhost:{port}
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const {{ v4: uuidv4 }} = require('uuid');
const {{ {main_class} }} = require('./algorithm');

const app = express();
app.use(express.json());

// Directories
const DATA_DIR = path.join(__dirname, 'data');
const RESULTS_DIR = path.join(__dirname, 'results');
[DATA_DIR, RESULTS_DIR].forEach(dir => {{
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, {{ recursive: true }});
}});

// Algorithm instance
const algorithm = new {main_class}();

// Landing page HTML
const indexHTML = `
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
    </style>
</head>
<body>
    <div class="container">
        <h1>{project_name_display} <span class="badge">MVP</span></h1>
        <p class="subtitle">{summary}</p>
        <p><span class="status">🟢 Running on Node.js</span></p>

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
            <p style="margin-bottom: 1rem; color: #888;">Enter JSON input data and click "Run Algorithm" to test:</p>
            <textarea id="input">{{"data": [5, 2, 8, 1, 9], "parameters": {{}}}}</textarea>
            <button onclick="runAlgorithm()">Run Algorithm</button>
            <div id="result"></div>

            <h3>Expected Input Format</h3>
            <div class="code-block">
{{
  "data": [...],        // Your input data (array, object, or value)
  "parameters": {{}}     // Optional algorithm parameters
}}
            </div>

            <h3>Expected Output Format</h3>
            <div class="code-block">
{{
  "success": true,
  "result_id": "abc123",
  "result": {{...}}       // Algorithm output
}}
            </div>
        </div>

        <div class="card">
            <h2>📡 API Endpoints</h2>
            <div class="endpoint"><span class="method get">GET</span> /health - Health check</div>
            <div class="endpoint"><span class="method get">GET</span> /info - Algorithm information</div>
            <div class="endpoint"><span class="method post">POST</span> /run - Execute algorithm</div>
            <div class="endpoint"><span class="method get">GET</span> /results - List all results</div>
            <div class="endpoint"><span class="method get">GET</span> /results/:id - Get specific result</div>

            <h3>Example cURL Request</h3>
            <div class="code-block">curl -X POST http://localhost:{port}/run \\
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
        async function runAlgorithm() {{
            const input = document.getElementById('input').value;
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.textContent = 'Running...';

            try {{
                const response = await fetch('/run', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: input
                }});
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
            }} catch (err) {{
                resultDiv.textContent = 'Error: ' + err.message;
            }}
        }}
    </script>
</body>
</html>
`;

// Routes
app.get('/', (req, res) => {{
    res.send(indexHTML);
}});

app.get('/health', (req, res) => {{
    res.json({{ status: 'healthy', timestamp: new Date().toISOString() }});
}});

app.get('/info', (req, res) => {{
    res.json({{
        name: '{project_name}',
        algorithm: '{algorithm_name}',
        summary: '{summary}',
        version: '1.0.0',
        runtime: 'Node.js',
        endpoints: ['/health', '/info', '/run', '/results']
    }});
}});

app.post('/run', async (req, res) => {{
    try {{
        const inputData = req.body;
        if (!inputData || Object.keys(inputData).length === 0) {{
            return res.status(400).json({{ error: 'No input data provided' }});
        }}

        // Check if algorithm has run method
        if (typeof algorithm.run !== 'function') {{
            return res.status(500).json({{
                error: "Algorithm does not have a 'run' method. Please check algorithm.js implementation.",
                hint: "The algorithm class must have a 'run(inputData)' method."
            }});
        }}

        // Run algorithm
        const result = await algorithm.run(inputData);

        // Save result
        const resultId = uuidv4().slice(0, 8);
        const resultEntry = {{
            id: resultId,
            input: inputData,
            output: result,
            timestamp: new Date().toISOString()
        }};

        const resultFile = path.join(RESULTS_DIR, `${{resultId}}.json`);
        fs.writeFileSync(resultFile, JSON.stringify(resultEntry, null, 2));

        res.json({{
            success: true,
            result_id: resultId,
            result: result
        }});

    }} catch (err) {{
        res.status(500).json({{ error: err.message }});
    }}
}});

app.get('/results', (req, res) => {{
    const files = fs.readdirSync(RESULTS_DIR).filter(f => f.endsWith('.json'));
    const results = files.map(f => {{
        const data = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, f)));
        return {{ id: data.id, timestamp: data.timestamp }};
    }});
    res.json(results.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)));
}});

app.get('/results/:id', (req, res) => {{
    const resultFile = path.join(RESULTS_DIR, `${{req.params.id}}.json`);
    if (!fs.existsSync(resultFile)) {{
        return res.status(404).json({{ error: 'Result not found' }});
    }}
    res.json(JSON.parse(fs.readFileSync(resultFile)));
}});

// Start server
const PORT = process.env.PORT || {port};
app.listen(PORT, '0.0.0.0', () => {{
    console.log('\\n' + '='.repeat(50));
    console.log('  {project_name}');
    console.log(`  Running at: http://localhost:${{PORT}}`);
    console.log('='.repeat(50) + '\\n');
}});
'''


class ExpressGenerator:
    """Generates Express.js web API from paper analysis."""

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
        """Generate an Express.js API project."""
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

        # Write algorithm
        write_file(project_dir / "algorithm.js", algorithm_code)

        # Generate sample input for cURL example
        sample_curl = _generate_sample_input_js(analysis.inputs)

        # Generate Express app
        app_code = EXPRESS_APP_TEMPLATE.format(
            project_name=project_name,
            project_name_display=analysis.title,
            summary=(analysis.summary[:150] if analysis.summary else "").replace("'", "\\'"),
            algorithm_name=analysis.core_algorithm.replace("'", "\\'"),
            main_class=main_class,
            port=port,
            sample_curl=sample_curl
        )
        write_file(project_dir / "app.js", app_code)

        # Package.json
        self._write_package_json(project_dir, project_name, analysis)

        # Generate all supporting artifacts (Dockerfile, run scripts, etc.)
        generate_project_artifacts(
            project_dir=project_dir,
            project_name=project_name,
            language="nodejs",
            algorithm_name=analysis.core_algorithm,
            summary=analysis.summary or "",
            port=port
        )

        return project_dir

    def _generate_algorithm_code(self, analysis: PaperAnalysis) -> str:
        """Generate algorithm using configured AI provider."""
        algorithm_steps = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(analysis.algorithm_steps))

        user_prompt = EXPRESS_USER_PROMPT.format(
            title=analysis.title,
            summary=analysis.summary,
            core_algorithm=analysis.core_algorithm,
            algorithm_steps=algorithm_steps,
            data_structures=json.dumps(analysis.data_structures),
            inputs=json.dumps(analysis.inputs),
            outputs=json.dumps(analysis.outputs),
            pseudocode=analysis.pseudocode
        )

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXPRESS_CODE_GEN_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=16000
            )
            code = response.choices[0].message.content
        else:  # gemini
            from google import genai
            full_prompt = f"{EXPRESS_CODE_GEN_PROMPT}\n\n{user_prompt}"
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=16000
                )
            )
            code = response.text

        # Clean markdown and ensure exports (using shared utilities)
        code = clean_code_markdown(code, "javascript")
        code = ensure_module_exports(code, "javascript")

        return code

    def _write_package_json(self, project_dir: Path, project_name: str, analysis: PaperAnalysis):
        package = {
            "name": project_name,
            "version": "1.0.0",
            "description": analysis.summary[:100] if analysis.summary else "Generated API",
            "main": "app.js",
            "scripts": {
                "start": "node app.js"
            },
            "dependencies": {
                "express": "^4.18.2",
                "uuid": "^9.0.0"
            }
        }
        write_file(project_dir / "package.json", json.dumps(package, indent=2))

    def _write_readme(self, project_dir: Path, analysis: PaperAnalysis, port: int):
        readme = f"""# {analysis.title}

> Generated Express.js API by PaperForge AI

## Quick Start

```bash
npm install
npm start
```

Then open: **http://localhost:{port}**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Landing page |
| GET | /health | Health check |
| GET | /info | Algorithm info |
| POST | /run | Execute algorithm |
| GET | /results | List results |

## Algorithm: {analysis.core_algorithm}

{analysis.summary}

---
*Generated by PaperForge AI*
"""
        write_file(project_dir / "README.md", readme)
