"""
Prompt templates for PaperForge AI.
Structured prompts for analysis, architecture planning, and code generation.
"""

# =============================================================================
# PAPER ANALYSIS PROMPTS
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect analyzing research papers to extract implementable specifications.

Your task is to analyze algorithm/systems papers and extract ONLY what can be reasonably implemented as an MVP.

IMPORTANT CONSTRAINTS:
1. Focus on the CORE algorithm only - ignore optimizations, variants, and extensions
2. If the paper lacks implementation details, make reasonable simplifying assumptions
3. Output must be practically implementable in Python or Node.js
4. No external services (databases, cloud APIs, etc.) - use in-memory/JSON storage
5. Assume single-user, local execution

You must respond with valid JSON only."""


ANALYSIS_USER_PROMPT = """Analyze this research paper and extract implementation details.

PAPER TITLE: {title}

PAPER CONTENT (may be truncated):
{content}

---

Extract and return a JSON object with these fields:

{{
  "title": "Short descriptive title",
  "summary": "2-3 sentence summary of what it does",
  "core_algorithm": "Name of the main algorithm/method",
  "algorithm_steps": ["Step 1 description", "Step 2 description", ...],
  "data_structures": [
    {{"name": "structure_name", "type": "list/dict/class/etc", "description": "what it holds"}}
  ],
  "inputs": [
    {{"name": "input_name", "type": "expected type", "description": "what the input is"}}
  ],
  "outputs": [
    {{"name": "output_name", "type": "expected type", "description": "what is returned"}}
  ],
  "dependencies": ["numpy", "etc"],
  "pseudocode": "Core algorithm in pseudocode",
  "implementation_notes": "Any important details for implementation",
  "complexity": "simple|moderate|complex",
  "missing_details": ["List of things not specified in paper that will need assumptions"]
}}

If the paper is math-heavy or lacks code details:
- Extract the conceptual algorithm steps
- Note missing implementation details in "missing_details"
- Provide reasonable defaults in "implementation_notes"

Respond with valid JSON only:"""


# =============================================================================
# ARCHITECTURE PLANNING PROMPTS
# =============================================================================

ARCHITECTURE_SYSTEM_PROMPT = """You are a senior software architect planning MVP implementations.

Your job is to create a practical, minimal architecture for implementing a research paper algorithm as a web API.

CONSTRAINTS:
1. This is an MVP - keep it minimal and functional
2. Python uses Flask, Node.js uses Express
3. No databases - use JSON files for any persistence
4. Single file for core algorithm, single file for API
5. Must be runnable with `python app.py` or `node app.js`
6. Include health check endpoint at /health
7. Main algorithm endpoint at POST /run"""


ARCHITECTURE_USER_PROMPT = """Plan the architecture for implementing this algorithm as a web API:

ALGORITHM: {algorithm_name}
SUMMARY: {summary}

STEPS:
{steps}

DATA STRUCTURES:
{data_structures}

INPUTS: {inputs}
OUTPUTS: {outputs}

TARGET LANGUAGE: {language}

---

Return a JSON architecture plan:

{{
  "files": [
    {{
      "path": "app.py",
      "purpose": "Flask API entry point",
      "key_components": ["routes", "error handling"]
    }},
    {{
      "path": "algorithm.py",
      "purpose": "Core algorithm implementation",
      "key_components": ["main class", "helper functions"]
    }}
  ],
  "api_endpoints": [
    {{
      "method": "POST",
      "path": "/run",
      "input_schema": {{}},
      "output_schema": {{}}
    }}
  ],
  "main_class": "AlgorithmClassName",
  "dependencies": ["flask"],
  "assumptions": ["List of assumptions made due to missing paper details"]
}}

Keep it minimal. Respond with JSON only:"""


# =============================================================================
# CODE GENERATION PROMPTS
# =============================================================================

PYTHON_CODE_GEN_SYSTEM = """You are an expert Python developer creating MVP implementations.

RULES:
1. Generate COMPLETE, RUNNABLE code - no placeholders or TODOs in critical paths
2. Use type hints
3. Include docstrings
4. Handle errors gracefully with try/except
5. Use JSON files for any data storage (no databases)
6. Code must work immediately when run with `python app.py`
7. Include a working /health endpoint that returns {{"status": "healthy"}}
8. Main algorithm should be in a class with a `run(input_data)` method

FORBIDDEN:
- No `pass` statements in functions that should have logic
- No `# TODO: implement` comments - implement it or use a reasonable placeholder
- No external API calls or database connections
- No environment variables except PORT"""


PYTHON_CODE_GEN_USER = """Generate a complete Flask API implementing this algorithm:

ALGORITHM: {algorithm_name}
SUMMARY: {summary}

ARCHITECTURE:
{architecture}

PSEUDOCODE:
{pseudocode}

IMPLEMENTATION NOTES:
{implementation_notes}

---

Generate TWO files:

FILE 1: algorithm.py
- Contains the main algorithm class
- Class name: {main_class}
- Must have: __init__(), run(input_data) -> result
- Include helper methods as needed
- If algorithm details are unclear, implement a reasonable working version

FILE 2: app.py
- Flask API with routes
- Imports algorithm from algorithm.py
- Endpoints: GET /, GET /health, POST /run, GET /results
- Landing page at / with basic HTML showing API info
- Store results in results/ directory as JSON

Return code in this format:

### FILE: algorithm.py
```python
[complete algorithm.py code]
```

### FILE: app.py
```python
[complete app.py code]
```"""


NODEJS_CODE_GEN_SYSTEM = """You are an expert Node.js developer creating MVP implementations.

RULES:
1. Generate COMPLETE, RUNNABLE code - no placeholders
2. Use modern ES6+ JavaScript
3. Include JSDoc comments
4. Handle errors with try/catch
5. Use JSON files for storage (no databases)
6. Code must work immediately with `node app.js`
7. Include /health endpoint returning {{"status": "healthy"}}
8. Main algorithm in a class with `run(inputData)` method

FORBIDDEN:
- No `// TODO` comments - implement it
- No external API calls or databases
- No environment variables except PORT"""


NODEJS_CODE_GEN_USER = """Generate a complete Express API implementing this algorithm:

ALGORITHM: {algorithm_name}
SUMMARY: {summary}

ARCHITECTURE:
{architecture}

PSEUDOCODE:
{pseudocode}

IMPLEMENTATION NOTES:
{implementation_notes}

---

Generate TWO files:

FILE 1: algorithm.js
- Contains the main algorithm class
- Class name: {main_class}
- Must have: constructor(), run(inputData) -> result
- Export the class with module.exports

FILE 2: app.js
- Express API with routes
- Imports algorithm from algorithm.js
- Endpoints: GET /, GET /health, POST /run, GET /results
- Landing page at / with HTML showing API info
- Store results in results/ directory as JSON

Return code in this format:

### FILE: algorithm.js
```javascript
[complete algorithm.js code]
```

### FILE: app.js
```javascript
[complete app.js code]
```"""


# =============================================================================
# CHUNKING UTILITIES
# =============================================================================

def chunk_paper_content(full_text: str, sections: dict, max_chars: int = 12000) -> str:
    """
    Intelligently chunk paper content to fit within token limits.
    Prioritizes methodology and algorithm sections.
    """
    priority_order = [
        "abstract",
        "methodology",
        "algorithm",
        "method",
        "approach",
        "implementation",
        "introduction",
        "background",
        "experiments",
        "results",
        "conclusion"
    ]

    chunks = []
    char_count = 0

    # Add sections in priority order
    for section_key in priority_order:
        for key, content in sections.items():
            if section_key in key.lower() and char_count < max_chars:
                available = max_chars - char_count
                section_text = content[:available] if len(content) > available else content
                chunks.append(f"## {key.upper()}\n{section_text}")
                char_count += len(section_text)
                break

    # If we have room, add any remaining sections
    for key, content in sections.items():
        if key.lower() not in [s.lower() for s in priority_order]:
            if char_count < max_chars:
                available = max_chars - char_count
                section_text = content[:available]
                chunks.append(f"## {key.upper()}\n{section_text}")
                char_count += len(section_text)

    # If still under limit, add raw text
    if char_count < max_chars // 2:
        available = max_chars - char_count
        chunks.append(f"## RAW TEXT\n{full_text[:available]}")

    result = "\n\n".join(chunks)

    # Add truncation notice if needed
    if len(full_text) > max_chars:
        result += "\n\n[Content truncated for length]"

    return result


def extract_code_blocks(response: str) -> dict[str, str]:
    """
    Extract code blocks from LLM response.
    Returns dict mapping filename to code content.
    """
    import re

    files = {}

    # Pattern: ### FILE: filename.ext followed by code block
    pattern = r'###\s*FILE:\s*(\S+)\s*\n```(?:\w+)?\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)

    for filename, code in matches:
        files[filename.strip()] = code.strip()

    # Fallback: look for any code blocks if no FILE markers
    if not files:
        code_blocks = re.findall(r'```(?:python|javascript|js)?\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            # Guess filenames based on content
            for i, code in enumerate(code_blocks):
                if 'flask' in code.lower() or 'express' in code.lower():
                    files['app.py' if 'flask' in code.lower() else 'app.js'] = code.strip()
                elif 'class' in code:
                    files['algorithm.py' if 'def ' in code else 'algorithm.js'] = code.strip()

    return files


# =============================================================================
# PAPERFORGE AI-STYLE MULTI-STAGE CODING PROMPTS
# =============================================================================

STAGED_CODING_SYSTEM = """You are an expert researcher and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper, an overview of the plan, a Design in JSON format, and a Task breakdown.
Your task is to write code to implement the concepts and methodologies described in the paper as a web MVP.

The code you write must be elegant, modular, and maintainable, adhering to Google-style guidelines.
Write code with triple quotes for documentation."""


STAGED_CODING_USER = """# Context
## Paper Overview
{paper_overview}

-----

## Design
{design}

-----

## Task
{task}

-----

## Configuration
```yaml
{config}
```

-----

## Previously Generated Files
{done_files}

-----

# Format example
## Code: {filename}
```{language}
## {filename}
...
```

-----

# Instruction
Based on the paper, plan, design, task and configuration specified previously, follow "Format example", write the code.

We have {done_file_list}.
Next, you must write only the "{filename}".
1. Only One file: do your best to implement THIS ONLY ONE FILE.
2. COMPLETE CODE: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
3. Set default value: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE. AVOID circular import.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN.
5. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
6. Before using a external variable/module, make sure you import it first.
7. Write out EVERY CODE DETAIL, DON'T LEAVE TODO.
8. REFER TO CONFIGURATION: you must use configuration from config if applicable.

**CRITICAL FOR algorithm.py / algorithm.js:**
If you are writing an algorithm file, the class MUST have these methods:
- `__init__(self)` / `constructor()` - Initialize the algorithm with title, description, version, author
- `get_info(self)` / `getInfo()` - Return dict with title, description, version, author
- `run(self, input_data)` / `run(inputData)` - Execute the algorithm and return result

The `run()` method is MANDATORY - the web API calls algorithm.run(data) to execute.
Example: `def run(self, input_data: dict) -> dict: ...`

{detailed_analysis}

## Code: {filename}"""


# =============================================================================
# SIMPLE FILE ANALYSIS PROMPT (for detailed per-file guidance)
# =============================================================================

SIMPLE_ANALYSIS_PROMPT = """Analyze what needs to be implemented in {filename} for this MVP:

## Paper Context
{paper_summary}

## File Role
{file_description}

## Dependencies
This file depends on: {dependencies}

---

Provide implementation guidance:
1. What classes/functions to implement
2. Key methods with signatures
3. Error handling needed
4. How it integrates with other files

Be specific and actionable."""


# =============================================================================
# ENHANCED EXTRACTION FOR MULTI-STAGE PIPELINE
# =============================================================================

def extract_code_from_staged_response(response: str, filename: str) -> str:
    """
    Extract code for a specific file from staged coding response.
    Handles PaperForge AI-style ## Code: filename format.
    """
    import re

    # Look for ## Code: filename pattern
    pattern = rf'##\s*Code:\s*{re.escape(filename)}\s*\n```(?:\w+)?\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Fallback: try to find any code block
    pattern = r'```(?:python|javascript|js)?\s*\n(.*?)```'
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


def format_done_files_for_prompt(done_files: dict[str, str], max_chars: int = 20000) -> str:
    """
    Format completed files for inclusion in coding prompt.
    Truncates if needed to stay within limits.
    """
    if not done_files:
        return "No files generated yet."

    parts = []
    total_chars = 0

    for filename, code in done_files.items():
        if filename.endswith(('.yaml', '.yml', '.json')):
            continue

        file_block = f"```\n## File: {filename}\n{code}\n```\n"

        if total_chars + len(file_block) > max_chars:
            if total_chars == 0:
                # At least include truncated first file
                truncated = code[:max_chars - 200]
                parts.append(f"```\n## File: {filename}\n{truncated}\n... [truncated]\n```\n")
            break

        parts.append(file_block)
        total_chars += len(file_block)

    return "\n".join(parts) if parts else "No code files yet."
