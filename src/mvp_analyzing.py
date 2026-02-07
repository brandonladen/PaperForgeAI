"""
MVP Analysis Module - Adapted from PaperForge AI's analysis stage.
Provides detailed analysis for each file before code generation.
"""
import json
import os
from pathlib import Path
from typing import Optional, Callable, Literal
from dataclasses import dataclass, field

from .utils import content_to_json, extract_code_from_content
from .mvp_planning import MVPPlan
from .config import get_openai_model, get_gemini_model


@dataclass
class FileAnalysis:
    """Detailed analysis for a single file."""
    filename: str
    purpose: str
    classes: list[dict] = field(default_factory=list)
    functions: list[dict] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    implementation_notes: str = ""


@dataclass
class MVPAnalysis:
    """Complete analysis for MVP implementation."""
    file_analyses: dict[str, FileAnalysis] = field(default_factory=dict)
    api_endpoints: list[dict] = field(default_factory=list)
    data_models: list[dict] = field(default_factory=list)
    shared_components: list[str] = field(default_factory=list)


# =============================================================================
# Analysis Prompts
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an expert software architect analyzing how to implement specific modules in a web MVP.
Your task is to provide detailed implementation guidance for individual files.

Guidelines:
1. Be specific about classes, methods, and their parameters
2. Include type hints in Python, JSDoc in JavaScript
3. Consider error handling and edge cases
4. Reference the paper's methodology when applicable
5. Keep implementations minimal but complete"""


FILE_ANALYSIS_PROMPT = """Based on the following context, provide a detailed analysis for implementing: {filename}

## Paper Overview
{paper_overview}

## Architecture Design
{architecture}

## Logic Analysis from Plan
{logic_analysis}

## File Dependencies
This file depends on: {dependencies}
This file is depended on by: {dependents}

---

Provide a detailed implementation guide for {filename}:

1. **Purpose**: What this file does and its role in the system
2. **Classes/Functions**: List each class/function with:
   - Name and signature
   - Parameters with types
   - Return type
   - Brief description
3. **Imports**: What to import
4. **Implementation Notes**: Key details, algorithms, or patterns to use
5. **Error Handling**: What errors to handle

Format your response as structured text that will guide code generation."""


ENDPOINT_ANALYSIS_PROMPT = """Analyze the API endpoints needed for this MVP.

## Paper Overview
{paper_overview}

## Architecture Design
{architecture}

---

Define the API endpoints in detail:

For each endpoint provide:
1. HTTP Method and Path (e.g., POST /run)
2. Purpose
3. Request body schema (JSON)
4. Response schema (JSON)
5. Error responses
6. Example request/response

Format as a structured list that can guide implementation."""


class MVPAnalyzer:
    """
    Detailed analyzer for MVP implementation.
    Provides file-by-file analysis before code generation.
    """

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

    def analyze(
        self,
        plan: MVPPlan,
        output_dir: Optional[Path] = None,
        on_progress: Optional[Callable[[str, str], None]] = None
    ) -> MVPAnalysis:
        """
        Create detailed analysis for each file in the MVP plan.

        Args:
            plan: The MVP plan from planning stage
            output_dir: Optional directory to save artifacts
            on_progress: Callback for progress updates

        Returns:
            Complete MVPAnalysis
        """
        def log(stage: str, msg: str):
            if on_progress:
                on_progress(stage, msg)

        analysis = MVPAnalysis()

        # Build dependency graph
        dependencies, dependents = self._build_dependency_graph(plan)

        # Analyze API endpoints first
        log("analyzing", "Analyzing API endpoints...")
        analysis.api_endpoints = self._analyze_endpoints(plan)

        # Analyze each file
        for i, filename in enumerate(plan.task_list):
            if filename.endswith(('.yaml', '.yml', '.json', '.md')):
                continue

            log("analyzing", f"Analyzing {filename} ({i+1}/{len(plan.task_list)})...")

            file_analysis = self._analyze_file(
                filename=filename,
                plan=plan,
                dependencies=dependencies.get(filename, []),
                dependents=dependents.get(filename, [])
            )

            analysis.file_analyses[filename] = file_analysis

        # Save artifacts
        if output_dir:
            self._save_artifacts(output_dir, analysis, plan)

        log("analyzing", "Analysis complete.")
        return analysis

    def _build_dependency_graph(self, plan: MVPPlan) -> tuple[dict, dict]:
        """Build file dependency graph from logic analysis."""
        dependencies = {}  # file -> list of files it depends on
        dependents = {}    # file -> list of files that depend on it

        for filename in plan.task_list:
            dependencies[filename] = []
            dependents[filename] = []

        # Parse dependencies from logic analysis
        for item in plan.logic_analysis:
            if len(item) >= 2:
                filename = item[0]
                description = item[1].lower()

                # Look for import/dependency mentions
                for other_file in plan.task_list:
                    if other_file != filename and other_file.replace('.py', '').replace('.js', '') in description:
                        if other_file not in dependencies.get(filename, []):
                            dependencies.setdefault(filename, []).append(other_file)
                            dependents.setdefault(other_file, []).append(filename)

        return dependencies, dependents

    def _analyze_endpoints(self, plan: MVPPlan) -> list[dict]:
        """Analyze API endpoints from plan."""
        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": ENDPOINT_ANALYSIS_PROMPT.format(
                paper_overview=plan.overview[:2000],
                architecture=plan.implementation_approach
            )}
        ]

        content = self._api_call(messages)

        # Parse endpoints from response
        endpoints = self._parse_endpoints(content)
        return endpoints

    def _analyze_file(
        self,
        filename: str,
        plan: MVPPlan,
        dependencies: list[str],
        dependents: list[str]
    ) -> FileAnalysis:
        """Generate detailed analysis for a single file."""

        # Find logic analysis for this file
        logic_desc = ""
        for item in plan.logic_analysis:
            if len(item) >= 2 and item[0] == filename:
                logic_desc = item[1]
                break

        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": FILE_ANALYSIS_PROMPT.format(
                filename=filename,
                paper_overview=plan.overview[:1500],
                architecture=f"Implementation: {plan.implementation_approach}\n\nFiles: {plan.file_list}",
                logic_analysis=logic_desc,
                dependencies=", ".join(dependencies) if dependencies else "None",
                dependents=", ".join(dependents) if dependents else "None"
            )}
        ]

        content = self._api_call(messages)

        # Parse the analysis
        file_analysis = self._parse_file_analysis(filename, content, logic_desc)
        return file_analysis

    def _api_call(self, messages: list[dict]) -> str:
        """Make API call using configured provider."""
        if self.provider == "openai":
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            return completion.choices[0].message.content
        else:  # gemini
            from google import genai
            # Combine messages into a single prompt for Gemini
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"Instructions: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            full_prompt = "\n\n".join(prompt_parts)
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2000
                )
            )
            return response.text

    def _parse_endpoints(self, content: str) -> list[dict]:
        """Parse API endpoints from analysis response."""
        endpoints = []

        # Default endpoints for web MVP
        default_endpoints = [
            {
                "method": "GET",
                "path": "/",
                "purpose": "Landing page with API info",
                "response": {"type": "html"}
            },
            {
                "method": "GET",
                "path": "/health",
                "purpose": "Health check",
                "response": {"status": "healthy"}
            },
            {
                "method": "POST",
                "path": "/run",
                "purpose": "Execute main algorithm",
                "request": {"data": "object"},
                "response": {"result": "object", "id": "string"}
            },
            {
                "method": "GET",
                "path": "/results",
                "purpose": "List all results",
                "response": {"results": "array"}
            },
            {
                "method": "GET",
                "path": "/results/{id}",
                "purpose": "Get specific result",
                "response": {"result": "object"}
            }
        ]

        # Try to parse additional endpoints from content
        lines = content.split('\n')
        current_endpoint = None

        for line in lines:
            line = line.strip()
            # Look for HTTP method patterns
            for method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                if line.upper().startswith(method + ' /') or f'**{method}**' in line.upper():
                    # Extract path
                    import re
                    match = re.search(r'(GET|POST|PUT|DELETE|PATCH)\s+(/\S+)', line, re.IGNORECASE)
                    if match:
                        current_endpoint = {
                            "method": match.group(1).upper(),
                            "path": match.group(2),
                            "purpose": ""
                        }
                        endpoints.append(current_endpoint)
                    break

        # Merge with defaults, avoiding duplicates
        result = list(default_endpoints)
        for ep in endpoints:
            if not any(d['method'] == ep['method'] and d['path'] == ep['path'] for d in result):
                result.append(ep)

        return result

    def _parse_file_analysis(self, filename: str, content: str, logic_desc: str) -> FileAnalysis:
        """Parse file analysis from response."""
        analysis = FileAnalysis(
            filename=filename,
            purpose=logic_desc or f"Implementation of {filename}"
        )

        # Extract classes and functions from content
        lines = content.split('\n')
        current_section = ""

        for line in lines:
            line_lower = line.lower().strip()

            # Detect section headers
            if 'class' in line_lower and ('**' in line or '#' in line):
                current_section = "classes"
            elif 'function' in line_lower and ('**' in line or '#' in line):
                current_section = "functions"
            elif 'import' in line_lower and ('**' in line or '#' in line):
                current_section = "imports"
            elif 'purpose' in line_lower and ('**' in line or '#' in line):
                current_section = "purpose"
            elif 'note' in line_lower and ('**' in line or '#' in line):
                current_section = "notes"

            # Parse content based on section
            if current_section == "imports" and line.strip().startswith(('-', '*', '`')):
                import_text = line.strip().lstrip('-*` ').rstrip('`')
                if import_text:
                    analysis.imports.append(import_text)

            if current_section == "purpose" and not line.startswith('#'):
                if line.strip():
                    analysis.purpose = line.strip()

            if current_section == "notes" and not line.startswith('#'):
                if line.strip():
                    analysis.implementation_notes += line.strip() + "\n"

        # Extract class/function patterns
        import re

        # Look for class definitions
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            analysis.classes.append({
                "name": match.group(1),
                "methods": []
            })

        # Look for function definitions
        func_pattern = r'def\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(func_pattern, content):
            analysis.functions.append({
                "name": match.group(1),
                "params": match.group(2)
            })

        return analysis

    def _save_artifacts(self, output_dir: Path, analysis: MVPAnalysis, plan: MVPPlan):
        """Save analysis artifacts to directory."""
        output_dir = Path(output_dir)
        artifacts_dir = output_dir / "analyzing_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save each file analysis
        for filename, file_analysis in analysis.file_analyses.items():
            safe_name = filename.replace('/', '_').replace('.', '_')
            artifact_path = artifacts_dir / f"{safe_name}_analysis.json"

            with open(artifact_path, "w") as f:
                json.dump({
                    "filename": file_analysis.filename,
                    "purpose": file_analysis.purpose,
                    "classes": file_analysis.classes,
                    "functions": file_analysis.functions,
                    "imports": file_analysis.imports,
                    "implementation_notes": file_analysis.implementation_notes
                }, f, indent=2)

        # Save endpoints analysis
        with open(artifacts_dir / "api_endpoints.json", "w") as f:
            json.dump(analysis.api_endpoints, f, indent=2)


def get_file_analysis_for_coding(
    analysis: MVPAnalysis,
    filename: str
) -> str:
    """
    Format file analysis as a prompt section for code generation.
    Used to provide context when generating specific files.
    """
    if filename not in analysis.file_analyses:
        return ""

    fa = analysis.file_analyses[filename]

    sections = [
        f"## Detailed Analysis for {filename}",
        f"\n### Purpose\n{fa.purpose}",
    ]

    if fa.imports:
        sections.append(f"\n### Required Imports\n" + "\n".join(f"- {imp}" for imp in fa.imports))

    if fa.classes:
        sections.append("\n### Classes to Implement")
        for cls in fa.classes:
            sections.append(f"- {cls['name']}")

    if fa.functions:
        sections.append("\n### Functions to Implement")
        for func in fa.functions:
            sections.append(f"- {func['name']}({func.get('params', '')})")

    if fa.implementation_notes:
        sections.append(f"\n### Implementation Notes\n{fa.implementation_notes}")

    return "\n".join(sections)
