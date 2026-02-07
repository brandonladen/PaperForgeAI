"""
Simple Generator Module - Single-prompt MVP generation.
Extracts paper text and asks AI to return all code files as JSON in one call.
"""
import json
import os
from pathlib import Path
from typing import Optional, Callable, Literal
from dataclasses import dataclass

from .pdf_parser import ExtractedPaper
from .config import get_openai_model, get_gemini_model
from .templates import (
    generate_project_artifacts,
    generate_deployment_configs,
    generate_json_storage,
    inject_safety_banner
)


@dataclass
class SimpleGeneratedProject:
    """Result from simple generation."""
    project_dir: Path
    files: dict[str, str]
    project_name: str
    run_command: str
    language: str
    success: bool
    errors: list[str]


# =============================================================================
# SIMPLE MODE PROMPT - Single prompt that returns all files as JSON
# =============================================================================

SIMPLE_GENERATION_PROMPT = """You are an expert full-stack developer. Read this research paper and build a COMPLETE, WORKING web application that demonstrates the paper's concept.

## Research Paper
Title: {title}

Content:
{content}

---

## What You Must Build
Read the paper and build an appropriate web app. For example:
- If the paper is about a todo list → build a todo list app with add/edit/delete/list
- If the paper is about sorting algorithms → build an app where users can input data and see it sorted with visualization
- If the paper is about recommendation systems → build an app where users can get recommendations
- If the paper is about image processing → build an app with file upload and processing
- If the paper is about NLP → build an app with text input and analysis output

DO NOT force everything into a single generic "/run" endpoint. Build proper routes and UI that match what the paper describes.

## Technical Requirements
- Language: {language}
- Framework: {framework}
- The app must be IMMEDIATELY runnable with `{run_cmd}`
- Include a GET `/health` endpoint returning {{"status": "healthy"}}
- Use JSON file storage (no databases) - store data in a `data/` directory
- All code must be COMPLETE - no TODOs, no placeholders, no "implement here"
- Include ALL necessary imports

## UI Requirements
- The GET `/` route must serve a beautiful, modern HTML page (inline CSS, no external CDN)
- The UI must be specific to the paper's topic (NOT a generic "Run Algorithm" button)
- Include proper forms, buttons, and interactive elements appropriate to the functionality
- Use a clean, modern design with good colors and spacing
- The UI must actually work - forms must submit, results must display
- Use fetch() for API calls, show loading states, display results nicely

## File Structure
- Put ALL server code in a single `app.py` or `app.js` file (keep it simple)
- If the core logic is complex, you may put it in a separate `logic.py` or `logic.js`
- Include `requirements.txt` (Python) or `package.json` (Node.js) with dependencies

---

## Output Format
Return ONLY valid JSON in this exact format:

{{
  "project_name": "descriptive_name_in_snake_case",
  "language": "{language}",
  "run_command": "{run_cmd}",
  "files": [
    {{
      "path": "app.py",
      "content": "# Complete app code with routes, templates, and logic..."
    }},
    {{
      "path": "requirements.txt",
      "content": "flask>=3.0.0"
    }}
  ],
  "description": "One sentence describing what this app does"
}}

CRITICAL RULES:
1. Return ONLY the JSON object - no markdown, no explanation
2. Every file must contain COMPLETE, RUNNABLE code
3. The HTML UI must be specific to what the paper describes - NOT a generic API tester
4. All strings in JSON must be properly escaped (newlines as \\n, quotes as \\", backslashes as \\\\)
5. The app must start and work immediately

Generate the complete app now:"""


class SimpleGenerator:
    """
    Single-prompt generator for MVP projects.
    Sends paper to AI and gets back all files as JSON.
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

    def generate(
        self,
        paper: ExtractedPaper,
        output_dir: Path,
        language: str = "python",
        framework: str = "flask",
        on_progress: Optional[Callable[[str, str], None]] = None
    ) -> SimpleGeneratedProject:
        """
        Generate MVP project from paper using single AI prompt.

        Args:
            paper: Extracted paper content
            output_dir: Directory to write generated files
            language: Target language (python/nodejs)
            framework: Target framework (flask/express)
            on_progress: Callback for progress updates

        Returns:
            SimpleGeneratedProject with all generated files
        """
        def log(stage: str, msg: str):
            if on_progress:
                on_progress(stage, msg)

        output_dir = Path(output_dir)
        errors = []

        # Prepare paper content (truncate if too long)
        content = self._prepare_content(paper)

        log("simple", "Sending paper to AI for complete MVP generation...")

        # Build the prompt
        run_cmd = "python app.py" if language == "python" else "node app.js"
        fw = framework if language == "python" else "express"
        prompt = SIMPLE_GENERATION_PROMPT.format(
            title=paper.title or "Research Paper",
            content=content,
            language=language,
            framework=fw,
            run_cmd=run_cmd
        )

        # Call AI
        try:
            response_text = self._call_ai(prompt)
            log("simple", "Received AI response, parsing JSON...")
        except Exception as e:
            errors.append(f"AI call failed: {str(e)}")
            return SimpleGeneratedProject(
                project_dir=output_dir,
                files={},
                project_name="failed_project",
                run_command="",
                language=language,
                success=False,
                errors=errors
            )

        # Parse JSON response
        try:
            result = self._parse_response(response_text)
        except Exception as e:
            errors.append(f"Failed to parse AI response: {str(e)}")
            # Try to salvage by extracting code blocks
            result = self._fallback_parse(response_text, language)
            if not result:
                return SimpleGeneratedProject(
                    project_dir=output_dir,
                    files={},
                    project_name="failed_project",
                    run_command="",
                    language=language,
                    success=False,
                    errors=errors
                )

        # Create project directory
        project_name = self._sanitize_name(result.get("project_name", paper.title or "mvp_project"))
        project_dir = output_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "data").mkdir(exist_ok=True)
        (project_dir / "results").mkdir(exist_ok=True)

        log("simple", f"Creating project: {project_name}")

        # Write generated files
        generated_files = {}
        files_list = result.get("files", [])

        for file_info in files_list:
            file_path = file_info.get("path", "")
            file_content = file_info.get("content", "")

            if not file_path or not file_content:
                continue

            # Add safety banner for code files
            if file_path.endswith('.py'):
                file_content = inject_safety_banner(file_content, "python")
            elif file_path.endswith('.js'):
                file_content = inject_safety_banner(file_content, "javascript")

            # Write file
            full_path = project_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(file_content, encoding="utf-8")
            generated_files[file_path] = file_content

            log("simple", f"  ✓ Created {file_path}")

        # Generate supporting artifacts (Dockerfile, README, etc.)
        log("simple", "Generating project artifacts...")
        try:
            self._generate_artifacts(
                project_dir=project_dir,
                project_name=project_name,
                language=language,
                description=result.get("description", "")
            )
        except Exception as e:
            errors.append(f"Failed to generate some artifacts: {str(e)}")
            log("simple", f"⚠ Warning: {e}")

        # Ensure requirements.txt exists for Python
        if language == "python" and "requirements.txt" not in generated_files:
            req_content = "flask>=3.0.0\n"
            (project_dir / "requirements.txt").write_text(req_content, encoding="utf-8")
            generated_files["requirements.txt"] = req_content

        # Ensure package.json exists for Node.js
        if language == "nodejs" and "package.json" not in generated_files:
            package_json = {
                "name": project_name,
                "version": "1.0.0",
                "main": "app.js",
                "scripts": {"start": "node app.js"},
                "dependencies": {"express": "^4.18.2"}
            }
            pkg_content = json.dumps(package_json, indent=2)
            (project_dir / "package.json").write_text(pkg_content, encoding="utf-8")
            generated_files["package.json"] = pkg_content

        log("simple", "Simple generation complete!")

        return SimpleGeneratedProject(
            project_dir=project_dir,
            files=generated_files,
            project_name=project_name,
            run_command=result.get("run_command", f"python app.py" if language == "python" else "node app.js"),
            language=language,
            success=len(errors) == 0 and len(generated_files) > 0,
            errors=errors
        )

    def _prepare_content(self, paper: ExtractedPaper, max_chars: int = 15000) -> str:
        """Prepare paper content for prompt, truncating if needed."""
        # Prioritize methodology/algorithm sections
        priority_sections = ["abstract", "methodology", "method", "algorithm", "approach", "implementation"]

        content_parts = []
        char_count = 0

        # Add priority sections first
        for section_key in priority_sections:
            for key, text in paper.sections.items():
                if section_key in key.lower() and char_count < max_chars:
                    available = max_chars - char_count
                    section_text = text[:available] if len(text) > available else text
                    content_parts.append(f"## {key}\n{section_text}")
                    char_count += len(section_text)
                    break

        # Add remaining sections if space
        for key, text in paper.sections.items():
            if key.lower() not in priority_sections and char_count < max_chars:
                available = max_chars - char_count
                section_text = text[:available]
                content_parts.append(f"## {key}\n{section_text}")
                char_count += len(section_text)

        # Fallback to raw text if no sections
        if not content_parts:
            content_parts.append(paper.full_text[:max_chars])

        result = "\n\n".join(content_parts)

        if len(paper.full_text) > max_chars:
            result += "\n\n[Content truncated for length]"

        return result

    def _call_ai(self, prompt: str) -> str:
        """Call AI provider and return response text."""
        if self.provider == "openai":
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert full-stack developer. Build complete working web apps from research papers. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=16000
            )
            return completion.choices[0].message.content
        else:  # gemini
            from google import genai
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=16000
                )
            )
            return response.text

    def _parse_response(self, response_text: str) -> dict:
        """Parse JSON response from AI."""
        # Clean up response - remove markdown code blocks if present
        text = response_text.strip()

        # Remove ```json ... ``` wrapper
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")

        json_text = text[start:end]
        return json.loads(json_text)

    def _fallback_parse(self, response_text: str, language: str) -> Optional[dict]:
        """Fallback parsing when JSON is invalid - extract code blocks."""
        import re

        files = []

        # Look for code blocks with file indicators
        pattern = r'(?:#+\s*(?:FILE|File):?\s*)?(\S+\.(?:py|js|txt|json))\s*\n```(?:\w+)?\n(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)

        for filename, code in matches:
            files.append({
                "path": filename.strip(),
                "content": code.strip()
            })

        # If no matches, try generic code block extraction
        if not files:
            code_blocks = re.findall(r'```(?:python|javascript|js)?\n(.*?)```', response_text, re.DOTALL)
            for i, code in enumerate(code_blocks):
                if "flask" in code.lower() or "express" in code.lower() or "app.run" in code.lower():
                    ext = ".py" if "flask" in code.lower() or "def " in code else ".js"
                    files.append({"path": f"app{ext}", "content": code.strip()})
                elif "class " in code:
                    ext = ".py" if "def " in code else ".js"
                    files.append({"path": f"algorithm{ext}", "content": code.strip()})

        if files:
            return {
                "project_name": "recovered_project",
                "files": files,
                "run_command": "python app.py" if language == "python" else "node app.js"
            }

        return None

    def _generate_artifacts(
        self,
        project_dir: Path,
        project_name: str,
        language: str,
        description: str = ""
    ):
        """Generate supporting project artifacts."""
        port = 5001

        # Basic artifacts (Dockerfile, docker-compose, README, etc.)
        try:
            generate_project_artifacts(
                project_dir=project_dir,
                project_name=project_name,
                language=language,
                algorithm_name=project_name.replace("_", " ").title(),
                summary=description[:200] if description else "MVP generated from research paper",
                port=port
            )
        except Exception:
            pass  # Will be handled by fallback in app.py

        # Deployment configs
        try:
            generate_deployment_configs(
                project_dir=project_dir,
                project_name=project_name,
                language=language,
                port=port
            )
        except Exception:
            pass

        # JSON storage module
        try:
            generate_json_storage(project_dir, language)
        except Exception:
            pass

    def _sanitize_name(self, name: str) -> str:
        """Convert to valid directory name."""
        import re
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[-\s]+", "_", name)
        return name.lower()[:50]


# =============================================================================
# Convenience function
# =============================================================================

def run_simple_pipeline(
    paper: ExtractedPaper,
    output_dir: str | Path,
    api_key: str,
    language: str = "python",
    framework: str = "flask",
    model: str = None,
    provider: Literal["openai", "gemini"] = "openai",
    on_progress: Optional[Callable[[str, str], None]] = None
) -> SimpleGeneratedProject:
    """
    Run simple single-prompt generation pipeline.

    Args:
        paper: ExtractedPaper from pdf_parser
        output_dir: Output directory
        api_key: API key for the selected provider
        language: Target language (python/nodejs)
        framework: Target framework (flask/express)
        model: Model to use
        provider: AI provider (openai/gemini)
        on_progress: Progress callback

    Returns:
        SimpleGeneratedProject with complete MVP
    """
    generator = SimpleGenerator(api_key, model, provider=provider)
    return generator.generate(
        paper=paper,
        output_dir=Path(output_dir),
        language=language,
        framework=framework,
        on_progress=on_progress
    )
