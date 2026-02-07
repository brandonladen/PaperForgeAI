"""
MVP Coding Module - Adapted from PaperForge AI's coding stage.
Generates code file-by-file with context from previous files.
"""
import json
import os
from pathlib import Path
from typing import Optional, Callable, Literal
from dataclasses import dataclass

from .utils import extract_code_from_content
from .mvp_planning import MVPPlan
from .mvp_analyzing import MVPAnalysis, get_file_analysis_for_coding
from .config import get_openai_model, get_gemini_model
from .prompts import (
    STAGED_CODING_SYSTEM,
    STAGED_CODING_USER,
    extract_code_from_staged_response,
    format_done_files_for_prompt
)
from .templates import (
    generate_project_artifacts,
    generate_deployment_configs,
    generate_json_storage,
    inject_safety_banner,
    PYTHON_SAFETY_BANNER,
    NODEJS_SAFETY_BANNER
)


@dataclass
class GeneratedProject:
    """Complete generated project."""
    project_dir: Path
    files: dict[str, str]
    success: bool
    errors: list[str]


class MVPCoder:
    """
    Multi-stage coder for MVP generation.
    Generates files one-by-one with context from previous files.
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
        plan: MVPPlan,
        analysis: MVPAnalysis,
        output_dir: Path,
        on_progress: Optional[Callable[[str, str], None]] = None
    ) -> GeneratedProject:
        """
        Generate complete MVP project from plan and analysis.

        Args:
            plan: The MVP plan from planning stage
            analysis: The detailed analysis from analysis stage
            output_dir: Directory to write generated files
            on_progress: Callback for progress updates

        Returns:
            GeneratedProject with all generated files
        """
        def log(stage: str, msg: str):
            if on_progress:
                on_progress(stage, msg)

        output_dir = Path(output_dir)
        project_name = self._sanitize_name(plan.paper_title)
        project_dir = output_dir / project_name

        # Create project structure
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "data").mkdir(exist_ok=True)
        (project_dir / "results").mkdir(exist_ok=True)

        generated_files = {}
        errors = []
        done_files = {}

        # Add config to done files if available
        if plan.config_yaml:
            config_file = "config.yaml"
            (project_dir / config_file).write_text(plan.config_yaml, encoding="utf-8")
            done_files[config_file] = plan.config_yaml
            generated_files[config_file] = plan.config_yaml

        # Generate each file in task order
        for i, filename in enumerate(plan.task_list):
            # Skip config files (already handled)
            if filename.endswith(('.yaml', '.yml')):
                continue

            log("coding", f"Generating {filename} ({i+1}/{len(plan.task_list)})...")

            try:
                code = self._generate_file(
                    filename=filename,
                    plan=plan,
                    analysis=analysis,
                    done_files=done_files
                )

                if code:
                    # Add safety banner
                    lang = "python" if filename.endswith('.py') else "javascript"
                    code = inject_safety_banner(code, lang)

                    # Write file
                    file_path = project_dir / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(code, encoding="utf-8")

                    done_files[filename] = code
                    generated_files[filename] = code

                    log("coding", f"  ✓ Generated {filename}")
                else:
                    errors.append(f"Failed to generate {filename}: Empty response")
                    log("coding", f"  ✗ Failed to generate {filename}")

            except Exception as e:
                errors.append(f"Failed to generate {filename}: {str(e)}")
                log("coding", f"  ✗ Error generating {filename}: {e}")

        # Generate supporting artifacts
        log("coding", "Generating project artifacts...")
        self._generate_artifacts(
            project_dir=project_dir,
            project_name=project_name,
            plan=plan
        )

        # Save coding artifacts
        self._save_artifacts(output_dir, generated_files)

        log("coding", "Code generation complete.")

        return GeneratedProject(
            project_dir=project_dir,
            files=generated_files,
            success=len(errors) == 0,
            errors=errors
        )

    def _generate_file(
        self,
        filename: str,
        plan: MVPPlan,
        analysis: MVPAnalysis,
        done_files: dict[str, str]
    ) -> str:
        """Generate a single file with context."""

        # Get detailed analysis for this file
        detailed_analysis = get_file_analysis_for_coding(analysis, filename)

        # Find logic description from plan
        file_description = ""
        for item in plan.logic_analysis:
            if len(item) >= 2 and item[0] == filename:
                file_description = item[1]
                break

        # Build the prompt
        lang_hint = "python" if filename.endswith('.py') else "javascript"

        messages = [
            {"role": "system", "content": STAGED_CODING_SYSTEM},
            {"role": "user", "content": STAGED_CODING_USER.format(
                paper_overview=plan.overview[:3000],
                design=f"Implementation: {plan.implementation_approach}\n\nFiles: {plan.file_list}\n\nData Structures:\n{plan.data_structures}",
                task=f"Task list: {plan.task_list}\n\nLogic:\n" + "\n".join(
                    f"- {item[0]}: {item[1]}" for item in plan.logic_analysis if len(item) >= 2
                ),
                config=plan.config_yaml[:1000] if plan.config_yaml else "# No config",
                done_files=format_done_files_for_prompt(done_files),
                done_file_list=list(done_files.keys()),
                filename=filename,
                language=lang_hint,
                detailed_analysis=detailed_analysis or f"Implement {filename}: {file_description}"
            )}
        ]

        # Make API call
        if self.provider == "openai":
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=4000
            )
            response_content = completion.choices[0].message.content
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

            full_prompt = "\n\n".join(prompt_parts)
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=4000
                )
            )
            response_content = response.text

        # Extract code
        code = extract_code_from_staged_response(response_content, filename)

        if not code:
            # Fallback: try generic extraction
            code = extract_code_from_content(response_content)

        return code

    def _generate_artifacts(
        self,
        project_dir: Path,
        project_name: str,
        plan: MVPPlan
    ):
        """Generate all supporting project artifacts."""

        language = plan.target_language
        port = 5001  # Default port for generated MVPs

        # Basic artifacts (Dockerfile, docker-compose, README, etc.)
        generate_project_artifacts(
            project_dir=project_dir,
            project_name=project_name,
            language=language,
            algorithm_name=plan.paper_title,
            summary=plan.methodology_summary[:200] if plan.methodology_summary else "",
            port=port
        )

        # Deployment configs (Railway, Fly.io, Render)
        generate_deployment_configs(
            project_dir=project_dir,
            project_name=project_name,
            language=language,
            port=port
        )

        # JSON storage module
        generate_json_storage(project_dir, language)

        # Requirements file
        if language == "python":
            requirements = plan.required_packages or ["flask>=3.0.0"]
            if not any("flask" in r.lower() for r in requirements):
                requirements.insert(0, "flask>=3.0.0")
            (project_dir / "requirements.txt").write_text(
                "\n".join(requirements) + "\n",
                encoding="utf-8"
            )
        else:
            # package.json
            package_json = {
                "name": project_name,
                "version": "1.0.0",
                "description": f"MVP implementation of {plan.paper_title}",
                "main": "app.js",
                "scripts": {
                    "start": "node app.js"
                },
                "dependencies": {
                    "express": "^4.18.2"
                }
            }
            (project_dir / "package.json").write_text(
                json.dumps(package_json, indent=2),
                encoding="utf-8"
            )

    def _save_artifacts(self, output_dir: Path, generated_files: dict[str, str]):
        """Save coding artifacts for debugging/inspection."""
        artifacts_dir = output_dir / "coding_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save generated files summary
        summary = {
            "files": list(generated_files.keys())
        }
        (artifacts_dir / "coding_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8"
        )

        # Save each file's content
        for filename, content in generated_files.items():
            safe_name = filename.replace('/', '_').replace('.', '_')
            (artifacts_dir / f"{safe_name}_coding.txt").write_text(
                content,
                encoding="utf-8"
            )

    def _sanitize_name(self, name: str) -> str:
        """Convert to valid directory name."""
        import re
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[-\s]+", "_", name)
        return name.lower()[:50]


# =============================================================================
# Full Pipeline Function
# =============================================================================

def run_full_pipeline(
    paper,
    output_dir: str | Path,
    api_key: str,
    language: str = "python",
    framework: str = "flask",
    model: str = None,
    provider: Literal["openai", "gemini"] = "openai",
    on_progress: Optional[Callable[[str, str], None]] = None
) -> GeneratedProject:
    """
    Run the complete PaperForge AI-style pipeline:
    1. Planning
    2. Analysis
    3. Coding

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
        GeneratedProject with complete MVP
    """
    from .mvp_planning import MVPPlanner
    from .mvp_analyzing import MVPAnalyzer

    output_dir = Path(output_dir)

    def log(stage: str, msg: str):
        if on_progress:
            on_progress(stage, msg)

    # Stage 1: Planning
    log("pipeline", "Stage 1/3: Planning...")
    planner = MVPPlanner(api_key, model, provider=provider)
    plan = planner.create_plan(
        paper=paper,
        language=language,
        framework=framework,
        output_dir=output_dir,
        on_progress=on_progress
    )

    # Stage 2: Analysis
    log("pipeline", "Stage 2/3: Analysis...")
    analyzer = MVPAnalyzer(api_key, model, provider=provider)
    analysis = analyzer.analyze(
        plan=plan,
        output_dir=output_dir,
        on_progress=on_progress
    )

    # Stage 3: Coding
    log("pipeline", "Stage 3/3: Coding...")
    coder = MVPCoder(api_key, model, provider=provider)
    project = coder.generate(
        plan=plan,
        analysis=analysis,
        output_dir=output_dir,
        on_progress=on_progress
    )

    log("pipeline", "Pipeline complete!")
    return project
