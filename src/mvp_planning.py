"""
MVP Planning Module - Adapted from PaperForge AI's planning stage.
Generates structured plans for web MVP implementations.
"""
import json
import os
from pathlib import Path
from typing import Optional, Callable, Literal
from dataclasses import dataclass, field

from .utils import content_to_json
from .pdf_parser import ExtractedPaper
from .config import get_openai_model, get_gemini_model


@dataclass
class MVPPlan:
    """Complete MVP implementation plan."""
    # Stage 1: Overview
    overview: str = ""
    methodology_summary: str = ""
    key_features: list[str] = field(default_factory=list)

    # Stage 2: Architecture
    implementation_approach: str = ""
    file_list: list[str] = field(default_factory=list)
    data_structures: str = ""  # Mermaid diagram
    program_flow: str = ""  # Sequence diagram

    # Stage 3: Tasks
    required_packages: list[str] = field(default_factory=list)
    logic_analysis: list[list[str]] = field(default_factory=list)
    task_list: list[str] = field(default_factory=list)
    api_spec: str = ""
    shared_knowledge: str = ""

    # Stage 4: Config
    config_yaml: str = ""

    # Metadata
    paper_title: str = ""
    target_language: str = "python"
    target_framework: str = "flask"
    unclear_items: list[str] = field(default_factory=list)


# =============================================================================
# Planning Prompts - Adapted for Web MVPs
# =============================================================================

PLAN_SYSTEM_PROMPT = """You are an expert software architect with deep understanding of web application design.
You will receive a research paper. Your task is to create a detailed plan to implement the paper's concept as a web MVP.

Instructions:
1. Align with the Paper: Follow the methods and concepts described in the paper.
2. Keep it Simple: This is an MVP - focus on core functionality, not advanced features.
3. Web-First: Design for a REST API backend with optional simple frontend.
4. No Database: Use JSON file storage for persistence.
5. Be Practical: Make the plan immediately implementable."""


PLAN_USER_PROMPT = """## Paper
{paper_content}

## Task
1. We want to implement the method described in this paper as a web MVP.
2. Before writing any code, outline a comprehensive plan that covers:
   - Key details from the paper's methodology
   - How to translate the concept into API endpoints
   - Data models needed (as JSON schemas)
   - Technical requirements and dependencies
3. The plan should be detailed enough to guide code generation.

## Requirements
- Focus on a working MVP, not production-ready code
- No external databases - use JSON files
- Target language: {language} with {framework}
- Include reasonable defaults for any unspecified details

## Instruction
Provide a thorough roadmap that makes implementing the MVP straightforward."""


ARCHITECTURE_PROMPT = """Your goal is to create a concise, usable, and complete software system design for the MVP.
Keep the architecture simple and use appropriate open-source libraries.

Based on the plan, design a web MVP architecture.

-----

## Format Example
[CONTENT]
{{
    "Implementation approach": "We will create a {framework} backend that implements the core algorithm from the paper. The API will expose endpoints for running the algorithm and managing results. JSON file storage will handle persistence.",
    "File list": [
        "app.py",
        "algorithm.py",
        "storage.py",
        "config.py"
    ],
    "Data structures and interfaces": "\\nclassDiagram\\n    class App {{\\n        +run_server()\\n    }}\\n    class Algorithm {{\\n        +__init__(config: dict)\\n        +run(input_data: dict) -> dict\\n    }}\\n    class Storage {{\\n        +save(data: dict)\\n        +load(id: str) -> dict\\n        +list_all() -> list\\n    }}\\n",
    "Program call flow": "\\nsequenceDiagram\\n    participant Client\\n    participant App\\n    participant Algorithm\\n    participant Storage\\n    Client->>App: POST /run\\n    App->>Algorithm: run(input)\\n    Algorithm-->>App: result\\n    App->>Storage: save(result)\\n    App-->>Client: response\\n",
    "Anything UNCLEAR": "Need clarification on specific parameters if not in paper."
}}
[/CONTENT]

## Nodes
- Implementation approach: <class 'str'>  # Brief strategy summary
- File list: List[str]  # All files to generate
- Data structures and interfaces: str  # Mermaid classDiagram
- Program call flow: str  # Mermaid sequenceDiagram
- Anything UNCLEAR: str  # Questions or assumptions

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT], nothing else.

## Action
Generate the architecture design following the format example."""


TASK_LIST_PROMPT = """Break down the implementation into specific tasks with dependency analysis.

-----

## Format Example
[CONTENT]
{{
    "Required packages": [
        "flask>=3.0.0",
        "typing-extensions>=4.0.0"
    ],
    "Required Other language third-party packages": [
        "No third-party dependencies required"
    ],
    "Logic Analysis": [
        [
            "storage.py",
            "Storage class with save(), load(), list_all() methods. Handles JSON file I/O in data/ directory."
        ],
        [
            "algorithm.py",
            "Main Algorithm class implementing the paper's method. Has __init__(config) and run(input_data) methods."
        ],
        [
            "app.py",
            "Flask application with routes: GET /, GET /health, POST /run, GET /results. Entry point."
        ]
    ],
    "Task list": [
        "storage.py",
        "algorithm.py",
        "app.py"
    ],
    "Full API spec": "openapi: 3.0.0\\ninfo:\\n  title: MVP API\\npaths:\\n  /run:\\n    post:\\n      summary: Run algorithm",
    "Shared Knowledge": "All modules use JSON for data serialization. Config loaded from config.json.",
    "Anything UNCLEAR": "None - proceeding with reasonable defaults."
}}
[/CONTENT]

## Nodes
- Required packages: List[str]  # Python packages in requirements.txt format
- Required Other language third-party packages: List[str]  # Non-Python deps
- Logic Analysis: List[List[str]]  # [filename, detailed description] pairs
- Task list: List[str]  # Files in dependency order
- Full API spec: str  # OpenAPI spec or empty
- Shared Knowledge: str  # Common patterns across modules
- Anything UNCLEAR: str  # Outstanding questions

## Constraint
Format: output wrapped inside [CONTENT][/CONTENT], nothing else.

## Action
Generate the task breakdown following the format example."""


CONFIG_PROMPT = """Generate a configuration file based on the paper's parameters.
Extract any mentioned values (learning rates, sizes, thresholds, etc.) or provide reasonable defaults.

ATTENTION: Use '##' to SPLIT SECTIONS. Follow the format exactly.

-----

# Format Example
## Code: config.yaml
```yaml
## config.yaml
# Configuration for {project_name}

app:
  host: "0.0.0.0"
  port: 5001
  debug: false

algorithm:
  # Parameters from the paper
  default_param: 10
  threshold: 0.5

storage:
  data_dir: "data"
  results_dir: "results"
```

-----

## Code: config.yaml
"""


class MVPPlanner:
    """
    Multi-stage planner for web MVP generation.
    Follows PaperForge AI's planning approach adapted for web applications.
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

    def create_plan(
        self,
        paper: ExtractedPaper,
        language: str = "python",
        framework: str = "flask",
        output_dir: Optional[Path] = None,
        on_progress: Optional[Callable[[str, str], None]] = None
    ) -> MVPPlan:
        """
        Create a complete MVP implementation plan through 4 stages.

        Args:
            paper: Extracted paper content
            language: Target language (python/nodejs)
            framework: Target framework (flask/express)
            output_dir: Optional directory to save artifacts
            on_progress: Callback for progress updates (stage, message)

        Returns:
            Complete MVPPlan
        """
        def log(stage: str, msg: str):
            if on_progress:
                on_progress(stage, msg)

        plan = MVPPlan(
            paper_title=paper.title,
            target_language=language,
            target_framework=framework
        )

        trajectories = []

        # Prepare paper content
        paper_content = self._prepare_paper_content(paper)

        # Stage 1: Overview Plan
        log("planning", "Creating overview plan...")
        overview_response = self._stage_overview(paper_content, language, framework, trajectories)
        plan.overview = overview_response
        plan.methodology_summary = self._extract_methodology(overview_response)

        # Stage 2: Architecture Design
        log("planning", "Designing architecture...")
        arch_response = self._stage_architecture(framework, trajectories)
        arch_data = content_to_json(arch_response)
        if arch_data:
            plan.implementation_approach = arch_data.get("Implementation approach", "")
            plan.file_list = arch_data.get("File list", [])
            plan.data_structures = arch_data.get("Data structures and interfaces", "")
            plan.program_flow = arch_data.get("Program call flow", "")
            unclear = arch_data.get("Anything UNCLEAR", "")
            if unclear:
                plan.unclear_items.append(unclear)

        # Stage 3: Task Breakdown
        log("planning", "Breaking down tasks...")
        task_response = self._stage_tasks(trajectories)
        task_data = content_to_json(task_response)
        if task_data:
            plan.required_packages = task_data.get("Required packages", [])
            plan.logic_analysis = task_data.get("Logic Analysis", [])
            plan.task_list = task_data.get("Task list", [])
            plan.api_spec = task_data.get("Full API spec", "")
            plan.shared_knowledge = task_data.get("Shared Knowledge", "")
            unclear = task_data.get("Anything UNCLEAR", "")
            if unclear and unclear != "None":
                plan.unclear_items.append(unclear)

        # Stage 4: Configuration
        log("planning", "Generating configuration...")
        config_response = self._stage_config(paper.title, trajectories)
        plan.config_yaml = self._extract_yaml_config(config_response)

        # Save artifacts if output_dir provided
        if output_dir:
            self._save_artifacts(output_dir, plan, trajectories)

        log("planning", "Planning complete.")
        return plan

    def _prepare_paper_content(self, paper: ExtractedPaper, max_chars: int = 15000) -> str:
        """Prepare paper content for prompts, prioritizing important sections."""
        priority_sections = [
            "abstract", "methodology", "method", "algorithm",
            "approach", "implementation", "introduction"
        ]

        parts = []
        char_count = 0

        # Add priority sections first
        for section in priority_sections:
            for key, content in paper.sections.items():
                if section in key.lower() and char_count < max_chars:
                    available = max_chars - char_count
                    text = content[:available] if len(content) > available else content
                    parts.append(f"## {key.upper()}\n{text}")
                    char_count += len(text)
                    break

        # Add full text if we have room
        if char_count < max_chars // 2:
            available = max_chars - char_count
            parts.append(f"## FULL TEXT\n{paper.full_text[:available]}")

        return "\n\n".join(parts)

    def _api_call(self, messages: list[dict], stage_name: str) -> str:
        """Make API call using configured provider."""
        if self.provider == "openai":
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
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
                config=genai.types.GenerateContentConfig(temperature=0.3)
            )
            return response.text

    def _stage_overview(
        self,
        paper_content: str,
        language: str,
        framework: str,
        trajectories: list
    ) -> str:
        """Stage 1: Create overview plan."""
        messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": PLAN_USER_PROMPT.format(
                paper_content=paper_content,
                language=language,
                framework=framework
            )}
        ]

        trajectories.extend(messages)
        content = self._api_call(messages, "Overview")
        trajectories.append({"role": "assistant", "content": content})

        return content

    def _stage_architecture(self, framework: str, trajectories: list) -> str:
        """Stage 2: Design architecture."""
        prompt = ARCHITECTURE_PROMPT.format(framework=framework)
        messages = trajectories + [{"role": "user", "content": prompt}]

        content = self._api_call(messages, "Architecture")
        trajectories.append({"role": "user", "content": prompt})
        trajectories.append({"role": "assistant", "content": content})

        return content

    def _stage_tasks(self, trajectories: list) -> str:
        """Stage 3: Break down tasks."""
        messages = trajectories + [{"role": "user", "content": TASK_LIST_PROMPT}]

        content = self._api_call(messages, "Tasks")
        trajectories.append({"role": "user", "content": TASK_LIST_PROMPT})
        trajectories.append({"role": "assistant", "content": content})

        return content

    def _stage_config(self, project_name: str, trajectories: list) -> str:
        """Stage 4: Generate configuration."""
        prompt = CONFIG_PROMPT.format(project_name=project_name)
        messages = trajectories + [{"role": "user", "content": prompt}]

        content = self._api_call(messages, "Config")
        trajectories.append({"role": "user", "content": prompt})
        trajectories.append({"role": "assistant", "content": content})

        return content

    def _extract_methodology(self, overview: str) -> str:
        """Extract methodology summary from overview."""
        # Look for methodology section or first few paragraphs
        lines = overview.split('\n')
        methodology = []

        in_method = False
        for line in lines:
            lower = line.lower()
            if 'method' in lower or 'approach' in lower or 'algorithm' in lower:
                in_method = True
            if in_method:
                methodology.append(line)
                if len(methodology) > 10:
                    break

        return '\n'.join(methodology) if methodology else overview[:500]

    def _extract_yaml_config(self, config_response: str) -> str:
        """Extract YAML config from response."""
        import re

        # Try to extract from code block
        match = re.search(r'```yaml\s*(.*?)```', config_response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to extract from ## Code: config.yaml section
        match = re.search(r'## Code: config\.yaml\s*(.*?)(?=##|$)', config_response, re.DOTALL)
        if match:
            return match.group(1).strip()

        return config_response

    def _save_artifacts(self, output_dir: Path, plan: MVPPlan, trajectories: list):
        """Save planning artifacts to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trajectories
        with open(output_dir / "planning_trajectories.json", "w") as f:
            json.dump(trajectories, f, indent=2)

        # Save plan summary
        plan_summary = {
            "paper_title": plan.paper_title,
            "target_language": plan.target_language,
            "target_framework": plan.target_framework,
            "file_list": plan.file_list,
            "task_list": plan.task_list,
            "required_packages": plan.required_packages,
            "unclear_items": plan.unclear_items
        }
        with open(output_dir / "plan_summary.json", "w") as f:
            json.dump(plan_summary, f, indent=2)

        # Save config
        if plan.config_yaml:
            with open(output_dir / "planning_config.yaml", "w") as f:
                f.write(plan.config_yaml)
