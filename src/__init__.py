# PaperForge AI source modules
"""
PaperForge AI - Transform research papers into runnable MVP code.

Enhanced with PaperForge AI's multi-stage pipeline approach:
1. Planning - Create implementation plan
2. Analysis - Detailed file-by-file analysis
3. Coding - Generate code with context
4. Debugging - Auto-fix errors with SEARCH/REPLACE

Modules:
- pdf_parser: Extract text from PDF/TXT files
- analyzer: Original single-stage paper analysis
- mvp_planning: Multi-stage planning (PaperForge AI-style)
- mvp_analyzing: Detailed file analysis
- mvp_coding: Staged code generation
- simple_generator: Single-prompt MVP generation (simple mode)
- flask_generator: Python/Flask code generation
- express_generator: Node.js/Express code generation
- error_fixer: Enhanced debugging with SEARCH/REPLACE
- runner: Install dependencies and run generated projects
- storage: JSON-based project storage
- templates: Dockerfile, deployment configs, etc.
- prompts: All LLM prompts
- utils: Utilities (JSON parsing, file reading, etc.)
- deployer: Docker deployment for generated projects
"""

from .pdf_parser import extract_text_from_pdf, extract_from_text_file, ExtractedPaper
from .analyzer import PaperAnalyzer, PaperAnalysis
from .flask_generator import FlaskGenerator
from .express_generator import ExpressGenerator
from .error_fixer import ErrorFixer, ExecutionResult, FixAttempt, CodeValidator, ValidationError
from .runner import runner
from .storage import ProjectStorage

# New PaperForge AI-style modules
from .mvp_planning import MVPPlanner, MVPPlan
from .mvp_analyzing import MVPAnalyzer, MVPAnalysis
from .mvp_coding import MVPCoder, GeneratedProject, run_full_pipeline
from .simple_generator import SimpleGenerator, SimpleGeneratedProject, run_simple_pipeline
from .utils import (
    content_to_json,
    extract_code_from_content,
    read_all_files,
    read_python_files,
    sanitize_name,
    extract_main_class,
    write_file,
    clean_code_markdown,
    ensure_module_exports
)
from .deployer import DockerDeployer, DeploymentResult, PushResult, deploy_to_docker, push_to_dockerhub, build_and_push_to_dockerhub
from .config import (
    get_openai_model,
    get_gemini_model,
    get_model_for_provider,
    get_default_provider,
    get_available_providers,
    get_api_key_for_provider
)

__all__ = [
    # Original modules
    "extract_text_from_pdf",
    "extract_from_text_file",
    "ExtractedPaper",
    "PaperAnalyzer",
    "PaperAnalysis",
    "FlaskGenerator",
    "ExpressGenerator",
    "ErrorFixer",
    "ExecutionResult",
    "FixAttempt",
    "CodeValidator",
    "ValidationError",
    "runner",
    "ProjectStorage",

    # PaperForge AI-style modules
    "MVPPlanner",
    "MVPPlan",
    "MVPAnalyzer",
    "MVPAnalysis",
    "MVPCoder",
    "GeneratedProject",
    "run_full_pipeline",

    # Simple mode
    "SimpleGenerator",
    "SimpleGeneratedProject",
    "run_simple_pipeline",

    # Utilities
    "content_to_json",
    "extract_code_from_content",
    "read_all_files",
    "read_python_files",
    "sanitize_name",
    "extract_main_class",
    "write_file",
    "clean_code_markdown",
    "ensure_module_exports",

    # Deployment
    "DockerDeployer",
    "DeploymentResult",
    "PushResult",
    "deploy_to_docker",
    "push_to_dockerhub",
    "build_and_push_to_dockerhub",

    # Config
    "get_openai_model",
    "get_gemini_model",
    "get_model_for_provider",
    "get_default_provider",
    "get_available_providers",
    "get_api_key_for_provider",
]
