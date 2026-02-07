"""
Error Fixer Module - Enhanced debugging system adapted from PaperForge AI.
Uses SEARCH/REPLACE format for precise, minimal code fixes.
Includes proactive static analysis to catch errors before runtime.
Supports both OpenAI and Gemini providers.

SMART ERROR FIXING FEATURES:
1. Auto-add missing modules to requirements.txt for ImportError
2. Detect truncated/incomplete code and regenerate entire files
3. Fallback to full file rewrite when SEARCH/REPLACE fails
4. Code structure validation (required methods, balanced brackets)
"""
import re
import subprocess
import sys
import os
import ast
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Literal
from dataclasses import dataclass, field

from .utils import read_python_files, read_all_files
from .config import get_openai_model, get_gemini_model


# Common module name mappings (pip package name might differ from import name)
MODULE_TO_PACKAGE = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "Image": "Pillow",
    "dateutil": "python-dateutil",
    "serial": "pyserial",
    "usb": "pyusb",
    "cv": "opencv-python",
    "tensorflow": "tensorflow",
    "torch": "torch",
    "transformers": "transformers",
    "requests": "requests",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
}


# ============================================================================
# SMART ERROR FIXING HELPER FUNCTIONS
# ============================================================================

def detect_truncated_code(content: str, language: str = "python") -> Tuple[bool, str]:
    """
    Detect if code is truncated/incomplete.
    Returns (is_truncated, reason).
    """
    if not content or len(content.strip()) < 10:
        return True, "File is empty or too short"

    content = content.strip()

    if language == "python":
        # PRIMARY CHECK: If ast.parse succeeds, the file is syntactically valid.
        # Brace/bracket counting is unreliable because HTML/CSS/JS inside Python
        # triple-quoted strings have their own braces that don't need to balance
        # with Python's braces.
        try:
            ast.parse(content)
            # AST parsed successfully - file is NOT truncated
            return False, ""
        except SyntaxError as e:
            lines = content.split('\n')
            # Syntax error near end of file suggests truncation
            if e.lineno and e.lineno >= len(lines) - 3:
                return True, f"Syntax error near end of file (line {e.lineno}): likely truncated"
            # Syntax error elsewhere - still a bug but not truncation
            return False, ""

    elif language in ["nodejs", "javascript"]:
        # For JS, no AST parser available - use heuristics but be conservative.
        # Only flag if MULTIPLE indicators suggest truncation.
        indicators = []

        open_braces = content.count('{') - content.count('}')
        if open_braces > 2:  # Allow small imbalance from template literals
            indicators.append(f"Unbalanced braces (missing {open_braces} closing)")

        open_parens = content.count('(') - content.count(')')
        if open_parens > 2:
            indicators.append(f"Unbalanced parentheses (missing {open_parens} closing)")

        # Check for unclosed template literals
        backticks = content.count('`')
        if backticks % 2 != 0:
            indicators.append("Unclosed template literal (`)")

        # Only flag as truncated if we have strong evidence
        if len(indicators) >= 2:
            return True, "; ".join(indicators)
        # Single indicator with large imbalance
        if indicators and (open_braces > 5 or open_parens > 5):
            return True, indicators[0]

    return False, ""


def extract_missing_module(stderr: str) -> Optional[str]:
    """
    Extract the missing module name from ImportError/ModuleNotFoundError.
    """
    # Python patterns
    patterns = [
        r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        r"ImportError: No module named ['\"]?([^'\"'\s]+)['\"]?",
        r"ImportError: cannot import name ['\"]([^'\"]+)['\"]",
        r"No module named ['\"]?([^'\"'\s]+)['\"]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, stderr)
        if match:
            module = match.group(1)
            # Get the top-level module (e.g., 'sklearn.model_selection' -> 'sklearn')
            top_module = module.split('.')[0]
            return top_module

    # Node.js patterns
    node_patterns = [
        r"Cannot find module ['\"]([^'\"]+)['\"]",
        r"Error: Cannot find module ['\"]([^'\"]+)['\"]",
    ]

    for pattern in node_patterns:
        match = re.search(pattern, stderr)
        if match:
            return match.group(1)

    return None


def add_to_requirements(project_dir: Path, package_name: str) -> bool:
    """
    Add a package to requirements.txt and return True if successful.
    """
    req_file = project_dir / "requirements.txt"
    try:
        existing = ""
        if req_file.exists():
            existing = req_file.read_text(encoding="utf-8")

        # Check if already present
        if package_name.lower() in existing.lower():
            return False

        # Add the package
        with open(req_file, "a", encoding="utf-8") as f:
            f.write(f"\n{package_name}\n")
        return True
    except Exception:
        return False


def reinstall_requirements(project_dir: Path) -> bool:
    """
    Reinstall requirements.txt after adding new packages.
    """
    req_file = project_dir / "requirements.txt"
    if not req_file.exists():
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q", "--disable-pip-version-check"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0
    except Exception:
        return False


def validate_code_structure(content: str, filename: str, language: str = "python") -> List[str]:
    """
    Validate that code has required structure (methods, exports, etc.).
    Returns list of issues found.
    """
    issues = []

    if language == "python":
        # For algorithm.py, check for required methods
        if "algorithm" in filename.lower():
            required_methods = ["run", "get_info"]
            for method in required_methods:
                if f"def {method}" not in content:
                    issues.append(f"Missing required method: {method}()")

            # Check for class definition
            if "class " not in content:
                issues.append("Missing class definition in algorithm file")

        # For app.py, check for Flask routes
        if filename == "app.py":
            if "/health" not in content:
                issues.append("Missing /health endpoint")
            if "@app.route" not in content and "app.route" not in content:
                issues.append("Missing Flask route decorators")

    elif language in ["nodejs", "javascript"]:
        # For algorithm.js, check for exports
        if "algorithm" in filename.lower():
            if "module.exports" not in content and "export " not in content:
                issues.append("Missing module.exports in algorithm file")

        # For app.js, check for routes
        if filename == "app.js":
            if "/health" not in content:
                issues.append("Missing /health endpoint")

    return issues


# Prompt for full file regeneration (fallback when SEARCH/REPLACE fails)
REGENERATE_FILE_PROMPT = """You are an expert {language} developer. The following file has errors and needs to be completely rewritten.

FILENAME: {filename}

CURRENT BROKEN CODE:
```{lang_hint}
{current_content}
```

ERROR MESSAGE:
```
{error_msg}
```

{context}

TASK: Rewrite the COMPLETE file to fix all errors. The file must:
1. Be syntactically correct
2. Include all necessary imports
3. Implement all required functionality
4. Be complete (no TODOs, no truncation)

CRITICAL: If rewriting app.py, you MUST only call methods that ACTUALLY EXIST on the algorithm class shown in the context files. Do NOT invent methods like train(), predict(), get_recommendations() etc. unless they exist in algorithm.py. If the algorithm class only has a run(self, input_data) method, then the app MUST use that run() method for all functionality. Build the UI forms and routes around what the algorithm actually supports.

Return ONLY the complete fixed code, no explanations:"""


@dataclass
class ExecutionResult:
    """Result of running generated code."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ValidationError:
    """A static analysis validation error."""
    file: str
    error_type: str
    message: str
    suggested_fix: Optional[str] = None


@dataclass
class FixAttempt:
    """Record of a fix attempt."""
    attempt_number: int
    error: str
    fix_applied: str
    file_modified: str
    success: bool


class CodeValidator:
    """
    Static code validator that checks for common issues BEFORE running.
    Catches missing exports, import mismatches, and structural problems.
    """

    def validate(self, project_dir: Path, language: str) -> List[ValidationError]:
        """
        Validate generated code for common issues.
        Returns list of validation errors found.
        """
        errors = []

        if language == "nodejs":
            errors.extend(self._validate_nodejs(project_dir))
        else:
            errors.extend(self._validate_python(project_dir))

        return errors

    def _validate_nodejs(self, project_dir: Path) -> List[ValidationError]:
        """Validate Node.js project for common issues."""
        errors = []

        app_file = project_dir / "app.js"
        if not app_file.exists():
            errors.append(ValidationError(
                file="app.js",
                error_type="MissingFile",
                message="Main app.js file not found"
            ))
            return errors

        app_content = app_file.read_text(encoding="utf-8")

        # Find all require statements for local modules
        require_pattern = r"(?:const|let|var)\s*\{\s*([^}]+)\s*\}\s*=\s*require\(['\"]\.\/(\w+)['\"]\)"
        requires = re.findall(require_pattern, app_content)

        for imported_names, module_name in requires:
            module_file = project_dir / f"{module_name}.js"

            if not module_file.exists():
                errors.append(ValidationError(
                    file=f"{module_name}.js",
                    error_type="MissingModule",
                    message=f"Required module '{module_name}.js' not found"
                ))
                continue

            module_content = module_file.read_text(encoding="utf-8")

            # Check if module has exports
            has_module_exports = "module.exports" in module_content
            has_exports_dot = re.search(r"exports\.\w+\s*=", module_content)

            if not has_module_exports and not has_exports_dot:
                # Find the main class in the module
                class_match = re.search(r"class\s+(\w+)", module_content)
                class_name = class_match.group(1) if class_match else "ClassName"

                errors.append(ValidationError(
                    file=f"{module_name}.js",
                    error_type="MissingExport",
                    message=f"Module '{module_name}.js' has no exports. Add: module.exports = {{ {class_name} }};",
                    suggested_fix=f"\n\nmodule.exports = {{ {class_name} }};\n"
                ))
            else:
                # Check if the specific imports are exported
                imported_list = [name.strip() for name in imported_names.split(",")]
                for imp in imported_list:
                    imp = imp.strip()
                    if imp and imp not in module_content:
                        errors.append(ValidationError(
                            file=f"{module_name}.js",
                            error_type="MissingExportedName",
                            message=f"'{imp}' is imported but may not be defined in {module_name}.js"
                        ))

        # NEW: Validate code structure for all JS files
        for js_file in project_dir.glob("*.js"):
            try:
                content = js_file.read_text(encoding="utf-8")

                # Check for truncated code
                is_truncated, reason = detect_truncated_code(content, "nodejs")
                if is_truncated:
                    errors.append(ValidationError(
                        file=js_file.name,
                        error_type="TruncatedCode",
                        message=f"File appears truncated: {reason}"
                    ))

                # Check for required structure
                structure_issues = validate_code_structure(content, js_file.name, "nodejs")
                for issue in structure_issues:
                    errors.append(ValidationError(
                        file=js_file.name,
                        error_type="StructureError",
                        message=issue
                    ))
            except Exception:
                pass

        return errors

    def _validate_python(self, project_dir: Path) -> List[ValidationError]:
        """Validate Python project for common issues."""
        errors = []

        app_file = project_dir / "app.py"
        if not app_file.exists():
            errors.append(ValidationError(
                file="app.py",
                error_type="MissingFile",
                message="Main app.py file not found"
            ))
            return errors

        app_content = app_file.read_text(encoding="utf-8")

        # Find local imports
        import_pattern = r"from\s+(\w+)\s+import\s+([^\n]+)"
        imports = re.findall(import_pattern, app_content)

        for module_name, imported_names in imports:
            # Skip standard library and installed packages
            if module_name in ['flask', 'json', 'os', 'sys', 'datetime', 'pathlib', 'uuid', 'typing']:
                continue

            module_file = project_dir / f"{module_name}.py"
            if not module_file.exists():
                # Might be installed package
                continue

            module_content = module_file.read_text(encoding="utf-8")

            # Check if imported names exist in module
            imported_list = [name.strip() for name in imported_names.split(",")]
            for imp in imported_list:
                imp = imp.strip()
                # Check for class or function definition
                if imp and f"class {imp}" not in module_content and f"def {imp}" not in module_content:
                    errors.append(ValidationError(
                        file=f"{module_name}.py",
                        error_type="MissingDefinition",
                        message=f"'{imp}' is imported but not defined in {module_name}.py"
                    ))

        # NEW: Validate code structure for all Python files
        for py_file in project_dir.glob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")

                # Check for truncated code
                is_truncated, reason = detect_truncated_code(content, "python")
                if is_truncated:
                    errors.append(ValidationError(
                        file=py_file.name,
                        error_type="TruncatedCode",
                        message=f"File appears truncated: {reason}"
                    ))

                # Check for required structure
                structure_issues = validate_code_structure(content, py_file.name, "python")
                for issue in structure_issues:
                    errors.append(ValidationError(
                        file=py_file.name,
                        error_type="StructureError",
                        message=issue
                    ))
            except Exception:
                pass

        return errors

    def auto_fix(self, project_dir: Path, errors: List[ValidationError], language: str) -> List[str]:
        """
        Automatically fix validation errors where possible.
        Returns list of files that were modified.
        """
        modified_files = []

        for error in errors:
            if error.error_type == "MissingExport" and error.suggested_fix:
                file_path = project_dir / error.file
                if file_path.exists():
                    content = file_path.read_text(encoding="utf-8")
                    # Add export at the end of file
                    content = content.rstrip() + error.suggested_fix
                    file_path.write_text(content, encoding="utf-8")
                    modified_files.append(error.file)

        return modified_files


# Enhanced debugging prompt using PaperForge AI's SEARCH/REPLACE format
DEBUG_SYSTEM_PROMPT = """You are a highly capable code assistant specializing in debugging real-world code repositories. You will be provided with:
(1) a code repository (in part or in full), and
(2) one or more execution error messages generated during the execution of the repository.

Your objective is to debug the code so that it executes successfully.
This may involve identifying the root causes of the errors, modifying faulty logic or syntax, handling missing dependencies, or making other appropriate corrections.

Guidelines:
- Provide the exact lines or file changes needed to resolve the issue.
- Show only the modified lines using a unified diff format:

<<<<<<< SEARCH
    original line
=======
    corrected line
>>>>>>> REPLACE

- If multiple fixes are needed, provide them sequentially with clear separation.
- Each fix MUST be preceded by "Filename: <filename>"
- If external dependencies or environment setups are required, specify them explicitly.

Constraints:
- Do not make speculative edits without justification.
- Prioritize minimal and effective fixes that preserve the original intent of the code.
- Maintain the coding style and structure used in the original repository."""


DEBUG_USER_PROMPT = """### Code Repository
{codes}

--

### Execution Error Messages
{error_msg}

--

## Instruction
Now, you need to debug the above code so that it runs without errors. Identify the cause of the execution error and modify the code appropriately. Your output must follow the exact format as shown in the example below.

--

## Format Example
Filename: app.py
<<<<<<< SEARCH
result = model.predict(input_data)
=======
result = model(input_data)
>>>>>>> REPLACE

Filename: algorithm.py
<<<<<<< SEARCH
import nonexistent_module
=======
# import nonexistent_module  # removed: module not found
>>>>>>> REPLACE

--

## Answer
"""


class ErrorFixer:
    """
    Enhanced error fixer using PaperForge AI's SEARCH/REPLACE debugging approach.
    Provides more precise, minimal fixes instead of replacing entire files.
    Now includes proactive static validation before runtime.
    Supports both OpenAI and Gemini providers.
    """

    MAX_RETRIES = 3
    STARTUP_TIMEOUT = 30

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
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.fix_history: list[FixAttempt] = []
        self.validator = CodeValidator()

    def _api_call(self, messages: list[dict]) -> str:
        """Make API call using configured provider."""
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=16000
            )
            return response.choices[0].message.content
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
                    temperature=0.1,
                    max_output_tokens=16000
                )
            )
            return response.text

    def validate_and_fix(
        self,
        project_dir: Path,
        language: str,
        on_log: Optional[Callable[[str, str], None]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Proactively validate and fix code BEFORE running.
        This catches export/import issues that syntax checks miss.

        Returns:
            (all_fixed, list_of_modified_files)
        """
        def log(msg: str, msg_type: str = "info"):
            if on_log:
                on_log(msg, msg_type)

        log("Running static code validation...", "info")

        validation_errors = self.validator.validate(project_dir, language)

        if not validation_errors:
            log("✓ Static validation passed", "success")
            return True, []

        log(f"Found {len(validation_errors)} validation issue(s)", "warning")

        # Try to auto-fix
        modified_files = self.validator.auto_fix(project_dir, validation_errors, language)

        if modified_files:
            log(f"✓ Auto-fixed: {', '.join(modified_files)}", "success")

        # Check remaining errors
        remaining_errors = self.validator.validate(project_dir, language)

        if remaining_errors:
            for err in remaining_errors:
                log(f"⚠ {err.file}: {err.message}", "warning")
            return False, modified_files

        log("✓ All validation issues resolved", "success")
        return True, modified_files

    def run_and_fix(
        self,
        project_dir: Path,
        language: str,
        on_log: Optional[Callable[[str, str], None]] = None
    ) -> tuple[bool, Optional[subprocess.Popen], list[FixAttempt]]:
        """
        Run the project and auto-fix errors up to MAX_RETRIES times.
        Now includes:
        - Proactive static validation BEFORE runtime checks
        - Truncated code detection and regeneration
        - ImportError auto-fix (add missing modules to requirements.txt)
        - Fallback to full file regeneration when SEARCH/REPLACE fails

        Returns:
            (success, process, fix_history)
        """
        def log(msg: str, msg_type: str = "info"):
            if on_log:
                on_log(msg, msg_type)

        self.fix_history = []

        # STEP 0: Check for truncated/incomplete code files
        log("Checking for truncated code...", "info")
        truncated_files = self._check_truncated_files(project_dir, language)
        if truncated_files:
            log(f"Found {len(truncated_files)} truncated file(s), regenerating...", "warning")
            for filename, reason in truncated_files:
                log(f"  - {filename}: {reason}", "warning")
                regenerated = self._regenerate_file(project_dir, filename, reason, language)
                if regenerated:
                    self.fix_history.append(FixAttempt(
                        attempt_number=0,
                        error=f"Truncated file: {reason}",
                        fix_applied="Regenerated complete file",
                        file_modified=filename,
                        success=True
                    ))
                    log(f"✓ Regenerated {filename}", "success")

        # STEP 1: Proactive static validation
        validation_ok, fixed_files = self.validate_and_fix(project_dir, language, on_log)

        if fixed_files:
            self.fix_history.append(FixAttempt(
                attempt_number=0,
                error="Static validation errors",
                fix_applied="Added missing exports",
                file_modified=", ".join(fixed_files),
                success=True
            ))

        for attempt in range(1, self.MAX_RETRIES + 1):
            log(f"Starting runtime check {attempt}/{self.MAX_RETRIES}...", "info")

            # Try to run the project
            result = self._test_run(project_dir, language)

            if result.success:
                log(f"✓ Project validation passed!", "success")
                # Don't start the server here - runner.py handles that with the correct PORT
                return True, None, self.fix_history

            # Analyze error type
            log(f"Error detected: {result.error_type}", "warning")

            # SMART FIX 1: Handle ImportError by adding to requirements.txt
            if result.error_type == "ImportError" and language == "python":
                missing_module = extract_missing_module(result.stderr)
                if missing_module:
                    package_name = MODULE_TO_PACKAGE.get(missing_module, missing_module)
                    log(f"Missing module detected: {missing_module} -> {package_name}", "info")

                    if add_to_requirements(project_dir, package_name):
                        log(f"Added {package_name} to requirements.txt", "success")
                        if reinstall_requirements(project_dir):
                            log(f"✓ Reinstalled dependencies", "success")
                            self.fix_history.append(FixAttempt(
                                attempt_number=attempt,
                                error=f"ImportError: {missing_module}",
                                fix_applied=f"Added {package_name} to requirements.txt",
                                file_modified="requirements.txt",
                                success=True
                            ))
                            continue  # Retry after installing

            # FIX: Go straight to full file regeneration (SEARCH/REPLACE was too brittle)
            log(f"Regenerating broken file with error context...", "info")

            fallback_result = self._fallback_regenerate(
                project_dir, language, result, attempt, on_log=log
            )

            if fallback_result:
                self.fix_history.append(fallback_result)
                log(f"Regenerated {fallback_result.file_modified}", "success")
            else:
                log(f"Could not determine fix", "error")
                break

        log(f"Failed after {self.MAX_RETRIES} attempts", "error")
        return False, None, self.fix_history

    def _check_truncated_files(
        self,
        project_dir: Path,
        language: str
    ) -> List[Tuple[str, str]]:
        """
        Check all code files for truncation.
        Returns list of (filename, reason) tuples.
        """
        truncated = []
        extension = "*.py" if language == "python" else "*.js"

        for file_path in project_dir.glob(extension):
            try:
                content = file_path.read_text(encoding="utf-8")
                is_truncated, reason = detect_truncated_code(content, language)
                if is_truncated:
                    truncated.append((file_path.name, reason))
            except Exception:
                pass

        return truncated

    def _regenerate_file(
        self,
        project_dir: Path,
        filename: str,
        error_reason: str,
        language: str
    ) -> bool:
        """
        Regenerate a truncated/broken file using AI.
        """
        file_path = project_dir / filename
        if not file_path.exists():
            return False

        try:
            current_content = file_path.read_text(encoding="utf-8")

            # Build context from other files
            context = ""
            if language == "python":
                file_dict = read_python_files(project_dir)
            else:
                file_dict = read_all_files(project_dir, allowed_extensions=['.js'])

            # Include other files as context (excluding the broken one)
            for other_file, other_content in file_dict.items():
                if other_file != filename and len(other_content) < 10000:
                    lang_hint = "python" if language == "python" else "javascript"
                    context += f"\n\nRelated file ({other_file}):\n```{lang_hint}\n{other_content}\n```"

            lang_hint = "python" if language == "python" else "javascript"
            prompt = REGENERATE_FILE_PROMPT.format(
                language=language.title(),
                filename=filename,
                lang_hint=lang_hint,
                current_content=current_content[:3000],
                error_msg=error_reason,
                context=f"Context from related files:{context}" if context else ""
            )

            messages = [{"role": "user", "content": prompt}]
            response = self._api_call(messages)

            # Extract code from response
            new_content = self._extract_code_from_response(response, language)

            if new_content and len(new_content) > len(current_content) * 0.5:
                # Create backup
                backup_path = file_path.with_suffix(f".truncated.bak")
                import shutil
                shutil.copy(file_path, backup_path)

                # Write new content
                file_path.write_text(new_content, encoding="utf-8")
                return True

        except Exception:
            pass

        return False

    def _fallback_regenerate(
        self,
        project_dir: Path,
        language: str,
        error_result: ExecutionResult,
        attempt_number: int,
        on_log: Optional[Callable] = None
    ) -> Optional[FixAttempt]:
        """
        Fallback: regenerate the entire file when SEARCH/REPLACE fails.
        """
        # Try to identify which file caused the error
        error_file = self._identify_error_file(error_result.stderr, project_dir, language)

        if not error_file:
            return None

        try:
            regenerated = self._regenerate_file(
                project_dir,
                error_file,
                error_result.stderr[:1000],
                language
            )

            if regenerated:
                return FixAttempt(
                    attempt_number=attempt_number,
                    error=error_result.stderr[:500],
                    fix_applied="Full file regeneration",
                    file_modified=error_file,
                    success=True
                )

        except Exception as e:
            if on_log:
                on_log(f"Regeneration failed: {e}", "error")

        return None

    def _identify_error_file(
        self,
        stderr: str,
        project_dir: Path,
        language: str
    ) -> Optional[str]:
        """
        Identify which file caused the error from stderr.
        """
        # Look for file mentions in error
        extension = ".py" if language == "python" else ".js"

        # Pattern: File "path/to/file.py", line X
        file_pattern = rf'File ["\']([^"\']+{extension.replace(".", "\\.")})["\']'
        matches = re.findall(file_pattern, stderr)

        for match in matches:
            filename = Path(match).name
            if (project_dir / filename).exists():
                return filename

        # Pattern: at path/to/file.js:line:col
        js_pattern = rf'at\s+(?:[^\(]+\()?([^\s:]+{extension.replace(".", "\\.")})'
        matches = re.findall(js_pattern, stderr)

        for match in matches:
            filename = Path(match).name
            if (project_dir / filename).exists():
                return filename

        # Fallback: check algorithm file (commonly has issues)
        algo_file = f"algorithm{extension}"
        if (project_dir / algo_file).exists():
            return algo_file

        return None

    def _extract_code_from_response(self, response: str, language: str) -> str:
        """
        Extract code from AI response, handling various formats.
        """
        # Try markdown code block first
        lang_hint = "python" if language == "python" else "javascript"
        patterns = [
            rf"```{lang_hint}\n(.*?)```",
            r"```\n(.*?)```",
            rf"```{language}\n(.*?)```",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block, check if response looks like code
        lines = response.strip().split('\n')
        if language == "python":
            if any(line.strip().startswith(('import ', 'from ', 'def ', 'class ')) for line in lines[:10]):
                return response.strip()
        else:
            if any(line.strip().startswith(('const ', 'let ', 'var ', 'function ', 'class ')) for line in lines[:10]):
                return response.strip()

        return response.strip()

    def _test_run(self, project_dir: Path, language: str) -> ExecutionResult:
        """
        Test run the project to check for errors.
        Captures stdout/stderr and detects common error patterns.
        """
        if language == "python":
            # For Python, we do a syntax check first, then try importing
            main_file = project_dir / "app.py"

            # Syntax check all Python files
            for py_file in project_dir.glob("*.py"):
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    return ExecutionResult(
                        success=False,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        return_code=result.returncode,
                        error_type="SyntaxError",
                        error_message=result.stderr
                    )

            # Try importing to check for import errors
            result = subprocess.run(
                [sys.executable, "-c", f"import sys; sys.path.insert(0, '{project_dir}'); import app"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_dir
            )

            if result.returncode != 0:
                error_type = self._classify_error(result.stderr)
                return ExecutionResult(
                    success=False,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    return_code=result.returncode,
                    error_type=error_type,
                    error_message=result.stderr
                )

        else:  # nodejs
            main_file = project_dir / "app.js"

            # Syntax check all JS files
            for js_file in project_dir.glob("*.js"):
                result = subprocess.run(
                    ["node", "--check", str(js_file)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    error_type = self._classify_error(result.stderr)
                    return ExecutionResult(
                        success=False,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        return_code=result.returncode,
                        error_type=error_type,
                        error_message=result.stderr
                    )

            # Test require of local modules (NOT app.js - that starts the server and binds a port)
            # Only test algorithm.js and other non-server modules for export issues
            for js_file in project_dir.glob("*.js"):
                if js_file.name == "app.js":
                    continue  # skip - require('./app.js') would start the server
                test_script = f"""
try {{
    require('./{js_file.name}');
    process.exit(0);
}} catch (e) {{
    console.error(e.message);
    process.exit(1);
}}
"""
                result = subprocess.run(
                    ["node", "-e", test_script],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=project_dir
                )

                if result.returncode != 0:
                    error_type = self._classify_error(result.stderr + result.stdout)
                    return ExecutionResult(
                        success=False,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        return_code=result.returncode,
                        error_type=error_type,
                        error_message=result.stderr or result.stdout
                    )

        return ExecutionResult(
            success=True,
            stdout="",
            stderr="",
            return_code=0
        )

    def _classify_error(self, stderr: str) -> str:
        """Classify the error type from stderr."""
        stderr_lower = stderr.lower()

        if "syntaxerror" in stderr_lower:
            return "SyntaxError"
        elif "importerror" in stderr_lower or "modulenotfounderror" in stderr_lower:
            return "ImportError"
        elif "nameerror" in stderr_lower:
            return "NameError"
        elif "typeerror" in stderr_lower:
            return "TypeError"
        elif "attributeerror" in stderr_lower:
            return "AttributeError"
        elif "indentationerror" in stderr_lower:
            return "IndentationError"
        elif "cannot find module" in stderr_lower:
            return "ModuleNotFound"
        elif "unexpected token" in stderr_lower:
            return "SyntaxError"
        else:
            return "RuntimeError"

    def _attempt_search_replace_fix(
        self,
        project_dir: Path,
        language: str,
        error_result: ExecutionResult,
        attempt_number: int,
        on_log: Optional[Callable] = None
    ) -> Optional[FixAttempt]:
        """
        Attempt to fix the error using PaperForge AI's SEARCH/REPLACE approach.
        More precise than replacing the entire file.
        """
        # Get all relevant code files
        if language == "python":
            file_dict = read_python_files(project_dir)
        else:
            file_dict = read_all_files(project_dir, allowed_extensions=['.js', '.json'])

        # Build codes string for prompt
        codes = ""
        for filename, content in file_dict.items():
            lang_hint = "python" if language == "python" else "javascript"
            codes += f"```{lang_hint}\n## File name: {filename}\n{content}\n```\n\n"

        try:
            messages = [
                {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
                {"role": "user", "content": DEBUG_USER_PROMPT.format(
                    codes=codes,
                    error_msg=error_result.stderr[-3000:]
                )}
            ]

            answer = self._api_call(messages)

            # Apply the SEARCH/REPLACE fixes
            modified_files = self._parse_and_apply_changes(
                [answer],
                project_dir,
                save_num=attempt_number
            )

            if modified_files:
                return FixAttempt(
                    attempt_number=attempt_number,
                    error=error_result.stderr[:500],
                    fix_applied="SEARCH/REPLACE fix",
                    file_modified=", ".join(modified_files),
                    success=True
                )

        except Exception as e:
            if on_log:
                on_log(f"Fix attempt failed: {e}", "error")

        return None

    def _parse_and_apply_changes(
        self,
        responses: list[str],
        debug_dir: Path,
        save_num: int = 1
    ) -> list[str]:
        """
        Apply SEARCH/REPLACE edits produced by the LLM to files.
        Adapted from PaperForge AI's debugging system.
        """
        modified_files = []

        for response in responses:
            # Split into blocks per file
            file_blocks = re.split(r"Filename:\s*([^\n]+)", response)

            if len(file_blocks) < 3:
                continue

            # Process blocks per file (odd indices: filename, even indices: diff content)
            for i in range(1, len(file_blocks), 2):
                filename = file_blocks[i].strip()
                file_content_block = file_blocks[i + 1]

                filepath = debug_dir / filename

                # SEARCH/REPLACE pattern
                search_replace_pattern = (
                    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
                )
                matches = re.findall(search_replace_pattern, file_content_block, re.DOTALL)

                if not matches:
                    continue

                # Check file existence
                if not filepath.exists():
                    continue

                # Read file
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception as e:
                    continue

                modified = False

                # Apply SEARCH/REPLACE
                for idx, (search_text, replace_text) in enumerate(matches, 1):
                    search_text = search_text.strip()
                    replace_text = replace_text.strip()

                    if search_text in file_content:
                        file_content = file_content.replace(search_text, replace_text, 1)
                        modified = True

                # If modified, create backup and save
                if modified:
                    backup_path = filepath.with_suffix(f".{save_num:03d}.bak")
                    try:
                        # Create backup
                        if not backup_path.exists():
                            import shutil
                            shutil.copy(filepath, backup_path)

                        # Write fixed file
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(file_content)

                        modified_files.append(filename)
                    except Exception as e:
                        pass

        return modified_files

    def _start_server(self, project_dir: Path, language: str) -> subprocess.Popen:
        """Start the server process. Uses PORT=0 to avoid port conflicts during validation."""
        env = os.environ.copy()
        # Use a high ephemeral port to avoid conflicts with the actual server
        env["PORT"] = "0"

        if language == "python":
            return subprocess.Popen(
                [sys.executable, "app.py"],
                cwd=project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            return subprocess.Popen(
                ["node", "app.js"],
                cwd=project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )


# Legacy compatibility: keep the old prompt for fallback
ERROR_FIX_PROMPT = """You are an expert debugger. A generated Python/Node.js project failed to run.

ORIGINAL ERROR:
```
{error_output}
```

FILE THAT LIKELY CAUSED THE ERROR:
```{language}
{file_content}
```

FILE PATH: {file_path}

TASK: Fix the code to resolve this error.

RULES:
1. Return ONLY the complete fixed file content
2. Do NOT include markdown code fences
3. Do NOT explain - just return the fixed code
4. Make minimal changes to fix the specific error
5. If it's an import error, check if the import path is correct
6. If it's a syntax error, fix the exact line mentioned
7. If it's a runtime error, add proper error handling

Return the complete fixed file:"""
