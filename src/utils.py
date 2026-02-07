"""
Utilities module - Adapted for PaperForge AI MVP generation.
Provides JSON parsing, code extraction, and file utilities.
"""
import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


def content_to_json(data: str) -> dict:
    """
    Parse LLM JSON responses, handling common formatting issues.
    Multiple fallback strategies for messy outputs.
    """
    # First pass: remove [CONTENT] tags and clean up
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()

    # Remove comments after JSON values
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)

    # Fix trailing commas before ]
    clean_data = re.sub(r',\s*\]', ']', clean_data)

    # Remove newlines in JSON structure
    clean_data = re.sub(r'\n\s*', '', clean_data)

    try:
        return json.loads(clean_data)
    except json.JSONDecodeError:
        return _content_to_json_fallback1(data)


def _content_to_json_fallback1(data: str) -> dict:
    """Second fallback with more aggressive cleaning."""
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)
    clean_data = re.sub(r',\s*\]', ']', clean_data)
    clean_data = re.sub(r'\n\s*', '', clean_data)

    try:
        return json.loads(clean_data)
    except json.JSONDecodeError:
        return _content_to_json_fallback2(data)


def _content_to_json_fallback2(data: str) -> dict:
    """Third fallback handling triple quotes and escapes."""
    clean_data = re.sub(r'\[CONTENT\]|\[/CONTENT\]', '', data).strip()
    clean_data = re.sub(r'(".*?"),\s*#.*', r'\1,', clean_data)
    clean_data = re.sub(r'(".*?")\s*#.*', r'\1', clean_data)
    clean_data = re.sub(r',\s*\]', ']', clean_data)
    clean_data = re.sub(r'\n\s*', '', clean_data)
    clean_data = re.sub(r'"""', '"', clean_data)
    clean_data = re.sub(r"'''", "'", clean_data)
    clean_data = re.sub(r"\\\\", "'", clean_data)

    try:
        return json.loads(clean_data)
    except json.JSONDecodeError:
        # Final fallback: try to extract specific known patterns
        return _extract_known_patterns(data)


def _extract_known_patterns(data: str) -> dict:
    """Extract known JSON patterns from messy data."""
    result = {}

    # Try to extract Task list and Logic Analysis
    pattern = r'"Task list":\s*(\[[\s\S]*?\])'
    match = re.search(pattern, data)
    if match:
        try:
            result["Task list"] = json.loads(match.group(1))
        except:
            pass

    pattern = r'"Logic Analysis":\s*(\[[\s\S]*?\])'
    match = re.search(pattern, data)
    if match:
        try:
            result["Logic Analysis"] = json.loads(match.group(1))
        except:
            pass

    pattern = r'"File list":\s*(\[[\s\S]*?\])'
    match = re.search(pattern, data)
    if match:
        try:
            result["File list"] = json.loads(match.group(1))
        except:
            pass

    return result


def extract_code_from_content(content: str) -> str:
    """
    Extract code blocks from LLM response.
    Returns the first code block found.
    """
    # Pattern for markdown code blocks
    pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
    code = re.findall(pattern, content, re.DOTALL | re.MULTILINE)

    if code:
        return code[0].strip()

    # Fallback: try python-specific pattern
    pattern = r'```python\s*(.*?)```'
    result = re.search(pattern, content, re.DOTALL)

    if result:
        return result.group(1).strip()

    return ""


def extract_all_code_blocks(content: str) -> list[tuple[str, str]]:
    """
    Extract all code blocks from LLM response.
    Returns list of (language, code) tuples.
    """
    pattern = r'```(\w+)?\s*\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)

    return [(lang or "text", code.strip()) for lang, code in matches]


def extract_json_from_string(text: str) -> str:
    """Extract JSON content from markdown code block."""
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)

    if match:
        return match.group(1)
    return ""


# =============================================================================
# File Utilities
# =============================================================================

def read_all_files(
    directory: str | Path,
    allowed_extensions: list[str] = None,
    skip_hidden: bool = True,
    max_file_size: int = 204800  # 200KB
) -> dict[str, str]:
    """
    Recursively read all files in a directory.

    Args:
        directory: Directory to scan
        allowed_extensions: List of extensions to include (e.g., ['.py', '.js'])
        skip_hidden: Skip hidden files/directories
        max_file_size: Skip files larger than this (bytes)

    Returns:
        Dict mapping relative paths to file contents
    """
    directory = Path(directory)
    if allowed_extensions is None:
        allowed_extensions = ['.py', '.js', '.ts', '.json', '.yaml', '.yml', '.md', '.txt']

    files_content = {}

    for root, dirs, files in os.walk(directory):
        # Filter hidden directories
        if skip_hidden:
            dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            filepath = Path(root) / filename

            # Skip hidden files
            if skip_hidden and filename.startswith('.'):
                continue

            # Check extension
            ext = filepath.suffix.lower()
            if ext not in allowed_extensions and filename.lower() != 'readme':
                continue

            # Check file size
            try:
                if filepath.stat().st_size > max_file_size:
                    continue
            except:
                continue

            # Read file
            try:
                relative_path = filepath.relative_to(directory)
                with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
                    files_content[str(relative_path)] = f.read()
            except Exception as e:
                pass

    return files_content


def read_python_files(directory: str | Path) -> dict[str, str]:
    """Read all Python files in directory."""
    return read_all_files(directory, allowed_extensions=['.py'])


def format_files_for_prompt(files_dict: dict[str, str], max_total_chars: int = 50000) -> str:
    """
    Format files dict into a prompt-friendly string.
    Truncates if necessary.
    """
    parts = []
    total_chars = 0

    for filepath, content in files_dict.items():
        file_block = f"## File: {filepath}\n```\n{content}\n```\n"

        if total_chars + len(file_block) > max_total_chars:
            remaining = max_total_chars - total_chars - 100
            if remaining > 500:
                truncated = content[:remaining] + "\n... [truncated]"
                file_block = f"## File: {filepath}\n```\n{truncated}\n```\n"
                parts.append(file_block)
            break

        parts.append(file_block)
        total_chars += len(file_block)

    return "\n".join(parts)


# =============================================================================
# Timestamp Utilities
# =============================================================================

def get_timestamp_string() -> str:
    """Get formatted timestamp string for file naming."""
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def format_json_data(data: dict) -> str:
    """Format JSON data for readable display."""
    formatted = ""
    for key, value in data.items():
        formatted += "-" * 40 + "\n"
        formatted += f"[{key}]\n"
        if isinstance(value, list):
            for item in value:
                formatted += f"- {item}\n"
        else:
            formatted += str(value) + "\n"
        formatted += "\n"
    return formatted


# =============================================================================
# Shared Utilities (Consolidated from multiple files)
# =============================================================================

def sanitize_name(name: str, max_length: int = 50) -> str:
    """
    Convert string to valid file/directory/container name.
    Used by: flask_generator, express_generator, deployer, mvp_coding
    """
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[-\s]+", "_", name)
    return name.lower()[:max_length]


def extract_main_class(code: str, default: str = "Algorithm") -> str:
    """
    Extract main class name from code.
    Used by: flask_generator, express_generator
    """
    matches = re.findall(r"class\s+(\w+)", code)
    if matches:
        for match in matches:
            if not match.lower().endswith(("helper", "util", "utils")):
                return match
        return matches[0]
    return default


def write_file(path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Write content to file.
    Used by: flask_generator, express_generator, mvp_coding
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def clean_code_markdown(code: str, language: str = None) -> str:
    """
    Remove markdown code fences from generated code.
    Used by: flask_generator, express_generator, mvp_coding

    Args:
        code: Code string potentially wrapped in markdown
        language: Expected language (python, javascript, etc.)

    Returns:
        Clean code without markdown fences
    """
    if not code:
        return ""

    # Define prefixes based on language
    if language == "python":
        prefixes = ["```python", "```py", "```"]
    elif language in ["javascript", "nodejs", "js"]:
        prefixes = ["```javascript", "```js", "```"]
    else:
        prefixes = ["```python", "```py", "```javascript", "```js", "```"]

    # Remove prefix
    for prefix in prefixes:
        if code.startswith(prefix):
            code = code[len(prefix):]
            break

    # Remove suffix
    if code.endswith("```"):
        code = code[:-3]

    return code.strip()


def ensure_module_exports(code: str, language: str = "javascript") -> str:
    """
    Ensure generated code has proper exports.
    Used by: express_generator, error_fixer

    Args:
        code: Generated code
        language: Programming language

    Returns:
        Code with exports added if missing
    """
    if language not in ["javascript", "nodejs", "js"]:
        return code

    # Check if exports already exist
    if "module.exports" in code or re.search(r"exports\.\w+\s*=", code):
        return code

    # Find the main class name
    class_match = re.search(r"class\s+(\w+)", code)
    if class_match:
        class_name = class_match.group(1)
        code = code.rstrip() + f"\n\nmodule.exports = {{ {class_name} }};\n"

    return code
