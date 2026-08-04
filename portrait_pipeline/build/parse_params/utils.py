"""Pure utility functions: file I/O, string sanitization, logging setup."""

import ast
import logging
import re
import sys
from pathlib import Path

import yaml


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the root logger for parse_params."""
    logger = logging.getLogger("parse_params")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


def sanitize_filename(name: str) -> str:
    """Convert a parameter name into a safe filename.

    Replaces characters unsafe for filenames with underscores.
    """
    # Strip leading -- for CLI args
    safe = name
    if safe.startswith("--"):
        safe = safe[2:]
    # Replace path separators and special chars
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")
    safe = safe.replace("-", "_")
    safe = safe.replace("*", "_star_")
    safe = safe.replace("?", "_")
    safe = safe.replace("<", "_lt_")
    safe = safe.replace(">", "_gt_")
    safe = safe.replace("|", "_")
    safe = safe.replace(":", "_")
    safe = safe.replace('"', "_")
    safe = safe.replace("'", "_")
    # Collapse consecutive underscores
    safe = re.sub(r"_+", "_", safe)
    # Strip leading/trailing underscores
    safe = safe.strip("_")
    return safe or "unnamed_parameter"


def load_params(json_path: Path, logger: logging.Logger) -> list[dict]:
    """Load parameters from a JSON file."""
    import json
    logger.info(f"Loading parameters from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Handle case where JSON is a dict with a "parameters" key
    if isinstance(data, dict) and "parameters" in data:
        return data["parameters"]
    raise ValueError(f"Expected JSON array, got {type(data)}")


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def find_enclosing_scope(file_path: Path, line_number: int) -> tuple[int, int] | None:
    """Use AST to find the function/class that contains the given line number.

    Returns (start_line, end_line) of the enclosing scope, or None on failure.
    Fallback: returns (line-20, line+20) if AST parsing fails.
    """
    if not file_path.exists():
        return None
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (hasattr(node, "end_lineno") and node.end_lineno
                        and node.lineno <= line_number <= node.end_lineno):
                    return (node.lineno, node.end_lineno)
        # Not inside any function/class — return surrounding module-level context
        total = len(source.splitlines())
        return (max(1, line_number - 30), min(total, line_number + 30))
    except SyntaxError:
        return None


def resolve_cli_variable_name(file_path: Path, cli_flag: str) -> str | None:
    """Parse argparse add_argument calls to find the variable name for a CLI flag.

    vLLM uses a pattern like:
        parser.add_argument("--tensor-parallel-size", ..., **parallel_kwargs["tensor_parallel_size"])

    The flag's dest defaults to the flag name with -- stripped and - replaced with _.
    We find the add_argument call containing this flag, then look at the kwargs dict
    key which is the canonical variable name used throughout the codebase.

    Returns the variable name (e.g. "tensor_parallel_size") or None.
    """
    if not file_path.exists():
        return None
    # Default transformation: --tensor-parallel-size → tensor_parallel_size
    default_dest = cli_flag.lstrip("-").replace("-", "_")
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        # Find the add_argument line containing this flag (exact match to avoid
        # --model matching --model-impl)
        flag_pattern = re.compile(rf'"({re.escape(cli_flag)})"')
        for line in source.splitlines():
            if flag_pattern.search(line) and "add_argument" in line:
                # Try to extract kwargs key: **xxx_kwargs["variable_name"]
                m = re.search(r'\*\*\w+\["(\w+)"\]', line)
                if m:
                    return m.group(1)
                # Try dest= parameter
                m = re.search(r'dest\s*=\s*"(\w+)"', line)
                if m:
                    return m.group(1)
        return default_dest
    except Exception:
        return default_dest


def read_source_scope(file_path: Path, line_number: int | None,
                      max_lines: int = 200) -> str:
    """Read the enclosing function/class containing the given line.

    Uses AST to find scope boundaries. If the scope exceeds max_lines,
    it is capped to max_lines centered on the target line.
    Falls back to ±30 lines on AST failure.
    """
    if not file_path.exists():
        return f"# File not found: {file_path.name}\n"
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        total = len(all_lines)

        if line_number is None:
            start, end = 1, min(max_lines, total)
        else:
            scope = find_enclosing_scope(file_path, line_number)
            if scope:
                start, end = scope
                # Cap oversized scopes around the target line
                if end - start + 1 > max_lines:
                    half = max_lines // 2
                    start = max(start, line_number - half)
                    end = min(end, start + max_lines - 1)
            else:
                start = max(1, line_number - 30)
                end = min(total, line_number + 30)

        result = []
        for i in range(start, end + 1):
            marker = ">>>" if line_number and i == line_number else "   "
            result.append(f"{marker} {i:6d}: {all_lines[i - 1]}")
        return "\n".join(result)
    except Exception as e:
        return f"# Error reading {file_path.name}: {e}\n"


def extract_yaml_from_response(text: str) -> str:
    """Extract YAML content from an LLM response, with sanitization.

    Handles ```yaml fences and fixes common LLM YAML formatting mistakes.
    If sanitization breaks valid YAML, falls back to the original text.
    """
    text = text.strip()
    # Try to extract from ```yaml fence
    fence_match = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # Also try ``` without language tag
    fence_match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Strip leading colon artefact that some models emit before valid YAML
    # (e.g. ":\nname: FOO" instead of "name: FOO")
    if text.startswith(":\n"):
        text = text[2:]

    sanitized = _sanitize_llm_yaml(text)

    # If sanitization broke a valid YAML, use the original
    if sanitized != text:
        try:
            yaml.safe_load(sanitized)
        except Exception:
            try:
                yaml.safe_load(text)
                return text  # original was valid, sanitization broke it
            except Exception:
                pass  # both broken, return sanitized as best effort

    return sanitized


def _needs_quoting(value: str) -> bool:
    """Check if a YAML scalar value contains characters that need quoting."""
    if not value:
        return False
    # Block scalars and collections are self-delimiting
    if value.startswith("|") or value.startswith(">") or \
       value.startswith("[") or value.startswith("{"):
        return False
    # Double-quoted string: check for unescaped inner quotes.
    # LLMs often write `"text with "inner" quotes"` without escaping.
    if value.startswith('"') and value.endswith('"'):
        inner = value[1:-1]
        if '"' in inner.replace('\\"', ''):
            return True  # unescaped inner quotes — needs re-quoting
        return False
    # Braces, parens, colons, unquoted double-quotes — break YAML
    if any(c in value for c in ('{', '}', '(', ')', ':', '"')):
        return True
    # Single-quoted but potentially ambiguous if it contains spaces
    safely_single_quoted = (
        value.startswith("'") and value.endswith("'") and value.count("'") == 2)
    if ' ' in value and not safely_single_quoted:
        return True
    return False


def _sanitize_llm_yaml(text: str) -> str:
    """Fix common YAML formatting mistakes made by LLMs.

    Handles: unquoted colons, braces, parens, internal double-quotes in
    both key:value lines and list items.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        if not line.strip():
            result.append(line)
            continue

        # Case 1: key: value  (e.g. "  context: Validation: raises...")
        m = re.match(r'^(\s*)([\w][\w\s_-]*):\s+(.+)$', line)
        if m:
            indent, key, value = m.groups()
            if _needs_quoting(value):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                line = f'{indent}{key}: "{escaped}"'
            result.append(line)
            continue

        # Case 2: list item  (e.g. "  - Must be one of: ...")
        m = re.match(r'^(\s*-\s)(.+)$', line)
        if m:
            prefix, value = m.groups()
            # If the value already looks like a mapping key: value pair
            # (e.g. "scenario: >-"), check whether the inner value needs quoting.
            # LLMs sometimes write "- name: --flag '{\"key\": \"val\"}'"
            # where the outer key:value looks valid but the inner value contains
            # colons/braces that break YAML.
            inner_match = re.match(r'^(\w[\w_-]*):\s+(.+)$', value)
            if inner_match:
                inner_value = inner_match.group(2)
                if _needs_quoting(inner_value):
                    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                    line = f'{prefix}"{escaped}"'
            elif _needs_quoting(value):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                line = f'{prefix}"{escaped}"'
            result.append(line)
            continue

        result.append(line)

    # Post-processing: fix incorrectly quoted key:value pairs.
    # LLM sometimes writes `- "key: value"` or `"key: block-indicator"`,
    # which turns a mapping entry into a scalar string, breaking the YAML structure.
    for i in range(len(result)):
        # Pattern: `  - "key: value"` → `  - key: "value"`
        m = re.match(r'^(\s*)-\s+"(\w[\w\s_-]*):\s+(.+)"$', result[i])
        if m:
            indent, key, value = m.group(1), m.group(2), m.group(3)
            # Only fix if next line is indented beyond the list marker (indicating dict entry)
            min_next_indent = len(indent) + 2  # e.g. "  -" → next at "    "
            if i + 1 < len(result) and re.match(r'^\s{' + str(min_next_indent) + r',}\S', result[i + 1]):
                result[i] = f'{indent}- {key}: "{value}"'
                continue

    return "\n".join(result)
