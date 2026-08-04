"""Utility functions: YAML I/O, response extraction, tag merging."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import yaml


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the root logger for tag_params."""
    logger = logging.getLogger("tag_params")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


def load_param_yamls(input_dir: Path, logger: logging.Logger) -> list[dict]:
    """Load all parameter YAML files from the given directory.

    Returns a list of (file_path, param_dict) tuples.
    """
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        return []

    yaml_files = sorted(input_dir.glob("*.yaml"))
    params = []
    for yf in yaml_files:
        try:
            with open(yf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "name" in data:
                params.append(data)
            else:
                logger.warning("Skipping %s: not a valid parameter YAML", yf.name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", yf.name, e)

    logger.info("Loaded %d parameter YAMLs from %s", len(params), input_dir)
    return params


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Convert a parameter name into a safe filename.

    Strips leading -- for CLI args, replaces special characters with underscores.
    Consistent with parse_params sanitization so output filenames match.
    """
    safe = name
    if safe.startswith("--"):
        safe = safe[2:]
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
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_")
    return safe or "unnamed_parameter"


def extract_yaml_from_response(text: str) -> str:
    """Extract YAML content from an LLM response, with sanitization.

    Handles ```yaml fences and fixes common LLM YAML formatting mistakes.
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

    # Strip leading colon artefact
    if text.startswith(":\n"):
        text = text[2:]

    return text


def load_tags_schema(yaml_path: Path) -> dict:
    """Load the tag dimension definitions from tags.yaml."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
