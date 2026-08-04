"""Stage 1: Coarse filtering of parameters using local rules (no LLM calls)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml


def load_rules(rules_path: Path) -> dict:
    """Load Stage 1 filtering rules from a YAML file."""
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    """Check if name matches any regex pattern in the list."""
    for pat in patterns:
        if re.search(pat, name):
            return True
    return False


def _get_keep_keywords(rules: dict, category: str) -> list[str]:
    """Get keep-keywords for a category from the rules."""
    for rule_group in ["conditional_categories", "low_priority_categories"]:
        cat_rules = rules.get(rule_group, {}).get(category, {})
        if cat_rules:
            return cat_rules.get("keep_keywords", [])
    return []


def _has_any_keyword(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run_stage1(
    params: list[dict],
    rules_path: Path,
    logger: logging.Logger,
) -> tuple[list[dict], list[dict]]:
    """Run Stage 1 coarse filtering.

    Args:
        params: List of parameter dicts from parameters.json.
        rules_path: Path to stage1_rules.yaml.
        logger: Logger instance.

    Returns:
        Tuple of (passed_params, skipped_params).
        passed_params go to Stage 2; skipped_params are recorded but not analyzed.
    """
    rules = load_rules(rules_path)
    always_keep = set(rules.get("always_keep_categories", []))
    skip_name_pats = rules.get("skip_name_patterns", [])
    skip_desc_pats = rules.get("skip_description_patterns", [])
    force_keep_pats = rules.get("force_keep_name_patterns", [])

    passed = []
    skipped = []

    for param in params:
        name = param.get("name", "")
        category = param.get("category", "other")
        description = param.get("description") or ""
        keep_keywords = _get_keep_keywords(rules, category)

        # Rule: force-keep name patterns override everything
        if _matches_any_pattern(name, force_keep_pats):
            passed.append(param)
            continue

        # Rule: always-keep categories
        if category in always_keep:
            passed.append(param)
            continue

        # Rule: skip by name pattern
        if _matches_any_pattern(name, skip_name_pats):
            skipped.append({**param, "skip_reason": "matched skip_name_pattern"})
            continue

        # Rule: skip by description pattern
        if description and _matches_any_pattern(description, skip_desc_pats):
            skipped.append({**param, "skip_reason": "matched skip_description_pattern"})
            continue

        # Rule: low-priority categories must match a keep keyword
        if category in rules.get("low_priority_categories", {}):
            # Check name + description against keep keywords
            search_text = f"{name} {description}"
            if not _has_any_keyword(search_text, keep_keywords):
                skipped.append({
                    **param,
                    "skip_reason": f"category '{category}' with no keep-keyword match",
                })
                continue

        # Rule: conditional categories also check keep keywords (less strict)
        if category in rules.get("conditional_categories", {}):
            # For conditional, keywords are a bonus signal but not required
            # Still pass if no keywords match - will be analyzed by LLM
            pass

        # Default: pass to Stage 2
        passed.append(param)

    logger.info(
        "Stage 1 complete: %d passed, %d skipped (from %d total)",
        len(passed), len(skipped), len(params),
    )
    return passed, skipped
