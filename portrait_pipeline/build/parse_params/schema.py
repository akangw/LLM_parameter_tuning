"""Pydantic models for output YAML validation, built from resources/schema.yaml."""

from __future__ import annotations

import re
from datetime import date as date_type
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Coercion helpers
# =============================================================================

def _coerce_to_str(v: Any) -> str:
    """Coerce common non-string types that YAML/LLMs produce."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, date_type):
        return v.isoformat()
    return str(v)


# =============================================================================
# Sub-models
# =============================================================================

class UsageLocation(BaseModel):
    file: str
    context: str = ""


class RelatedParameter(BaseModel):
    name: str
    relation: str = ""


class SuggestedValue(BaseModel):
    scenario: str
    value: Any
    reason: str


class TuningAdvice(BaseModel):
    summary: str = ""
    suggested_values: list[SuggestedValue] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    quick_guide: str = ""


# =============================================================================
# Main Parameter YAML model
# =============================================================================

class ParameterYAML(BaseModel):
    """Full schema for a performance-related parameter's output YAML file."""

    # --- 基础信息 ---
    name: str
    type: str
    category: str
    scope: str
    source_file: list[str] = Field(default_factory=list)

    # --- 参数元信息 ---
    value_type: str
    default: Any
    valid_choices: Any
    cli_example: str | None = None
    deprecated: bool = False

    # --- 性能影响分析 ---
    performance_impact: str  # high | medium | low
    performance_scope: list[str] = Field(default_factory=list)
    impact_detail: str

    # --- 使用点分析 ---
    usage_locations: list[UsageLocation] = Field(default_factory=list)

    # --- 参数间关系 ---
    related_parameters: list[RelatedParameter] = Field(default_factory=list)

    # --- 约束 ---
    constraints: list[str] = Field(default_factory=list)

    # --- 调优指导 ---
    tuning_advice: TuningAdvice = Field(default_factory=TuningAdvice)

    # --- 元数据 ---
    analysis_date: str = ""

    # --- Coercion validators ---
    @field_validator("analysis_date", mode="before")
    @classmethod
    def coerce_analysis_date(cls, v: Any) -> str:
        return _coerce_to_str(v)

    # --- Enum validators ---
    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"cli", "env", "nested"}
        if v not in allowed:
            raise ValueError(f"type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        allowed = {
            "scheduling", "memory", "parallelism", "communication",
            "quantization", "compilation", "speculation", "kv_cache",
            "model", "observability", "hardware", "other",
        }
        if v not in allowed:
            raise ValueError(f"category must be one of {allowed}, got '{v}'")
        return v

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, v: str) -> str:
        allowed = {"vllm", "vllm-ascend", "both"}
        if v not in allowed:
            raise ValueError(f"scope must be one of {allowed}, got '{v}'")
        return v

    @field_validator("performance_impact")
    @classmethod
    def validate_performance_impact(cls, v: str) -> str:
        allowed = {"high", "medium", "low", "none"}
        if v not in allowed:
            raise ValueError(f"performance_impact must be one of {allowed}, got '{v}'")
        return v


# =============================================================================
# Minimal model for skipped parameters
# =============================================================================

class SkippedParamYAML(BaseModel):
    """Minimal YAML for parameters with no performance impact."""
    name: str
    type: str
    performance_impact: str = "none"
    skip_reason: str = ""


# =============================================================================
# Schema loading and output-schema text generation
# =============================================================================

def load_schema_dict(schema_path: Path) -> dict:
    """Load the raw schema YAML into a Python dict."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_output_schema_text(schema_path: Path) -> str:
    """Generate a human-readable schema description for inclusion in LLM prompts.

    Reads resources/schema.yaml and formats it as a text description of the
    expected YAML structure, to be embedded in the user prompt template.
    Fields marked 'required' or 'non-null' are annotated to guide LLM output.
    """
    raw = load_schema_dict(schema_path)
    schema = raw.get("schema", raw)

    def _render_field(name: str, defn: dict, indent: int = 2) -> list[str]:
        """Render a single field definition with constraint annotations."""
        out = []
        prefix = " " * indent
        ftype = defn.get("type", "str")
        desc = defn.get("description", "")
        nullable = defn.get("nullable", False)
        required = defn.get("required", False)
        values = defn.get("values", [])

        # Build type string
        type_str = ftype
        if values and ftype.startswith("enum"):
            type_str = " | ".join(repr(v) for v in values)
        elif values and ftype.startswith("list[enum]"):
            type_str = f"list[{', '.join(repr(v) for v in values)}]"
        if nullable:
            type_str += " | null"

        # Build constraint tags
        tags = []
        if required:
            tags.append("REQUIRED")
        if not nullable and ftype not in ("list[str]", "list[dict]", "list[enum]"):
            tags.append("non-null")

        tag_str = f"  # <<< {', '.join(tags)}" if tags else ""
        out.append(f"{prefix}{name}: {type_str}{tag_str}")
        if desc:
            out.append(f"{prefix}  # {desc}")

        # Render nested item_schema
        item_schema = defn.get("item_schema", {})
        if item_schema:
            out.append(f"{prefix}  # Each item must have:")
            for sub_name, sub_def in item_schema.items():
                if isinstance(sub_def, dict):
                    out.extend(_render_field(sub_name, sub_def, indent + 4))
        return out

    lines = []
    for field_name, field_def in schema.items():
        if not isinstance(field_def, dict):
            continue
        lines.extend(_render_field(field_name, field_def))

    return "\n".join(lines)
