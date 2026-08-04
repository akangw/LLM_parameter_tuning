"""Pydantic models for tag validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


# Allowed values per dimension (module-level to avoid Pydantic introspection)
_ALLOWED: dict[str, set[str]] = {
    "model": {"dense", "moe", "mla", "vlm", "quantized"},
    "optimize_target": {"ttft", "tpot", "throughput", "memory"},
    "deploy_topology": {"single_node", "multi_node"},
    "hardware": {"a2", "a3"},
    "deploy_scenario": {"long_input", "long_output", "high_concurrency"},
}


class Tags(BaseModel):
    """Validated tags for a single parameter.

    Each field corresponds to a tag dimension. Multi-select dimensions use
    lists; an empty list means the dimension does not apply.

    The Python field for "model" is named "_model" to avoid Pydantic's
    reserved namespace, but it serializes as "model" via alias.
    """

    model_config = {"populate_by_name": True}

    model: list[str] = Field(default_factory=list, validation_alias="model", serialization_alias="model")
    optimize_target: list[str] = Field(default_factory=list)
    deploy_topology: list[str] = Field(default_factory=list)
    hardware: list[str] = Field(default_factory=list)
    deploy_scenario: list[str] = Field(default_factory=list)

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: list[str]) -> list[str]:
        _check_values(v, _ALLOWED["model"], "model")
        return v

    @field_validator("optimize_target")
    @classmethod
    def validate_optimize_target(cls, v: list[str]) -> list[str]:
        _check_values(v, _ALLOWED["optimize_target"], "optimize_target")
        return v

    @field_validator("deploy_topology")
    @classmethod
    def validate_deploy_topology(cls, v: list[str]) -> list[str]:
        _check_values(v, _ALLOWED["deploy_topology"], "deploy_topology")
        return v

    @field_validator("hardware")
    @classmethod
    def validate_hardware(cls, v: list[str]) -> list[str]:
        _check_values(v, _ALLOWED["hardware"], "hardware")
        return v

    @field_validator("deploy_scenario")
    @classmethod
    def validate_deploy_scenario(cls, v: list[str]) -> list[str]:
        _check_values(v, _ALLOWED["deploy_scenario"], "deploy_scenario")
        return v


def _check_values(values: list[str], allowed: set[str], dimension: str) -> None:
    """Validate that all values in the list are allowed for the dimension."""
    if not isinstance(values, list):
        raise ValueError(f"{dimension} must be a list, got {type(values)}")
    if len(values) != len(set(values)):
        raise ValueError(f"{dimension} must not contain duplicate values")
    for v in values:
        if v not in allowed:
            raise ValueError(
                f"Invalid value '{v}' for dimension '{dimension}'. "
                f"Allowed: {sorted(allowed)}"
            )


def generate_tags_schema_text(tags_yaml_path: Path) -> str:
    """Generate a human-readable description of the tag schema for LLM prompts.

    Reads resources/tags.yaml and formats it as text describing each dimension
    and its allowed values with descriptions.
    """
    with open(tags_yaml_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    lines = []
    for dim_name, dim_def in schema.get("dimensions", {}).items():
        lines.append(f"## {dim_name}")
        lines.append(f"  Description: {dim_def.get('description', '')}")
        lines.append(f"  Type: {dim_def.get('type', 'multi_select')}")
        lines.append(f"  Allowed values:")
        for val_name, val_def in dim_def.get("values", {}).items():
            lines.append(f"    - {val_name}: {val_def.get('description', '')}")
            signals = val_def.get("signals", "")
            if signals:
                lines.append(f"      Signals: {signals}")
        lines.append("")

    return "\n".join(lines)
