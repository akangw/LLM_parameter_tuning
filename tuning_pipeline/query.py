#!/usr/bin/env python3
"""Query tool for the vLLM/vllm-ascend parameter performance knowledge base.

Supports filtering, field projection, full-text search, tag-based search,
sorting, and multiple output formats. Works on tagged parameter YAML files
produced by tag_params.

Usage examples:

    # Show all high-impact parameters
    python query.py --where performance_impact=high

    # Show Ascend-specific compilation params, with selected fields
    python query.py --where scope=vllm-ascend --where category=compilation --show name,default,tuning_advice.quick_guide

    # Full-text search
    python query.py --search "OOM"

    # Combined: search in a subset
    python query.py --where category=memory --search "KV cache"

    # Just count
    python query.py --where performance_impact=high --count

    # Tag-based search
    python query.py --tag model=moe
    python query.py --tag model=moe --tag deploy_scenario=long_input
    python query.py --tag hardware=a3 --where performance_impact=high
    python query.py --tag optimize_target=ttft,tpot

    # List available tag dimensions and values
    python query.py --list-tags

    # Full YAML output for piping
    python query.py --where type=cli --show name,default,cli_example --format yaml

    # List available filterable fields
    python query.py --list-fields
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMS_DIR = SCRIPT_DIR / "tag_params" / "output" / "params"

# Fields that are always present in full-analysis YAMLs (not in "none" params)
COMMON_FIELDS = {
    "name": ("str", "Parameter name (CLI flag, env var, or dotted nested path)"),
    "type": ("enum[cli,env,nested]", "Parameter type"),
    "category": ("enum", "Functional category (parallelism/memory/compilation/...)"),
    "scope": ("enum[vllm,vllm-ascend]", "Source repository"),
    "source_file": ("list[str]", "Definition file paths"),
    "value_type": ("enum", "Python value type"),
    "default": ("any", "Default value"),
    "valid_choices": ("any", "Valid choices or range description"),
    "cli_example": ("str", "Example CLI usage"),
    "deprecated": ("bool", "Whether deprecated"),
    "performance_impact": ("enum[high,medium,low,none]", "Performance impact level"),
    "performance_scope": ("list[enum[latency,throughput,memory]]", "Affected performance dimensions"),
    "impact_detail": ("str", "Detailed performance impact mechanism"),
    "usage_locations": ("list[dict]", "Source code usage sites with version tags"),
    "related_parameters": ("list[dict]", "Semantically related parameters"),
    "constraints": ("list[str]", "Hard constraints (violation causes errors)"),
    "tuning_advice.summary": ("str", "One-line tuning summary"),
    "tuning_advice.suggested_values": ("list[dict]", "Scenario-based recommended values"),
    "tuning_advice.caveats": ("list[str]", "Known pitfalls and warnings"),
    "tuning_advice.quick_guide": ("str", "One-line quick guidance"),
    "analysis_date": ("str", "Analysis date (YYYY-MM-DD)"),
    "skip_reason": ("str", "Reason for skipping (none-impact params only)"),
    "tags.model": ("list[str]", "Tag: model architecture (dense, moe, mla, vlm, quantized)"),
    "tags.optimize_target": ("list[str]", "Tag: optimization target (ttft, tpot, throughput, memory)"),
    "tags.deploy_topology": ("list[str]", "Tag: deployment topology (single_node, multi_node)"),
    "tags.hardware": ("list[str]", "Tag: hardware (a2, a3)"),
    "tags.deploy_scenario": ("list[str]", "Tag: deployment scenario (long_input, long_output, high_concurrency)"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

try:
    from yaml import CSafeLoader as _SafeLoader
except ImportError:
    from yaml import SafeLoader as _SafeLoader


def load_params(params_dir: Path) -> list[dict]:
    """Load all YAML parameter files into memory."""
    allowed_names: set[str] | None = None
    progress_path = params_dir.parent / "progress.json"
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            tagged = progress.get("tagged_params")
            if isinstance(tagged, list):
                allowed_names = {str(name) for name in tagged}
        except (OSError, ValueError):
            allowed_names = None

    # Key by logical parameter name so legacy and canonical filenames do not
    # produce duplicate query results during filename migrations. Sorted
    # iteration makes canonical sanitized filenames win over old "--foo"
    # filenames.
    params_by_name: dict[str, dict] = {}
    for fname in sorted(os.listdir(params_dir)):
        if fname.endswith(".yaml"):
            try:
                with open(params_dir / fname, encoding="utf-8") as f:
                    data = yaml.load(f, Loader=_SafeLoader)
                if isinstance(data, dict):
                    key = str(data.get("name") or fname)
                    if allowed_names is not None and key not in allowed_names:
                        continue
                    params_by_name[key] = data
            except yaml.YAMLError:
                pass  # skip corrupted files
    return list(params_by_name.values())


# ---------------------------------------------------------------------------
# Nested field access
# ---------------------------------------------------------------------------

def get_nested(d: dict, path: str):
    """Return the value at a dot-separated path within a nested dict.

    Examples:
        get_nested(d, "name")                  -> d["name"]
        get_nested(d, "tuning_advice.summary") -> d["tuning_advice"]["summary"]
    """
    parts = path.split(".")
    for part in parts:
        if isinstance(d, dict) and part in d:
            d = d[part]
        else:
            return None
    return d


def _collect_all_field_paths(params: list[dict]) -> list[str]:
    """Walk all params and collect every unique dot-notation field path."""
    seen = set()

    def _walk(obj, prefix=""):
        if isinstance(obj, dict):
            for key, val in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(val, dict):
                    _walk(val, path)  # recurse into nested dict, don't add dict itself
                elif isinstance(val, list) and val and isinstance(val[0], dict):
                    seen.add(path)  # add list-of-dicts as a whole
                    _walk(val[0], path)  # also walk first item for sub-fields
                else:
                    seen.add(path)  # leaf value (scalar, list of scalars, etc.)
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            _walk(obj[0], prefix)

    for p in params:
        _walk(p)

    # Sort: top-level fields first, then nested; within each group, alphabetical
    top = sorted(k for k in seen if "." not in k)
    nested = sorted(k for k in seen if "." in k)
    return top + nested


def _print_tag_reference(params: list[dict]) -> None:
    """Print available tag dimensions with values and their counts."""
    tag_dims: dict[str, dict[str, int]] = {}
    for p in params:
        tags = p.get("tags")
        if not isinstance(tags, dict):
            continue
        for dim, vals in tags.items():
            if dim not in tag_dims:
                tag_dims[dim] = {}
            for v in (vals or []):
                tag_dims[dim][v] = tag_dims[dim].get(v, 0) + 1

    print("Available tag dimensions and values:\n")
    for dim in sorted(tag_dims):
        print(f"  {dim}")
        for val, cnt in sorted(tag_dims[dim].items()):
            print(f"    - {val}: {cnt} params")
        print()


def _format_value(val) -> str:
    """Format a value for table display: flatten lists and dicts, no truncation."""
    if val is None:
        return "-"
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, list):
        if not val:
            return "[]"
        if all(isinstance(x, str) for x in val):
            return ", ".join(val)
        if all(isinstance(x, dict) for x in val):
            items = []
            for x in val:
                if not x:
                    items.append("?")
                elif "file" in x and "context" in x:
                    # usage_location: compact as "file (context...)"
                    ctx = x["context"]
                    items.append(f"{x['file']} ({ctx})")
                elif "name" in x and "relation" in x:
                    # related_parameter: compact as "name: relation"
                    items.append(f"{x['name']}: {x['relation']}")
                elif "scenario" in x:
                    # suggested_value: compact as "scenario → value"
                    items.append(f"{x['scenario']} → {x.get('value', '?')}")
                else:
                    items.append(str(x))
            return "; ".join(items)
        return ", ".join(str(x) for x in val)
    if isinstance(val, dict):
        return "{" + ", ".join(f"{k}: {v}" for k, v in val.items()) + "}"
    return str(val)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def parse_filter(filter_str: str) -> tuple[str, list[str]]:
    """Parse a 'key=value1,value2' filter string.

    Returns (key, [values]).
    """
    if "=" not in filter_str:
        raise ValueError(f"Filter must be 'key=value', got: {filter_str}")
    key, _, values = filter_str.partition("=")
    return key.strip(), [v.strip() for v in values.split(",") if v.strip()]


def matches_filter(param: dict, key: str, target_values: list[str]) -> bool:
    """Check if param's nested field value matches any target value."""
    actual = get_nested(param, key)
    if actual is None:
        return False
    actual_str = str(actual).lower()
    return any(tv.lower() in actual_str for tv in target_values)


def matches_tag(param: dict, dimension: str, target_values: list[str]) -> bool:
    """Check if param's tags[dimension] contains any target value.

    params without a 'tags' field never match.
    """
    tags = param.get("tags")
    if not isinstance(tags, dict):
        return False
    dim_values = tags.get(dimension)
    if not isinstance(dim_values, list):
        return False
    return any(tv in dim_values for tv in target_values)


def matches_search(param: dict, query: str) -> bool:
    """Full-text search across all string-typed leaf fields."""
    query_lower = query.lower()

    def _search_in(val) -> bool:
        if isinstance(val, str):
            return query_lower in val.lower()
        if isinstance(val, dict):
            return any(_search_in(v) for v in val.values())
        if isinstance(val, list):
            return any(_search_in(v) for v in val)
        return False

    return _search_in(param)


def apply_filters(params: list[dict],
                  filters: list[tuple[str, list[str]]],
                  tag_filters: list[tuple[str, list[str]]],
                  search: str | None) -> list[dict]:
    """Apply --where, --tag, and --search filters (AND logic across all).

    Multiple --where / --tag flags: all must match (AND logic).
    Within one filter with comma-separated values: any matches (OR logic).
    """
    result = params
    for key, values in filters:
        result = [p for p in result if matches_filter(p, key, values)]
    for dimension, values in tag_filters:
        result = [p for p in result if matches_tag(p, dimension, values)]
    if search:
        result = [p for p in result if matches_search(p, search)]
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _column_widths(rows: list[list[str]], headers: list[str]) -> list[int]:
    """Compute optimal column widths — no upper cap (agent-facing tool)."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    return widths


def _pad(text: str, width: int) -> str:
    """Left-justify text to width without truncating (agent-facing tool)."""
    return text.ljust(width)


def format_table(params: list[dict], fields: list[str]) -> str:
    """Format results as an aligned plain-text table."""
    if not params:
        return "(no results)"

    headers = fields
    rows = []
    for p in params:
        row = []
        for f in fields:
            val = get_nested(p, f)
            row.append(_format_value(val))
        rows.append(row)

    widths = _column_widths(rows, headers)
    total_width = sum(widths) + 3 * (len(headers) - 1) + 2
    sep = "-" * total_width

    lines = [sep]
    header_line = " | ".join(_pad(h, w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append(sep.replace("-", "="))
    for row in rows:
        row_line = " | ".join(_pad(c, w) for c, w in zip(row, widths))
        lines.append(row_line)
    lines.append(sep)
    lines.append(f"({len(params)} result{'s' if len(params) != 1 else ''})")
    return "\n".join(lines)


def format_yaml(params: list[dict], fields: list[str]) -> str:
    """Format results as YAML documents (compact, selected fields only)."""
    docs = []
    for p in params:
        subset = {}
        for f in fields:
            val = get_nested(p, f)
            if val is not None:
                subset[f] = val
        docs.append(yaml.dump(subset, allow_unicode=True, sort_keys=False,
                              default_flow_style=False, width=120))
    return "---\n".join(docs) if docs else "# (no results)"


def format_summary(params: list[dict], fields: list[str]) -> str:
    """Format results as compact summaries: one header block per parameter."""
    if not params:
        return "(no results)"

    lines = []
    for i, p in enumerate(params):
        if i > 0:
            lines.append("-" * 60)
        name = get_nested(p, "name") or "?"
        impact = get_nested(p, "performance_impact") or "?"
        lines.append(f"[{impact.upper()}] {name}")
        for f in fields:
            if f in ("name", "performance_impact"):
                continue
            val = get_nested(p, f)
            if val is None:
                continue
            label = f.replace("_", " ").replace(".", " > ")
            formatted = _format_value(val)
            lines.append(f"  {label:16s}: {formatted}")
    lines.append("")
    lines.append(f"({len(params)} result{'s' if len(params) != 1 else ''})")
    return "\n".join(lines)


def format_list(params: list[dict]) -> str:
    """Format results as one name per line."""
    if not params:
        return "(no results)"
    names = [get_nested(p, "name") or "?" for p in params]
    return "\n".join(names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Query the vLLM/vllm-ascend parameter knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python query.py --where performance_impact=high
          python query.py --where category=memory,compilation --show name,default,tuning_advice.quick_guide
          python query.py --where scope=vllm-ascend --where type=env --sort-by name
          python query.py --search "HCCL" --format summary
          python query.py --where performance_impact=high --count
          python query.py --tag model=moe --tag deploy_scenario=long_input
          python query.py --list-tags
          python query.py --list-fields
        """),
    )
    ap.add_argument(
        "--params-dir", "-d",
        type=Path,
        default=DEFAULT_PARAMS_DIR,
        help="Path to output/params/ directory (default: tag_params/output/params/)",
    )
    ap.add_argument(
        "--where", "-w",
        action="append",
        default=[],
        metavar="KEY=VALUE1[,VALUE2...]",
        help="Filter: key=value. Repeat for AND; comma-separate values for OR. "
             "Supports dot notation (e.g., tuning_advice.summary).",
    )
    ap.add_argument(
        "--tag", "-t",
        action="append",
        default=[],
        metavar="DIMENSION=VALUE1[,VALUE2...]",
        help="Tag filter: dimension=value. Repeat for AND; comma-separate values "
             "for OR. Example: --tag model=moe --tag deploy_scenario=long_input",
    )
    ap.add_argument(
        "--search", "-s",
        type=str,
        default=None,
        metavar="TEXT",
        help="Full-text search across all fields (case-insensitive).",
    )
    ap.add_argument(
        "--show",
        type=str,
        default="name,type,category,scope,performance_impact,tuning_advice.summary",
        metavar="FIELD1[,FIELD2...]",
        help="Fields to display (dot notation for nested). "
             "Default: name,type,category,scope,performance_impact,tuning_advice.summary",
    )
    ap.add_argument(
        "--show-all", "-a",
        action="store_true",
        help="Show all fields in every result (overrides --show).",
    )
    ap.add_argument(
        "--format", "-f",
        choices=["table", "yaml", "summary", "list"],
        default="table",
        help="Output format. 'table' = aligned columns (default), "
             "'yaml' = YAML documents, 'summary' = detailed blocks, "
             "'list' = names only.",
    )
    ap.add_argument(
        "--sort-by",
        type=str,
        default=None,
        metavar="FIELD",
        help="Sort results by this field before display.",
    )
    ap.add_argument(
        "--sort-reverse", "-r",
        action="store_true",
        help="Reverse sort order.",
    )
    ap.add_argument(
        "--count", "-c",
        action="store_true",
        help="Only print the count of matching parameters.",
    )
    ap.add_argument(
        "--list-fields",
        action="store_true",
        help="List all available fields and their types, then exit.",
    )
    ap.add_argument(
        "--list-tags",
        action="store_true",
        help="List available tag dimensions and values with counts, then exit.",
    )
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --list-fields: print field reference and exit
    if args.list_fields:
        print("Available fields (dot notation for nested):\n")
        max_key_len = max(len(k) for k in COMMON_FIELDS)
        for key, (ftype, desc) in COMMON_FIELDS.items():
            print(f"  {key:<{max_key_len}s}  {ftype:<45s}  {desc}")
        return

    # Validate params dir
    if not args.params_dir.is_dir():
        print(f"Error: params directory not found: {args.params_dir}", file=sys.stderr)
        print("Use --params-dir to specify the correct path.", file=sys.stderr)
        sys.exit(1)

    # Load all params
    params = load_params(args.params_dir)
    if not params:
        print(f"Error: no YAML files found in {args.params_dir}", file=sys.stderr)
        sys.exit(1)

    # --list-tags: print tag dimensions and value distribution, then exit
    if args.list_tags:
        _print_tag_reference(params)
        return

    # Parse where filters
    filters = []
    for fstr in args.where:
        try:
            filters.append(parse_filter(fstr))
        except ValueError as e:
            print(f"Error: invalid filter '{fstr}': {e}", file=sys.stderr)
            sys.exit(1)

    # Parse tag filters
    tag_filters = []
    for tstr in args.tag:
        try:
            tag_filters.append(parse_filter(tstr))
        except ValueError as e:
            print(f"Error: invalid tag filter '{tstr}': {e}", file=sys.stderr)
            sys.exit(1)

    # Parse show fields
    if args.show_all:
        show_fields = None  # signal: use all fields (computed after filtering)
    else:
        show_fields = [f.strip() for f in args.show.split(",") if f.strip()]

    # Apply filters
    results = apply_filters(params, filters, tag_filters, args.search)

    # Resolve show_fields when --show-all (must happen after filtering)
    if show_fields is None:
        show_fields = _collect_all_field_paths(results)

    # Sort
    if args.sort_by:
        def sort_key(p):
            val = get_nested(p, args.sort_by)
            if val is None:
                return ""
            if isinstance(val, list):
                return str(val)
            return val
        try:
            results.sort(key=sort_key, reverse=args.sort_reverse)
        except TypeError:
            # Mixed types — fall back to string sort
            results.sort(key=lambda p: str(sort_key(p)),
                         reverse=args.sort_reverse)

    # Output
    output = None
    if args.count:
        output = str(len(results))
    elif args.format == "table":
        output = format_table(results, show_fields)
    elif args.format == "yaml":
        output = format_yaml(results, show_fields)
    elif args.format == "summary":
        output = format_summary(results, show_fields)
    elif args.format == "list":
        output = format_list(results)

    if output:
        print(output)


if __name__ == "__main__":
    main()
