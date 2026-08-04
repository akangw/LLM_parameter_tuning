"""Stage 2: Deep LLM analysis of performance-relevant parameters.

For each parameter passed from Stage 1:
  a. Read source code context from the vllm/vllm-ascend repos
  b. Build a prompt using resource templates
  c. Call the Anthropic API to judge performance impact + generate YAML
  d. Validate the generated YAML against the schema
  e. Write the YAML to the output directory
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import traceback
from datetime import date
from pathlib import Path

import yaml
from anthropic import Anthropic, AsyncAnthropic, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from . import config
from .schema import ParameterYAML, SkippedParamYAML, generate_output_schema_text
from .utils import (
    extract_yaml_from_response,
    read_source_scope,
    resolve_cli_variable_name,
    sanitize_filename,
)


# =============================================================================
# Prompt building
# =============================================================================

class PromptBuilder:
    """Builds LLM prompts from resource template files."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.system_prompt = config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        self.user_template = config.USER_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        self.output_schema_text = generate_output_schema_text(config.SCHEMA_YAML_PATH)

    def build_user_prompt(
        self,
        param: dict,
        definition_context: str,
        usage_contexts: str,
        doc_contexts: str = "",
    ) -> str:
        """Render the user prompt template for a parameter."""
        return self.user_template.format(
            name=param.get("name", ""),
            type=param.get("type", ""),
            description=param.get("description", "No description available."),
            definition_context=definition_context or "# No definition context available.",
            usage_contexts=usage_contexts or "# No usage contexts found.",
            doc_contexts=doc_contexts or "# No relevant documentation found.",
            output_schema=self.output_schema_text,
        )


# =============================================================================
# Source code context reader
# =============================================================================

class ContextReader:
    """Reads source code context for a parameter from the vllm/vllm-ascend repos.

    Uses grep to locate definition lines, then AST to extract the enclosing
    function/class scope. For CLI params, resolves the argparse variable name
    to accurately find all usage sites across the codebase.
    """

    # Fields commonly passed to dispatch/factory functions — we also search
    # for `def func(..., field, ...)` patterns to catch the dispatcher body.
    _DISPATCH_FIELDS = {"method", "backend", "mode", "policy", "type", "strategy"}

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _grep_file(self, repo_root: Path, rel_path: str,
                   pattern: str) -> list[int]:
        """Find all lines in a file matching a pattern. Returns line numbers.

        Uses word-boundary regex for simple identifiers (avoids substring
        matches like 'mode' matching 'cudagraph_mode').
        """
        file_path = repo_root / rel_path
        if not file_path.exists():
            return []
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            # For simple identifiers (starts with word char), use word boundaries
            # to avoid substring matches like 'mode' matching 'cudagraph_mode'.
            # CLI flags like --param-name start with '-' so use exact match.
            if re.match(r'^\w', pattern):
                regex = re.compile(rf'\b{re.escape(pattern)}\b')
            else:
                regex = re.compile(re.escape(pattern))
            return sorted(set(
                i + 1 for i, line in enumerate(content.splitlines())
                if regex.search(line)
            ))
        except Exception as e:
            self.logger.debug("grep %s: %s", rel_path, e)
            return []

    def _prioritize_field_defs(self, lines: list[int], field: str,
                                file_path: Path) -> list[int]:
        """For nested params, prioritize lines that look like actual field definitions
        (e.g. `field_name: Type = default`) over method references.

        Keeps original order within each priority group.
        """
        if not file_path.exists() or len(lines) <= 1:
            return lines
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            all_lines = content.splitlines()
            field_defs = []
            others = []
            for ln in lines:
                line_text = all_lines[ln - 1].strip()
                # Field definition: `field: Type` or `field: Type = value`
                if re.match(rf'\b{field}\s*:\s*\w+', line_text):
                    field_defs.append(ln)
                else:
                    others.append(ln)
            return field_defs + others
        except Exception:
            return lines

    def _grep_repo(self, repo_root: Path, pattern: str,
                   file_filter: str = "*.py") -> list[dict]:
        """Search a repo with grep for usage sites.

        Results are sorted by relevance: engine/config/worker files first,
        tests/tools/benchmarks/docs last. Caller controls final cap.
        """
        results = []
        try:
            import subprocess
            proc = subprocess.run(
                ["grep", "-rnE", f"--include={file_filter}", pattern, str(repo_root)],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().split("\n"):
                    parts = line.split(":", 2)
                    if len(parts) >= 2:
                        file_path = parts[0]
                        line_num = parts[1]
                        try:
                            file_rel = str(Path(file_path).relative_to(repo_root))
                        except ValueError:
                            file_rel = file_path
                        results.append({
                            "file": file_rel,
                            "line": int(line_num),
                        })
        except Exception as e:
            self.logger.debug("grep %s: %s", repo_root.name, e)

        # Pre-sort safety cap: prevents pathological cases (10k+ hits for very
        # common variable names) while ensuring important but alphabetically-late
        # files (e.g. ascend_config.py at index 211 of 464) are not dropped.
        limit = config.GREP_MAX_RAW_RESULTS or len(results)
        return self._sort_by_relevance(results[:limit])[:50]

    @staticmethod
    def _sort_by_relevance(results: list[dict]) -> list[dict]:
        """Sort grep results so core engine code comes before tests/tools/docs.

        Priority rules are defined in config.FILE_PRIORITY_RULES, evaluated
        top-to-bottom with first-match-wins. Lower number = higher priority.
        """
        def _priority(file_path: str) -> int:
            p = file_path.lower()
            for pri, match_type, pattern in config.FILE_PRIORITY_RULES:
                if match_type == "startswith" and p.startswith(pattern):
                    return pri
                if match_type == "contains" and pattern in p:
                    return pri
            return config.DEFAULT_FILE_PRIORITY

        return sorted(results, key=lambda r: (_priority(r["file"]), r["file"]))

    def _find_definition(self, param: dict) -> list[dict]:
        """Find definition sites across both repos.

        For CLI params: searches known arg_utils files.
        For env vars: searches known envs.py files, falls back to broad grep.
        For nested params: greps for the field name, prioritizing config/ dirs.
        """
        name = param["name"]
        ptype = param["type"]
        results = []

        if ptype == "cli":
            cli_flag = name if name.startswith("--") else f"--{name}"
            for repo_root, repo_name in [(config.VLLM_ASCEND_ROOT, "vllm-ascend"),
                                          (config.VLLM_ROOT, "vllm")]:
                # Search the arg_utils.py file for this repo
                def_file = ("vllm_ascend/engine/arg_utils.py" if repo_name == "vllm-ascend"
                            else "vllm/engine/arg_utils.py")
                lines = self._grep_file(repo_root, def_file, cli_flag)
                for ln in lines[:2]:
                    results.append({"repo": repo_name, "file": def_file, "line": ln})

        elif ptype == "env":
            for repo_root, repo_name in [(config.VLLM_ASCEND_ROOT, "vllm-ascend"),
                                          (config.VLLM_ROOT, "vllm")]:
                def_file = ("vllm_ascend/envs.py" if repo_name == "vllm-ascend"
                            else "vllm/envs.py")
                lines = self._grep_file(repo_root, def_file, f'"{name}"')
                for ln in lines[:2]:
                    results.append({"repo": repo_name, "file": def_file, "line": ln})
            # Fallback: some env vars are only in docs/C++ — grep entire repo
            if not results:
                for repo_root, repo_name in [(config.VLLM_ASCEND_ROOT, "vllm-ascend"),
                                              (config.VLLM_ROOT, "vllm")]:
                    hits = self._grep_repo(repo_root, f'"{name}"')
                    for h in hits[:1]:
                        h["repo"] = repo_name
                        results.append(h)

        else:  # nested — use prefix to locate the exact config file
            parts = name.split(".")
            prefix = parts[0] if parts else name
            field = parts[-1] if parts else name
            file_map = config.NESTED_PREFIX_TO_FILE.get(prefix, {})

            for repo_root, repo_name in [(config.VLLM_ASCEND_ROOT, "vllm-ascend"),
                                          (config.VLLM_ROOT, "vllm")]:
                target_file = file_map.get(repo_name)
                if not target_file:
                    continue  # no known config file for this prefix in this repo
                lines = self._grep_file(repo_root, target_file, field)
                # Prioritize actual field definitions over method references,
                # then take top 2
                lines = self._prioritize_field_defs(
                    lines, field, repo_root / target_file)
                for ln in lines[:2]:
                    results.append(
                        {"repo": repo_name, "file": target_file, "line": ln})

        return results

    def _resolve_search_var(self, param: dict, defs: list[dict]) -> str:
        """Resolve the canonical variable name for searching usage sites.

        For CLI params: parses argparse call to find the variable name.
        For others: uses the parameter name directly.
        """
        name = param["name"]
        ptype = param["type"]

        if ptype == "cli":
            cli_flag = name if name.startswith("--") else f"--{name}"
            # Try vllm-ascend first, then vllm upstream
            for repo_root, def_file in [
                (config.VLLM_ASCEND_ROOT, "vllm_ascend/engine/arg_utils.py"),
                (config.VLLM_ROOT, "vllm/engine/arg_utils.py"),
            ]:
                var = resolve_cli_variable_name(repo_root / def_file, cli_flag)
                if var:
                    return var
            # Fallback: standard argparse transformation
            return cli_flag.lstrip("-").replace("-", "_")

        if ptype == "nested":
            parts = name.split(".")
            field = parts[-1] if parts else name
            if len(parts) >= 2:
                # Narrow search with parent prefix. `.*` covers:
                #   parent.field   parent.get("field")   parent["field"]
                parent = parts[-2]
                return rf'{parent}.*{field}\b'
            return field

        return name

    @staticmethod
    def _version_tag(repo: str, file_path: str) -> str:
        """Annotate file path with v1/v2 version info."""
        p = file_path.lower()
        if repo == "vllm-ascend":
            if "/v2/" in p:
                return "[v2-ascend]"
            if "/v1/" in p or "model_runner_v1" in p or "_v1" in p:
                return "[v1-ascend]"
            if "/spec_decode/" in p:
                return "[v1-ascend]"
            if "/_310p/" in p:
                return "[310p-ascend]"
            return "[ascend]"
        else:  # vllm
            if "/v1/" in p:
                return "[v1-upstream]"
            if "/v2/" in p:
                return "[v2-upstream]"
            return "[upstream]"

    # =========================================================================
    # Doc search
    # =========================================================================

    # Leaf names that are too common to search bare — for these we only use
    # parent-qualified patterns to avoid flooding results with noise.
    _GENERIC_LEAF_NAMES = {
        "method", "type", "model", "mode", "backend", "policy", "strategy",
        "size", "path", "name", "port", "host", "config", "enabled",
        "enable", "disable", "file", "dir", "url", "key", "token",
    }

    def _build_doc_search_patterns(self, param: dict) -> list[str]:
        """Build grep patterns for searching doc files by parameter type."""
        name = param.get("name", "")
        ptype = param.get("type", "")
        if not name or not ptype:
            return []
        patterns = []

        if ptype == "cli":
            cli_flag = name if name.startswith("--") else f"--{name}"
            patterns.append(re.escape(cli_flag))
            var_name = cli_flag.lstrip("-").replace("-", "_")
            if var_name and var_name != cli_flag:
                patterns.append(re.escape(var_name))
        elif ptype == "env":
            patterns.append(re.escape(name))
        else:  # nested
            parts = name.split(".")
            leaf = parts[-1] if parts else name
            parent = parts[-2] if len(parts) >= 2 else ""
            # For generic leaf names, skip the bare leaf search entirely —
            # it produces too much noise (e.g. "method" → 185 hits).
            # Always include parent.field qualified form.
            if parent:
                patterns.append(re.escape(f"{parent}.{leaf}"))
            if leaf and leaf not in self._GENERIC_LEAF_NAMES:
                patterns.append(re.escape(leaf))
            # Also try parent-flag form for nested params whose parent maps
            # to a CLI flag (e.g. speculative_config → --speculative-config).
            # Docs use the CLI form, so "--parent-form.*field" catches those.
            if parent:
                cli_parent = f"--{parent.replace('_', '-')}"
                patterns.append(rf'{re.escape(cli_parent)}.*\b{re.escape(leaf)}\b')

        return patterns

    def _grep_docs(self, pattern: str) -> list[dict]:
        """Search all .md files under configured doc dirs for a pattern."""
        results = []
        try:
            import subprocess
            for doc_dir, label in config.DOC_SEARCH_DIRS:
                if not doc_dir.exists():
                    continue
                proc = subprocess.run(
                    ["grep", "-rnE", "--include=*.md", pattern, str(doc_dir)],
                    capture_output=True, text=True, timeout=15,
                )
                if proc.returncode == 0:
                    for line in proc.stdout.strip().split("\n"):
                        parts = line.split(":", 2)
                        if len(parts) >= 2:
                            file_path = parts[0]
                            line_num = parts[1]
                            try:
                                file_rel = str(Path(file_path).relative_to(doc_dir))
                            except ValueError:
                                file_rel = file_path
                            results.append({
                                "file": file_rel,
                                "line": int(line_num),
                                "doc_label": label,
                                "doc_root": doc_dir,
                            })
        except Exception as e:
            self.logger.debug("grep docs: %s", e)
        return results

    @staticmethod
    def _sort_docs_by_priority(hits: list[dict]) -> list[dict]:
        """Sort doc hits so high-value docs (config/feature guides) come first."""
        def _priority(file_path: str) -> int:
            p = file_path.lower()
            for pri, match_type, pat in config.DOC_PRIORITY_RULES:
                if match_type == "startswith" and p.startswith(pat):
                    return pri
            return config.DEFAULT_DOC_PRIORITY

        return sorted(hits, key=lambda r: (_priority(r["file"]), r["file"]))

    def _read_doc_context(self, hit: dict) -> str | None:
        """Read surrounding context of a doc hit.

        For hits inside fenced code blocks, captures the block plus any
        explanatory text immediately after. For regular text hits, captures
        surrounding paragraphs. Capped to MAX_DOC_SNIPPET_LINES.
        """
        file_path = hit["doc_root"] / hit["file"]
        if not file_path.exists():
            return None
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            all_lines = content.splitlines()
            total = len(all_lines)
            line_num = hit["line"]
            max_lines = config.MAX_DOC_SNIPPET_LINES

            # Detect whether hit is inside a fenced code block.
            # Count ``` fences backwards from the hit: if the count is odd,
            # the nearest fence is an opening fence (hit is inside the block).
            # If even, the hit is between blocks (or outside all blocks).
            fence_range = config.DOC_FENCE_SEARCH_LINES
            in_code_block = False
            block_start = 0
            block_end = total
            back_fences = []
            for i in range(line_num - 1, max(0, line_num - fence_range), -1):
                if all_lines[i].strip().startswith("```"):
                    back_fences.append(i)
            if len(back_fences) % 2 == 1:
                # Odd count: nearest backwards fence is an opening fence
                in_code_block = True
                block_start = back_fences[0]  # closest one
            if in_code_block:
                for i in range(line_num, min(total, line_num + fence_range)):
                    if all_lines[i].strip().startswith("```"):
                        block_end = i
                        break

            if in_code_block:
                start = max(0, block_start)
                end = min(total, block_end + 1)
                # Include explanatory text after the code block.
                # Continue until we hit a new code fence, a heading, or run out
                # of budget — blank lines between paragraphs should not stop us.
                for i in range(block_end + 1, min(total, block_end + 1 + max_lines)):
                    stripped = all_lines[i].strip()
                    if stripped.startswith("```") or stripped.startswith("#"):
                        break
                    end = i + 1
            else:
                # Find paragraph boundaries (blank-line delimited)
                para_start = line_num - 1
                while para_start >= 0 and all_lines[para_start].strip():
                    para_start -= 1
                para_start += 1
                para_end = line_num
                while para_end < total and all_lines[para_end].strip():
                    para_end += 1
                start = max(0, para_start - 3)
                end = min(total, para_end + 3)

            # Cap oversized context around the hit line
            if end - start > max_lines:
                half = max_lines // 2
                start = max(start, line_num - half)
                end = min(end, start + max_lines)

            lines_out = []
            for i in range(start, end):
                marker = ">>>" if i + 1 == line_num else "   "
                lines_out.append(f"{marker} {i + 1:6d}: {all_lines[i]}")
            return "\n".join(lines_out)
        except Exception as e:
            self.logger.debug("read doc context %s: %s", hit["file"], e)
            return None

    @staticmethod
    def _is_doc_noise(context: str) -> bool:
        """Skip snippets that are pure command listings without explanation."""
        if not context:
            return True
        lines_lower = context.lower()
        line_count = context.count("\n") + 1

        # Docker/pip/apt/wget/curl snippets without explanation are noise
        noise_markers = [
            "docker run", "pip install", "apt update", "apt install",
            "wget http", "curl http",
        ]
        for marker in noise_markers:
            if marker in lines_lower and line_count <= 12:
                explanatory = any(
                    kw in lines_lower
                    for kw in ["note", "explain", "recommend", "should", "must",
                               "parameter", "mean", "effect", "impact", "important",
                               "notice", "indicate", "specify", "control"]
                )
                if not explanatory:
                    return True

        # Strip the line-number prefix added by _read_doc_context.
        raw_lines: list[str] = []
        for line in context.split("\n"):
            m = re.match(r"^[ >]{3} \s*\d+:\s?(.*)$", line)
            raw_lines.append(m.group(1) if m else line)

        # Classify each content line as code-like or prose-like.
        # This is used in both fence-aware and content-based checks below.
        def _classify(s: str) -> str:
            """Return 'code', 'prose', or 'skip' for a stripped raw line."""
            if not s or s.startswith("```"):
                return "skip"
            if re.match(
                r'^(export\s|vllm\s|python\d?\s|pip\d?\s|rm\s|mkdir\s|'
                r'source\s|cp\s|mv\s|ln\s|wget\s|curl\s|#!/|'
                r'--\w|\w+\s*=|[\{\}\]])',
                s,
            ) or ('":' in s and not s.startswith("[")) or s.endswith("\\"):
                return "code"
            # Prose: English sentences, markdown headings/list items,
            # table rows, shell comments (# ...)
            if re.match(r'^[#*\-\d|]', s) or (s[0].isupper() if s else False):
                return "prose"
            return "skip"

        # --- Fence-aware check: code blocks delimited by ``` fences. ---
        fence_indices = [i for i, l in enumerate(raw_lines)
                         if l.strip().startswith("```")]
        total = len(raw_lines)
        code_ranges: list[tuple[int, int]] = []

        if len(fence_indices) >= 2:
            for i in range(0, len(fence_indices) - 1, 2):
                code_ranges.append((fence_indices[i] + 1, fence_indices[i + 1]))
        elif len(fence_indices) == 1:
            fi = fence_indices[0]
            if fi < total * 0.5:
                code_ranges.append((fi + 1, total))
            else:
                code_ranges.append((0, fi))

        if code_ranges:
            code_in = 0
            prose_in = 0   # comments inside code blocks count as prose
            prose_out = 0
            for i, l in enumerate(raw_lines):
                if i in fence_indices:
                    continue
                s = l.strip()
                cls = _classify(s)
                if cls == "skip":
                    continue
                in_code = any(start <= i < end for start, end in code_ranges)
                if in_code:
                    if cls == "code":
                        code_in += 1
                    elif cls == "prose":
                        prose_in += 1
                else:
                    if cls == "prose":
                        prose_out += 1
            # Pure code listing: many code lines, negligible prose
            # (both inside and outside the block)
            total_prose = prose_in + prose_out
            if code_in >= 5 and total_prose <= 2:
                return True

        # --- Content-based fallback: fences may be outside the cap window ---
        code_like = 0
        prose_like = 0
        for raw in raw_lines:
            cls = _classify(raw.strip())
            if cls == "code":
                code_like += 1
            elif cls == "prose":
                prose_like += 1

        total_content = code_like + prose_like
        if total_content >= 5 and code_like > total_content * 0.65:
            return True

        return False

    def _search_docs(self, param: dict) -> str:
        """Search vllm-ascend docs for parameter mentions and explanatory context.

        Returns formatted doc context snippets, or empty string.
        """
        patterns = self._build_doc_search_patterns(param)
        all_hits = []
        for pat in patterns:
            hits = self._grep_docs(pat)
            all_hits.extend(hits)

        if not all_hits:
            return ""

        # Deduplicate by (file, line cluster)
        seen: set[tuple[str, int]] = set()
        unique_hits = []
        for h in all_hits:
            cluster = (h["file"], h["line"] // 20)
            if cluster not in seen:
                seen.add(cluster)
                unique_hits.append(h)

        # Sort by doc priority rules
        unique_hits = self._sort_docs_by_priority(unique_hits)

        # Read context, filter noise, cap results
        snippets = []
        total_lines = 0
        remaining_after_cap = 0
        for h in unique_hits:
            if len(snippets) >= config.MAX_DOC_SNIPPETS:
                remaining_after_cap += 1
                continue
            if total_lines >= config.MAX_DOC_TOTAL_LINES:
                remaining_after_cap += 1
                continue
            context = self._read_doc_context(h)
            if context and not self._is_doc_noise(context):
                header = f"### [doc] {h['file']}:{h['line']}"
                snippets.append(f"{header}\n{context}")
                total_lines += context.count("\n") + 2

        if remaining_after_cap > 0:
            self.logger.debug(
                "doc search capped: %d qualified hits dropped (limit %d snippets / %d lines)",
                remaining_after_cap, config.MAX_DOC_SNIPPETS,
                config.MAX_DOC_TOTAL_LINES,
            )

        return "\n".join(snippets) if snippets else ""

    def read_contexts(self, param: dict) -> tuple[str, str, str]:
        """Read definition, usage, and doc contexts for a parameter.

        Definition context: full enclosing function/class via AST.
        Usage context: grep for variable name, read enclosing scope at each hit.
        Doc context: search vllm-ascend docs for parameter explanations.
        Each site is annotated with version tag (v1/v2/upstream) or [doc] tag.

        Returns:
            Tuple of (definition_context_str, usage_contexts_str, doc_contexts_str).
        """
        # Step 1: Find definition sites
        defs = self._find_definition(param)

        # Step 2: Resolve the canonical variable name for usage search
        search_var = self._resolve_search_var(param, defs)

        # Step 3: Read definition context (AST-based scope)
        def_parts = []
        for d in defs:
            repo_root = config.VLLM_ASCEND_ROOT if d["repo"] == "vllm-ascend" else config.VLLM_ROOT
            file_path = repo_root / d["file"]
            scope_text = read_source_scope(file_path, d["line"])
            version = self._version_tag(d["repo"], d["file"])
            def_parts.append(f"### {version} {d['file']}:{d['line']}\n{scope_text}")
        def_context = "\n".join(def_parts) if def_parts else (
            "# No definition found in vllm or vllm-ascend repos.")

        # Step 4: Find and read usage sites, interleaving both repos.
        # Uses cluster-based deduplication: hits in the same file are grouped
        # by their line-number proximity (one cluster per scope-window-sized
        # block). A per-file cluster cap prevents any single large file from
        # consuming the entire quota (e.g. model_runner_v1.py with 5+ clusters
        # would crowd out platform.py).
        usage_parts = []
        seen_clusters: set[tuple[str, int]] = set()
        file_cluster_counts: dict[str, int] = {}
        per_repo = config.MAX_USAGE_LOCATIONS // 2

        for repo_name, repo_root in [("vllm-ascend", config.VLLM_ASCEND_ROOT),
                                      ("vllm", config.VLLM_ROOT)]:
            if len(usage_parts) >= config.MAX_USAGE_LOCATIONS:
                break
            hits = self._grep_repo(repo_root, search_var)
            repo_count = 0
            for h in hits:
                if repo_count >= per_repo or len(usage_parts) >= config.MAX_USAGE_LOCATIONS:
                    break
                if any(h["file"] == d["file"] and h["line"] == d["line"] for d in defs):
                    continue
                # Per-file cluster cap
                if file_cluster_counts.get(h["file"], 0) >= config.MAX_CLUSTERS_PER_FILE:
                    continue
                cluster_key = (h["file"], h["line"] // config.CLUSTER_WINDOW_LINES)
                if cluster_key in seen_clusters:
                    continue
                seen_clusters.add(cluster_key)
                file_cluster_counts[h["file"]] = file_cluster_counts.get(h["file"], 0) + 1
                file_path = repo_root / h["file"]
                scope_text = read_source_scope(file_path, h["line"],
                                               max_lines=config.MAX_USAGE_SCOPE_LINES)
                version = self._version_tag(repo_name, h["file"])
                usage_parts.append(
                    f"### {version} {h['file']}:{h['line']}\n{scope_text}")
                repo_count += 1

        # Step 5: Look for dispatch functions that consume this parameter's value.
        # For fields like "method", "backend", "mode", the actual dispatch logic
        # is often in a function like `get_spec_decode_method(method, ...)` where
        # the field name appears as a local parameter — invisible to our parent.field
        # grep. Only includes core source files (not tests/tools/CI/docs).
        if param["type"] == "nested":
            leaf = param["name"].split(".")[-1]
            if leaf in self._DISPATCH_FIELDS:
                dispatch_pat = rf"def \w+\([^)]*\b{leaf}\b[^)]*\)"
                _NOISE_DIRS = {"test", "tests", "tool", "tools", "benchmark",
                               "benchmarks", "doc", "docs", ".github", "example",
                               "examples", "ci_log"}
                for repo_name, repo_root in [
                    ("vllm-ascend", config.VLLM_ASCEND_ROOT),
                    ("vllm", config.VLLM_ROOT),
                ]:
                    if len([p for p in usage_parts if "[dispatch]" in p]) >= 2:
                        break
                    hits = self._grep_repo(repo_root, dispatch_pat)
                    for h in hits:
                        if len([p for p in usage_parts if "[dispatch]" in p]) >= 2:
                            break
                        # Per-file cluster cap (same constraint as Step 4)
                        if file_cluster_counts.get(h["file"], 0) >= config.MAX_CLUSTERS_PER_FILE:
                            continue
                        cluster_key = (h["file"], h["line"] // config.CLUSTER_WINDOW_LINES)
                        if cluster_key in seen_clusters:
                            continue
                        # Skip low-quality sources
                        path_parts = set(h["file"].lower().split("/"))
                        if path_parts & _NOISE_DIRS:
                            continue
                        seen_clusters.add(cluster_key)
                        file_cluster_counts[h["file"]] = file_cluster_counts.get(h["file"], 0) + 1
                        file_path = repo_root / h["file"]
                        scope_text = read_source_scope(
                            file_path, h["line"],
                            max_lines=config.MAX_USAGE_SCOPE_LINES)
                        version = self._version_tag(repo_name, h["file"])
                        usage_parts.append(
                            f"### {version} [dispatch] {h['file']}:{h['line']}\n{scope_text}")

        usage_contexts = "\n".join(usage_parts) if usage_parts else (
            "# No usage locations found.")

        # Step 6: Search vllm-ascend docs for parameter explanations
        doc_contexts = self._search_docs(param)

        return def_context, usage_contexts, doc_contexts


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Thin wrapper around Anthropic API for parameter analysis.

    Configuration priority: ~/.claude/settings.json > environment variables
    > config.py defaults. This allows the CLI's own settings (including custom
    API endpoints and auth tokens) to propagate to the analysis pipeline.

    Uses AsyncAnthropic so LLM calls don't block the event loop, enabling
    true concurrency (e.g. 15 parallel requests).
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

        # Load ~/.claude/settings.json for API credentials and model
        settings = self._load_claude_settings()

        api_key = settings.get("api_key")
        base_url = settings.get("base_url")
        model = settings.get("model", config.LLM_MODEL)

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncAnthropic(**client_kwargs)
        self.model = model
        self.max_tokens = config.LLM_MAX_TOKENS
        self.timeout = config.LLM_TIMEOUT
        self.max_retries = config.LLM_MAX_RETRIES
        self.temperature = config.LLM_TEMPERATURE

    @staticmethod
    def _load_claude_settings() -> dict:
        """Load LLM configuration from ~/.claude/settings.json.

        Returns a dict with keys: api_key, base_url, model.
        Falls back to environment variables when settings.json is missing
        or incomplete.
        """
        import os
        settings_path = Path.home() / ".claude" / "settings.json"
        result: dict = {}

        # 1. Try settings.json
        try:
            if settings_path.exists():
                data = json.loads(settings_path.read_text(encoding="utf-8"))
                env = data.get("env", {})
                result["api_key"] = env.get("ANTHROPIC_AUTH_TOKEN")
                result["base_url"] = env.get("ANTHROPIC_BASE_URL")
                result["model"] = (
                    env.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
                    or env.get("ANTHROPIC_MODEL")
                )
                if result.get("api_key"):
                    self_logger = logging.getLogger("parse_params")
                    self_logger.info(
                        "Loaded LLM config from %s (model=%s)",
                        settings_path, result.get("model", "default"),
                    )
        except (json.JSONDecodeError, OSError) as e:
            logging.getLogger("parse_params").warning(
                "Failed to read %s: %s", settings_path, e,
            )

        # 2. Fall back to environment variables
        if not result.get("api_key"):
            result["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
        if not result.get("base_url"):
            result["base_url"] = os.environ.get("ANTHROPIC_BASE_URL")
        if not result.get("model"):
            result["model"] = os.environ.get("ANTHROPIC_MODEL")

        # Strip None values
        return {k: v for k, v in result.items() if v}

    async def analyze(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to Claude and return the response text.

        Retries on transient errors with exponential backoff.
        """
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    timeout=self.timeout,
                )
                if config.LLM_DISABLE_THINKING:
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

                response = await self.client.messages.create(**kwargs)
                # Extract text from response. Skip thinking/reasoning blocks
                # (still needed as fallback if thinking=disabled isn't respected).
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        return block.text
                # Fallback: try any block with text-like content
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text or ""
                    if hasattr(block, "thinking"):
                        continue  # skip thinking blocks
                self.logger.warning(
                    "No text content in response (blocks=%d)", len(response.content),
                )
                return ""

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                wait = 2 ** attempt
                self.logger.warning(
                    "API error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, self.max_retries + 1, e, wait,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(wait)

            except APIStatusError as e:
                if e.status_code >= 500:
                    # Server error — retry
                    last_error = e
                    wait = 2 ** attempt
                    self.logger.warning(
                        "Server error %d (attempt %d/%d): retrying in %ds",
                        e.status_code, attempt + 1, self.max_retries + 1, wait,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait)
                else:
                    # Client error (4xx) — don't retry
                    self.logger.error("Non-retryable API error %d: %s", e.status_code, e)
                    raise

        raise RuntimeError(f"Max retries exceeded. Last error: {last_error}")


# =============================================================================
# YAML validator and writer
# =============================================================================

class YAMLWriter:
    """Validates and writes parameter YAML files to the output directory."""

    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger

    def validate_and_parse(self, yaml_text: str, param: dict) -> dict:
        """Parse YAML text and validate against schema.

        Returns a dict with keys:
          - valid: bool
          - data: ParameterYAML or SkippedParamYAML if valid
          - error: str if invalid
          - performance_impact: str
          - status: "ok" | "error"
        """
        clean_yaml = extract_yaml_from_response(yaml_text)

        try:
            raw_data = yaml.safe_load(clean_yaml)
        except yaml.YAMLError as e:
            return {
                "valid": False,
                "error": f"YAML parse error: {e}",
                "performance_impact": "none",
                "status": "error",
                "data": None,
            }

        if not isinstance(raw_data, dict):
            return {
                "valid": False,
                "error": f"Expected YAML dict, got {type(raw_data)}",
                "performance_impact": "none",
                "status": "error",
                "data": None,
            }

        # Fix common LLM type mistakes: list items that should be strings
        # but were parsed as dicts because of unquoted colons in prose.
        for field in ["constraints", "caveats"]:
            vals = raw_data.get(field)
            if isinstance(vals, list):
                for i, v in enumerate(vals):
                    if isinstance(v, dict) and len(v) == 1:
                        k, val = next(iter(v.items()))
                        vals[i] = f"{k}: {val}"

        impact = raw_data.get("performance_impact", "none")

        # Try full schema first, fall back to minimal skipped schema
        if impact == "none":
            try:
                data = SkippedParamYAML(**raw_data)
                return {
                    "valid": True,
                    "data": data,
                    "performance_impact": "none",
                    "status": "ok",
                    "error": None,
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Schema validation error (skipped param): {e}",
                    "performance_impact": "none",
                    "status": "error",
                    "data": None,
                }

        try:
            data = ParameterYAML(**raw_data)
            return {
                "valid": True,
                "data": data,
                "performance_impact": impact,
                "status": "ok",
                "error": None,
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Schema validation error: {e}",
                "performance_impact": impact,
                "status": "error",
                "data": None,
            }

    def write_yaml(self, param_data: ParameterYAML | SkippedParamYAML) -> Path:
        """Write a validated parameter to a YAML file.

        Returns:
            Path to the written file.
        """
        safe_name = sanitize_filename(param_data.name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.output_dir / f"{safe_name}.yaml"

        # Convert to dict - handle both Pydantic and SkippedParamYAML
        if isinstance(param_data, SkippedParamYAML):
            data_dict = param_data.model_dump()
        else:
            data_dict = param_data.model_dump(exclude_none=False)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data_dict, f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )

        return file_path


# =============================================================================
# Orchestrator
# =============================================================================

class Stage2Analyzer:
    """Orchestrates Stage 2: processes all parameters through LLM analysis."""

    def __init__(
        self,
        output_dir: Path,
        logs_dir: Path,
        logger: logging.Logger,
    ):
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.logger = logger
        self.prompt_builder = PromptBuilder(logger)
        self.context_reader = ContextReader(logger)
        self.llm_client = LLMClient(logger)
        self.yaml_writer = YAMLWriter(output_dir / "params", logger)

        # Ensure output directories exist
        (output_dir / "params").mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Accumulated results for manifest
        self.results: list[dict] = []
        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)

    async def _analyze_one(self, param: dict, index: int, total: int) -> dict:
        """Analyze a single parameter: read context, call LLM, validate, write.

        Returns a result dict for manifest generation.
        """
        name = param.get("name", "unknown")
        t_start = time.monotonic()

        try:
            # Step 2a: Read source code context (outside semaphore —
            # grep on local files doesn't need rate-limiting).
            def_context, usage_contexts, doc_contexts = (
                self.context_reader.read_contexts(param))

            # Step 2b: Build prompts
            user_prompt = self.prompt_builder.build_user_prompt(
                param, def_context, usage_contexts, doc_contexts,
            )

        except Exception as e:
            elapsed = time.monotonic() - t_start
            self.logger.error(
                "[%d/%d] %s → CONTEXT ERROR (%.1fs): %s",
                index + 1, total, name, elapsed, e,
            )
            return {
                "name": name,
                "performance_impact": "none",
                "status": "error",
                "error": f"Context reading failed: {e}",
            }

        async with self.semaphore:
            try:
                # Step 2c: Call LLM (truly async with AsyncAnthropic)
                response_text = await self.llm_client.analyze(
                    self.prompt_builder.system_prompt, user_prompt
                )

                if not response_text.strip():
                    # LLM returned only thinking blocks or empty content
                    elapsed = time.monotonic() - t_start
                    self.logger.warning(
                        "[%d/%d] %s → EMPTY RESPONSE (%.1fs)",
                        index + 1, total, name, elapsed,
                    )
                    return {
                        "name": name,
                        "performance_impact": "none",
                        "status": "error",
                        "error": "LLM returned empty response (no text blocks)",
                    }

                # Step 2d: Validate YAML
                validation = self.yaml_writer.validate_and_parse(response_text, param)

                if validation["valid"] and validation["data"] is not None:
                    impact = validation["performance_impact"]

                    if impact == "none":
                        # Don't write YAML files for non-performance params
                        elapsed = time.monotonic() - t_start
                        self.logger.info(
                            "[%d/%d] %s → none (%.1fs) → skipped",
                            index + 1, total, name, elapsed,
                        )
                    else:
                        # Auto-populate analysis_date for full ParameterYAML
                        if isinstance(validation["data"], ParameterYAML):
                            validation["data"].analysis_date = str(date.today())
                        file_path = self.yaml_writer.write_yaml(validation["data"])
                        elapsed = time.monotonic() - t_start
                        self.logger.info(
                            "[%d/%d] %s → %s (%.1fs) → %s",
                            index + 1, total, name, impact, elapsed,
                            file_path.name,
                        )

                    result = {
                        "name": name,
                        "performance_impact": impact,
                        "status": "ok",
                    }
                else:
                    elapsed = time.monotonic() - t_start
                    self.logger.warning(
                        "[%d/%d] %s → VALIDATION ERROR (%.1fs): %s",
                        index + 1, total, name, elapsed,
                        validation.get("error", "unknown"),
                    )
                    # Save raw LLM output for debugging
                    self._save_error_response(name, response_text)
                    result = {
                        "name": name,
                        "performance_impact": "none",
                        "status": "error",
                        "error": validation.get("error", "unknown"),
                    }

            except Exception as e:
                elapsed = time.monotonic() - t_start
                self.logger.error(
                    "[%d/%d] %s → ERROR (%.1fs): %s\n%s",
                    index + 1, total, name, elapsed, e,
                    traceback.format_exc(),
                )
                result = {
                    "name": name,
                    "performance_impact": "none",
                    "status": "error",
                    "error": str(e),
                }

        return result

    async def run(
        self,
        stage2_params: list[dict],
        progress_manager,
    ) -> list[dict]:
        """Run Stage 2 analysis on all provided parameters.

        Args:
            stage2_params: Parameters passed from Stage 1.
            progress_manager: ProgressManager instance for resumption.

        Returns:
            List of result dicts (for manifest generation).
        """
        # Record params that were already done BEFORE this run (for resume)
        already_done_names = {
            p["name"] for p in stage2_params
            if progress_manager.is_processed(p["name"])
        }

        # Filter out already-processed params (resume support)
        pending = progress_manager.get_pending(stage2_params)
        total = len(pending)

        if total == 0:
            self.logger.info("All %d Stage 2 parameters already processed.",
                             len(stage2_params))
            return self._reconstruct_results(stage2_params, progress_manager)

        self.logger.info("Stage 2: analyzing %d parameters (concurrency=%d)",
                         total, config.LLM_CONCURRENCY)

        # Process parameters with concurrency control
        tasks = []
        for i, param in enumerate(pending):
            task = self._analyze_one(param, i, total)
            tasks.append(task)

        # Gather results as they complete
        results = []
        completed_count = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed_count += 1

            # Update progress and mark processed
            if result["status"] == "ok":
                progress_manager.mark_processed(result["name"])
            else:
                progress_manager.mark_error(result["name"])

            # Periodic save
            if completed_count % config.SAVE_PROGRESS_EVERY == 0:
                progress_manager.save()
                summary = progress_manager.get_progress_summary()
                self.logger.info(
                    "Progress: %d/%d completed, %d errors",
                    summary["completed"], summary["total"], summary["errors"],
                )

        # Final save
        progress_manager.save()

        # Merge: results from this run + params already done in prior runs
        previous = self._reconstruct_results(
            [p for p in stage2_params if p["name"] in already_done_names],
            progress_manager,
        )
        all_results = previous + results

        # Write error log
        self._write_error_log(all_results)
        # Write skipped log
        self._write_skipped_log(all_results)

        return all_results

    def _reconstruct_results(self, stage2_params: list[dict],
                             progress_manager) -> list[dict]:
        """Reconstruct result dicts for parameters already processed in a prior run."""
        results = []
        for param in stage2_params:
            name = param["name"]
            if progress_manager.is_processed(name):
                # We don't know the exact impact without reading the YAML file
                # Try to infer from the output directory
                impact = self._read_impact_from_file(name)
                results.append({
                    "name": name,
                    "performance_impact": impact,
                    "status": "ok",
                })
        return results

    def _read_impact_from_file(self, param_name: str) -> str:
        """Read the performance_impact from an already-written YAML file."""
        safe_name = sanitize_filename(param_name)
        candidate = self.output_dir / "params" / f"{safe_name}.yaml"
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data.get("performance_impact", "none")
            except Exception:
                pass
        return "none"

    def _save_error_response(self, param_name: str, raw_text: str) -> None:
        """Save raw LLM output for debugging when YAML validation fails."""
        import re as _re
        safe = _re.sub(r'[<>:"/\\|?*]', '_', param_name)[:80]
        path = self.logs_dir / "raw_errors" / f"{safe}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(raw_text, encoding="utf-8")
        except OSError:
            pass

    def _write_error_log(self, results: list[dict]) -> None:
        """Write analysis_errors.json for failed parameters."""
        errors = [r for r in results if r.get("status") == "error"]
        if errors:
            error_path = self.logs_dir / "analysis_errors.json"
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            self.logger.warning("%d analysis errors written to %s", len(errors), error_path)

    def _write_skipped_log(self, results: list[dict]) -> None:
        """Write stage2_skipped.json for Stage 2 parameters judged as none."""
        skipped = [r for r in results if r.get("performance_impact") == "none"
                   and r.get("status") == "ok"]
        if skipped:
            skip_path = self.logs_dir / "stage2_skipped.json"
            with open(skip_path, "w", encoding="utf-8") as f:
                json.dump(skipped, f, indent=2, ensure_ascii=False)
            self.logger.info("%d skipped params (performance=none) written to %s",
                             len(skipped), skip_path)
