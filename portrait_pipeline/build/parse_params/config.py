"""Shared constants, paths, and configuration loaded from resource files."""

from pathlib import Path

# --- Repository paths ---
VLLM_ROOT = Path("/vllm-workspace/vllm")
VLLM_ASCEND_ROOT = Path("/vllm-workspace/vllm-ascend")

# Commit hashes (from build manifest, update when repos change)
SOURCE_COMMIT_VLLM = "bcf2be96"
SOURCE_COMMIT_VLLM_ASCEND = "99e1ea0f"

# --- Module paths ---
MODULE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = MODULE_DIR / "resources"
OUTPUT_DIR_DEFAULT = MODULE_DIR / "output"
LOGS_DIR_DEFAULT = MODULE_DIR / "logs"
PROGRESS_FILE_DEFAULT = MODULE_DIR / "progress.json"

# --- Resource files ---
SCHEMA_YAML_PATH = RESOURCES_DIR / "schema.yaml"
SYSTEM_PROMPT_PATH = RESOURCES_DIR / "system_prompt.txt"
USER_PROMPT_TEMPLATE_PATH = RESOURCES_DIR / "user_prompt_template.txt"
STAGE1_RULES_PATH = RESOURCES_DIR / "stage1_rules.yaml"

# --- LLM configuration ---
LLM_MODEL = "claude-sonnet-4-6"
LLM_JUDGE_MODEL = "claude-haiku-4-5"  # Optional fast judge (currently unused)
LLM_MAX_TOKENS = 16384  # Max observed output ~2K tokens; 4x headroom prevents truncation
LLM_TIMEOUT = 300  # seconds
LLM_MAX_RETRIES = 2
LLM_CONCURRENCY = 15
LLM_TEMPERATURE = 0.0  # For reproducibility
LLM_DISABLE_THINKING = True  # Skip chain-of-thought (saves tokens, faster output)

# --- Nested param prefix → config file mapping ---
# Maps a nested param prefix (e.g. "speculative_config") to its definition file(s)
# per repo. Used to narrow definition search for nested params.
NESTED_PREFIX_TO_FILE = {
    "additional_config":           {"vllm-ascend": "vllm_ascend/ascend_config.py"},
    "attention_config":            {"vllm": "vllm/config/attention.py"},
    "compilation_config":          {"vllm": "vllm/config/compilation.py",
                                    "vllm-ascend": "vllm_ascend/ascend_config.py"},
    "ec_transfer_config":          {"vllm": "vllm/config/ec_transfer.py"},
    "eplb_config":                 {"vllm": "vllm/config/parallel.py",
                                    "vllm-ascend": "vllm_ascend/ascend_config.py"},
    "kernel_config":               {"vllm": "vllm/config/kernel.py"},
    "kv_events_config":            {"vllm": "vllm/config/kv_events.py"},
    "kv_transfer_config":          {"vllm": "vllm/config/kv_transfer.py"},
    "pooler_config":               {"vllm": "vllm/config/pooler.py"},
    "profiler_config":             {"vllm": "vllm/config/profiler.py"},
    "speculative_config":          {"vllm": "vllm/config/speculative.py"},
    "structured_outputs_config":   {"vllm": "vllm/config/structured_outputs.py"},
    "weight_transfer_config":      {"vllm": "vllm/config/weight_transfer.py"},
}

# --- Context reading ---
MAX_USAGE_SCOPE_LINES = 40      # Max lines per usage site scope
MAX_USAGE_LOCATIONS = 18        # Max total usage sites across both repos
MAX_CLUSTERS_PER_FILE = 2       # Max clusters per file (prevents any single large
                                # file from consuming the entire quota)
CLUSTER_WINDOW_LINES = 40       # Line-distance threshold for grouping hits into
                                # the same cluster (one cluster per this-many-lines
                                # block). Hits in the same block are deduplicated.
GREP_MAX_RAW_RESULTS = 5000     # Safety cap on raw grep hits before relevance sort.
                                # Prevents pathological cases (10k+ hits for very
                                # common variable names). Set 0 for unlimited.

# --- File priority for grep result sorting ---
# Rules are evaluated top-to-bottom; first match wins.
# Lower number = higher priority (sorted first in results).
# match_type: "startswith" checks path prefix, "contains" checks substring
# (both case-insensitive, matched against repo-relative path).
FILE_PRIORITY_RULES: list[tuple[int, str, str]] = [
    # Priority 1: engine entry points (arg_utils, CLI parsing)
    (1, "startswith", "vllm/engine"),
    (1, "startswith", "vllm_ascend/engine"),

    # Priority 2: config/definition files (parameter schema, defaults)
    (2, "startswith", "vllm/config"),
    (2, "startswith", "vllm_ascend/ascend_config"),
    (2, "startswith", "vllm_ascend/profiling_config"),

    # Priority 3: platform override/orchestration + v1 engine
    (3, "startswith", "vllm_ascend/platform.py"),
    (3, "startswith", "vllm_ascend/patch/platform"),
    (3, "startswith", "vllm/v1"),
    (3, "startswith", "vllm_ascend/v1"),

    # Priority 4: worker/model_runner (where params are consumed)
    (4, "startswith", "vllm/worker"),
    (4, "startswith", "vllm_ascend/worker"),

    # Priority 5: general library code
    (5, "startswith", "vllm/"),
    (5, "startswith", "vllm_ascend/"),

    # Priority 7: examples
    (7, "contains", "example"),

    # Priority 8-9: non-runtime code (tests, tools, build infra)
    (8, "startswith", "tests/"),
    (8, "startswith", "test_"),
    (9, "startswith", "tools/"),
    (9, "startswith", "benchmarks/"),
    (9, "startswith", "docs/"),
    (9, "startswith", ".github/"),
    (9, "startswith", "ci_log"),
    (9, "startswith", "csrc/"),
]
DEFAULT_FILE_PRIORITY = 6  # Fallback for paths matching none of the above

# --- Doc search ---
# Directories to search for parameter documentation (all .md files recursively)
DOC_SEARCH_DIRS: list[tuple[Path, str]] = [
    # (path, label for context annotation)
    (VLLM_ASCEND_ROOT / "docs" / "source", "vllm-ascend"),
]
# Max doc snippets per parameter, max lines per snippet, max total doc lines
MAX_DOC_SNIPPETS = 5
MAX_DOC_SNIPPET_LINES = 30
MAX_DOC_TOTAL_LINES = 60
# Lines to search backwards/forwards for ``` fence when reading doc context.
# Model tutorial scripts can be 50+ lines; 80 covers nearly all cases.
DOC_FENCE_SEARCH_LINES = 80
# Priority rules for doc file paths (relative to doc root). Lower = higher priority.
DOC_PRIORITY_RULES: list[tuple[int, str, str]] = [
    (1, "startswith", "user_guide/configuration/"),
    (1, "startswith", "user_guide/feature_guide/"),
    (2, "startswith", "developer_guide/performance_and_debug/"),
    (3, "startswith", "tutorials/models/"),
    (4, "startswith", "developer_guide/Design_Documents/"),
    (5, "startswith", "tutorials/features/"),
    (6, "startswith", "developer_guide/"),
    (7, "startswith", "tutorials/"),
]
DEFAULT_DOC_PRIORITY = 8

# --- Progress ---
SAVE_PROGRESS_EVERY = 10  # Save progress every N parameters
