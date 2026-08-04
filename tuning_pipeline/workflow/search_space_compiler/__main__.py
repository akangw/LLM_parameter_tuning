from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .compiler import MODULE_DIR, PROJECT_ROOT, SearchSpaceCompiler, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile an offline, auditable search-space proposal. "
            "This command never connects to the remote server or edits the live controller."
        )
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=MODULE_DIR / "scenario.glm52-a3-aligned-l1.yaml",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=PROJECT_ROOT / "tag_params" / "output" / "params",
    )
    parser.add_argument("--registry", type=Path, default=MODULE_DIR / "registry.yaml")
    parser.add_argument("--policy", type=Path, default=MODULE_DIR / "policy.yaml")
    parser.add_argument(
        "--history",
        type=Path,
        help=(
            "Read-only prior trial history: legacy history_input.json or "
            "a normalized object containing a trials list"
        ),
    )
    parser.add_argument(
        "--previous-selection",
        type=Path,
        help="Previous compiled search-space result used for bounded rotation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New local output directory; defaults under workflow/search_space_compiler/runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and print the summary without writing any files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compiler = SearchSpaceCompiler(
        knowledge_dir=args.knowledge_dir,
        scenario_path=args.scenario,
        registry_path=args.registry,
        policy_path=args.policy,
        history_path=args.history,
        previous_selection_path=args.previous_selection,
    )
    result = compiler.compile()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("active:", ", ".join(result["active_search_limits"]))
    if result["rotation_audit"]["swaps"]:
        for swap in result["rotation_audit"]["swaps"]:
            print(
                "rotate:",
                swap["out"],
                "->",
                swap["in"],
                f"(margin={swap['score_margin']})",
            )
    if args.dry_run:
        print("dry-run: no files written")
        return 0
    output = args.output or (
        MODULE_DIR
        / "runs"
        / ("search_space_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    files = write_outputs(result, output)
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
