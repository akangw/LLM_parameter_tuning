#!/usr/bin/env python3
"""Export one immutable Session's attempted-history view for continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuning_pipeline.workflow.continuous import continuous_tuning as tuning


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    session_dir = args.session_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config_path = session_dir / "session_config.yaml"
    if not config_path.is_file():
        raise SystemExit(f"Session config is missing: {config_path}")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite continuation history: {output}")

    runtime_root = session_dir.parent.parent
    tuning.configure_runtime_root(runtime_root)
    controller = tuning.Controller(tuning.load_yaml(config_path))
    history = controller.attempted_history_summary(session_dir)
    if not history:
        raise SystemExit("Session has no attempted history to export")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "entries": len(history),
                "parameter_experiments": sum(
                    bool(item.get("counts_as_parameter_experiment"))
                    for item in history
                ),
                "last_round": history[-1].get("round"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
