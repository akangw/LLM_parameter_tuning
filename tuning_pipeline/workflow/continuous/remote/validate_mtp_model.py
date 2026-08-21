#!/usr/bin/env python3
"""Validate and freeze the small MTP config before starting vLLM.

Only ``config.json`` is read. Model weights are never opened by this preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


def read_config(model_path: Path) -> tuple[Path, dict[str, Any], str]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"MTP_CONFIG_INVALID: missing {config_path}")
    raw = config_path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"MTP_CONFIG_INVALID: {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"MTP_CONFIG_INVALID: {config_path} must contain an object")
    return config_path, value, hashlib.sha256(raw).hexdigest()


def identity(model_path: Path, tokens: int) -> dict[str, Any]:
    config_path, config, config_sha256 = read_config(model_path)
    n_predict = config.get("n_predict")
    if n_predict is not None and (
        isinstance(n_predict, bool)
        or not isinstance(n_predict, int)
        or n_predict <= 0
    ):
        raise SystemExit(
            "MTP_CONFIG_INVALID: n_predict must be a positive integer when present"
        )
    if tokens <= 0:
        raise SystemExit("MTP_CONFIG_INVALID: num_speculative_tokens must be positive")
    if tokens > 15:
        raise SystemExit(
            "MTP_CONFIG_INVALID: Ascend fused TND decode requires "
            "num_speculative_tokens <= 15"
        )
    if n_predict is not None and tokens > n_predict and tokens % n_predict != 0:
        raise SystemExit(
            "MTP_CONFIG_INVALID: num_speculative_tokens="
            f"{tokens} must be divisible by n_predict={n_predict} when K exceeds it"
        )
    return {
        "schema_version": 1,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "n_predict": n_predict,
        "num_speculative_tokens": tokens,
        "compatible": True,
    }


def freeze(current: dict[str, Any], path: Path) -> None:
    stable = {
        key: current[key]
        for key in ("schema_version", "model_path", "config_sha256", "n_predict")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(stable, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return
    except FileExistsError:
        pass
    # Master and worker tasks can reach this shared file almost together.  The
    # exclusive creator writes only a tiny record, but a loser must still avoid
    # mistaking an in-progress write for identity corruption.
    frozen: Any = None
    last_error: Exception | None = None
    for _ in range(20):
        try:
            frozen = json.loads(path.read_text(encoding="utf-8"))
            last_error = None
            break
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise SystemExit(
            f"MTP_IDENTITY_INVALID: cannot read {path}: {last_error}"
        ) from last_error
    if frozen != stable:
        raise SystemExit(
            "MTP_IDENTITY_DRIFT: model path/config digest/n_predict differs from "
            f"the frozen identity {path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--num-speculative-tokens", type=int, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--frozen-identity", type=Path, required=True)
    args = parser.parse_args()
    result = identity(args.model_path, args.num_speculative_tokens)
    freeze(result, args.frozen_identity)
    args.identity_output.parent.mkdir(parents=True, exist_ok=True)
    args.identity_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
