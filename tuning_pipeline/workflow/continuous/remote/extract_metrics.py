#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def normalize_key(label: str) -> str:
    key = label.strip().lower()
    key = re.sub(r"\([^)]*\)", "", key)
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key


def parse_value(raw: str):
    value = raw.strip().replace(",", "")
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value):
        return float(value)
    return raw.strip()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_metrics.py FORMAL_LOG METRICS_JSON", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    text = source.read_text(encoding="utf-8", errors="replace")

    metrics = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([^:]{2,80}):\s*(.+?)\s*$", line)
        if not match:
            continue
        label, raw_value = match.groups()
        key = normalize_key(label)
        if any(
            token in key
            for token in (
                "throughput",
                "ttft",
                "tpot",
                "itl",
                "latency",
                "duration",
                "successful_requests",
                "failed_requests",
                "input_tokens",
                "generated_tokens",
            )
        ):
            metrics[key] = parse_value(raw_value)

    payload = {
        "source_log": str(source),
        "metrics": metrics,
        "parse_status": "ok" if metrics else "no_metrics_matched",
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
