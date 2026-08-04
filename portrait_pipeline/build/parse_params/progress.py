"""Progress file management for resumable processing."""

from __future__ import annotations

import json
import logging
from pathlib import Path


class ProgressManager:
    """Manages progress.json for resumable Stage 2 processing.

    Tracks which parameters have been processed (successfully or with errors),
    enabling resume after interruption without re-processing completed items.
    """

    def __init__(self, progress_path: Path, logger: logging.Logger):
        self.path = progress_path
        self.logger = logger
        self.data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logger.info(
                    "Loaded progress: %d processed, %d errors",
                    len(data.get("processed_params", [])),
                    len(data.get("error_params", [])),
                )
                return data
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning("Corrupted progress file: %s — starting fresh", e)
        return self._empty_state()

    def _empty_state(self) -> dict:
        return {
            "stage1_passed": [],
            "stage1_skipped": [],
            "processed_params": [],    # list of param names fully processed
            "error_params": [],        # list of param names that errored
            "total_stage2": 0,
            "completed_stage2": 0,
        }

    def set_stage1_results(self, passed: list[dict], skipped: list[dict]) -> None:
        """Record Stage 1 results."""
        self.data["stage1_passed"] = [p["name"] for p in passed]
        self.data["stage1_skipped"] = [
            {"name": p["name"], "skip_reason": p.get("skip_reason", "unknown")}
            for p in skipped
        ]
        self.data["total_stage2"] = len(passed)
        self.data["completed_stage2"] = 0
        self.data["processed_params"] = []
        self.data["error_params"] = []
        self._save()

    def is_processed(self, param_name: str) -> bool:
        """Check if a parameter has already been processed."""
        return param_name in self.data.get("processed_params", [])

    def mark_processed(self, param_name: str) -> None:
        """Mark a parameter as successfully processed."""
        if param_name not in self.data["processed_params"]:
            self.data["processed_params"].append(param_name)
        self.data["completed_stage2"] = len(self.data["processed_params"])

    def mark_error(self, param_name: str) -> None:
        """Mark a parameter as having errored during processing.

        Errors are saved to disk immediately to ensure they are not lost.
        """
        if param_name not in self.data["error_params"]:
            self.data["error_params"].append(param_name)
            self._save()

    def get_pending(self, stage2_params: list[dict]) -> list[dict]:
        """Return only parameters not yet processed."""
        processed_set = set(self.data.get("processed_params", []))
        error_set = set(self.data.get("error_params", []))
        return [
            p for p in stage2_params
            if p["name"] not in processed_set and p["name"] not in error_set
        ]

    def get_progress_summary(self) -> dict:
        """Return a summary dict for logging."""
        total = self.data.get("total_stage2", 0)
        completed = self.data.get("completed_stage2", 0)
        return {
            "total": total,
            "completed": completed,
            "remaining": total - completed,
            "errors": len(self.data.get("error_params", [])),
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def save(self) -> None:
        """Public save method."""
        self._save()
