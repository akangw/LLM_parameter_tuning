from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from .executor_adapter import (
    EXECUTOR_ADAPTER_API_VERSION,
    CommandExecutorAdapter,
    resolve_executor_adapter,
    validate_snapshot,
)
from . import continuous_tuning as tuning
from .test_continuous_tuning import config as _test_config


class ExecutorAdapterUnitTests(unittest.TestCase):
    def test_legacy_default_does_not_load_external_code(self) -> None:
        adapter, identity = resolve_executor_adapter(
            {"execution_mode": "ktp_lab"}, tuning.KB_ROOT
        )
        self.assertIsNone(adapter)
        self.assertEqual("legacy_builtin", identity["kind"])
        self.assertEqual("ktp_lab", identity["name"])

    def test_project_bridge_is_fingerprinted(self) -> None:
        setting = {
            "execution_mode": "executor_adapter",
            "executor_adapter": {
                "kind": "command_v1",
                "name": "template-test",
                "bridge_path": "workflow/executor_adapters/template_executor_bridge.py",
                "allowlisted_roots": ["workflow/executor_adapters"],
            },
        }
        adapter, identity = resolve_executor_adapter(setting, tuning.KB_ROOT)
        self.assertIsNotNone(adapter)
        self.assertEqual("template-test", identity["name"])
        self.assertEqual(64, len(identity["bridge_sha256"]))
        self.assertEqual(64, len(identity["sha256"]))

    def test_bridge_cannot_escape_allowlisted_root(self) -> None:
        setting = {
            "execution_mode": "executor_adapter",
            "executor_adapter": {
                "bridge_path": "workflow/continuous/continuous_tuning.py",
                "allowlisted_roots": ["workflow/executor_adapters"],
            },
        }
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_executor_adapter(setting, tuning.KB_ROOT)

    def test_inline_adapter_secret_is_rejected(self) -> None:
        setting = {
            "execution_mode": "executor_adapter",
            "executor_adapter": {
                "bridge_path": "workflow/executor_adapters/template_executor_bridge.py",
                "allowlisted_roots": ["workflow/executor_adapters"],
                "config": {"api_key": "must-not-be-archived"},
            },
        }
        with self.assertRaisesRegex(ValueError, "inline secret"):
            resolve_executor_adapter(setting, tuning.KB_ROOT)

    def test_command_bridge_protocol_and_snapshot_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bridge = Path(temporary) / "bridge.py"
            bridge.write_text(
                "import json,sys\n"
                "r=json.load(sys.stdin)\n"
                "o={'api_version':r['api_version'],'ok':True,"
                "'snapshot':{'status':'Running','active_pods':2,"
                "'terminal':False,'partial_failure':False}}\n"
                "json.dump(o,sys.stdout)\n",
                encoding="utf-8",
            )
            adapter = CommandExecutorAdapter(
                bridge_path=bridge,
                setting={"python_command": sys.executable},
                identity={"name": "unit"},
            )
            result = adapter.invoke("snapshot", context={}, payload={"task_id": "x"})
            snapshot = validate_snapshot(result)
            self.assertEqual(2, snapshot["active_pods"])
            self.assertFalse(snapshot["terminal"])

    def test_snapshot_schema_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            validate_snapshot(
                {
                    "snapshot": {
                        "status": "Running",
                        "active_pods": 1,
                        "terminal": "no",
                        "partial_failure": False,
                    }
                }
            )


class ControllerExecutorAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        configured = _test_config()
        configured["execution_mode"] = "executor_adapter"
        configured["executor_adapter"] = {
            "kind": "command_v1",
            "name": "controller-test",
            "bridge_path": "workflow/executor_adapters/template_executor_bridge.py",
            "allowlisted_roots": ["workflow/executor_adapters"],
            "capabilities": {"start_benchmark": True},
        }
        self.controller = tuning.Controller(configured)

    def test_adapter_ready_path_does_not_query_ktp_lab(self) -> None:
        self.controller.executor_adapter.invoke = Mock(
            return_value={
                "api_version": EXECUTOR_ADAPTER_API_VERSION,
                "ok": True,
                "message": "scheduler ready",
            }
        )
        with patch.object(self.controller, "validate_deployment_configuration"), patch.object(
            self.controller, "validate_runtime_configuration"
        ), patch.object(
            self.controller,
            "ensure_lab_available",
            side_effect=AssertionError("ktp-lab path must remain isolated"),
        ):
            self.assertEqual("scheduler ready", self.controller.check_ready())

    def test_adapter_submission_records_frozen_identity(self) -> None:
        self.controller.executor_adapter.invoke = Mock(
            return_value={
                "api_version": EXECUTOR_ADAPTER_API_VERSION,
                "ok": True,
                "task_id": "slurm-123",
                "run_id": "a0_external",
                "message": "submitted",
                "task": {"scheduler": "slurm", "job_id": "123"},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            round_dir = Path(temporary)
            for child in ("02_parameters", "03_submission"):
                (round_dir / child).mkdir()
            task_id, run_id = self.controller.submit_executor_adapter(
                round_dir,
                "a0",
                self.controller.config["baseline"],
                dry_run=False,
            )
            self.assertEqual("slurm-123", task_id)
            self.assertEqual("a0_external", run_id)
            evidence = json.loads(
                (round_dir / "03_submission" / "submission.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                self.controller.executor_identity, evidence["executor_identity"]
            )

    def test_resume_rejects_changed_executor_identity(self) -> None:
        state = {
            "image_identity": self.controller.image_identity,
            "runtime_identity": self.controller.runtime_identity,
            "executor_identity": {"sha256": "different"},
        }
        with self.assertRaisesRegex(RuntimeError, "executor-adapter identity"):
            self.controller.assert_state_image_identity(state)


if __name__ == "__main__":
    unittest.main()
