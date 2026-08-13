import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "service_runtime", HERE / "server_autonomous" / "service_runtime.py"
)
service_runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(service_runtime)


class ServiceRecoveryDecisionTests(unittest.TestCase):
    def write_state(self, runtime: Path, **values: object) -> None:
        state = {
            "status": "running",
            "session_dir": str(runtime / "sessions" / "s1"),
            "round_index": 0,
            "round_label": "b0",
            "active_task_id": None,
            "active_run_id": None,
        }
        state.update(values)
        (runtime / "state.json").write_text(json.dumps(state), encoding="utf-8")

    def test_auto_starts_when_state_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            action, _ = service_runtime.decide(Path(temp_dir), "auto")
            self.assertEqual(action, "--start")

    def test_auto_resumes_an_active_task_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            self.write_state(runtime, active_task_id="task-1", active_run_id="run-1")
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "--resume")

    def test_pause_blocks_even_when_active_ids_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            self.write_state(
                runtime,
                status="paused_for_human",
                active_task_id="task-1",
                active_run_id="run-1",
            )
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "blocked")

    def test_recoverable_controller_error_resumes_terminal_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            session = runtime / "sessions" / "s1"
            results = session / "round_008_a8" / "05_results"
            results.mkdir(parents=True)
            (results / "metrics.json").write_text("{}", encoding="utf-8")
            self.write_state(
                runtime,
                status="recovering_controller_error",
                round_index=8,
                round_label="a8",
                active_task_id="task-1",
                active_run_id="run-1",
            )
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "--resume")

    def test_legacy_agent_protocol_pause_is_migrated_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            session = runtime / "sessions" / "s1"
            results = session / "round_008_a8" / "05_results"
            results.mkdir(parents=True)
            (results / "metrics.json").write_text("{}", encoding="utf-8")
            self.write_state(
                runtime,
                status="paused_controller_error",
                round_index=8,
                round_label="a8",
                controller_error="RuntimeError: codex Agent analysis failed: schema-valid JSON",
            )
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "--resume")

    def test_unclassified_controller_pause_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            self.write_state(
                runtime,
                status="paused_controller_error",
                controller_error="RuntimeError: unknown invariant violation",
            )
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "blocked")

    def test_stop_marker_blocks_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            (runtime / "STOP_REQUESTED").touch()
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "blocked")

    def test_completed_session_exits_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            self.write_state(runtime, status="completed_by_agent")
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "complete")

    def test_archived_round_requires_terminal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            session = runtime / "sessions" / "s1"
            results = session / "round_000_b0" / "05_results"
            results.mkdir(parents=True)
            (results / "metrics.json").write_text("{}", encoding="utf-8")
            self.write_state(runtime, status="stopped_after_current_round")
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "--resume")

    def test_partial_active_identity_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            self.write_state(runtime, active_task_id="task-1")
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "blocked")

    def test_pending_submission_transaction_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            self.write_state(
                runtime,
                status="running",
                pending_submission={
                    "round_index": 2,
                    "round_label": "a2",
                    "candidate": {"x": 1},
                },
            )
            action, reason = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "--resume")
            self.assertIn("submission", reason)

    def test_corrupt_primary_state_uses_last_known_good_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            self.write_state(runtime, active_task_id="task-1", active_run_id="run-1")
            (runtime / "state.json.previous").write_text(
                (runtime / "state.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (runtime / "state.json").write_text('{"status":', encoding="utf-8")
            action, reason = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "--resume")
            self.assertIn("recovering active", reason)

    def test_corrupt_primary_and_backup_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Path(temp_dir)
            (runtime / "state.json").write_text("{", encoding="utf-8")
            (runtime / "state.json.previous").write_text("{", encoding="utf-8")
            action, _ = service_runtime.decide(runtime, "auto")
            self.assertEqual(action, "blocked")

    def test_status_lease_prefers_frozen_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "runtime"
            runtime.mkdir()
            (root / "base.yaml").write_text(
                "lab:\n  lease_name: base-lease\n", encoding="utf-8"
            )
            config = root / "config.yaml"
            config.write_text("base_config: base.yaml\n", encoding="utf-8")
            self.write_state(runtime, lease_name="session-lease")
            self.assertEqual(
                "session-lease",
                service_runtime.resolve_lease_name(runtime, config),
            )

    def test_status_lease_falls_back_to_merged_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "runtime"
            runtime.mkdir()
            (root / "base.yaml").write_text(
                "lab:\n  lease_name: base-lease\n", encoding="utf-8"
            )
            config = root / "config.yaml"
            config.write_text(
                "base_config: base.yaml\nlab:\n  lease_name: configured-lease\n",
                encoding="utf-8",
            )
            self.assertEqual(
                "configured-lease",
                service_runtime.resolve_lease_name(runtime, config),
            )


class ServiceRenderTests(unittest.TestCase):
    def test_entrypoints_prefer_private_config_without_changing_default(self) -> None:
        autonomous = HERE / "server_autonomous"
        common = (autonomous / "common.sh").read_text(encoding="utf-8")
        self.assertIn('LOCAL_CONFIG="${SCRIPT_DIR}/config.local.yaml"', common)
        self.assertIn('[[ -f "${LOCAL_CONFIG}" ]]', common)
        self.assertIn('CONFIG="${DEFAULT_CONFIG}"', common)
        self.assertIn('CONFIG="${VLLMTKB_CONFIG}"', common)

        example = autonomous / "config.local.example.yaml"
        merged = service_runtime._load_config(example)
        self.assertEqual("server_autonomous", merged["operation_mode"])
        self.assertEqual("deepseek", merged["agent"]["provider"])
        self.assertEqual(
            "vllm_bench_public_v1", merged["benchmark"]["profile"]
        )
        self.assertEqual(
            "automatic_registry_v1", merged["search_space"]["profile"]
        )

        runner = (autonomous / "run_foreground.sh").read_text(encoding="utf-8")
        service = (autonomous / "service.sh").read_text(encoding="utf-8")
        self.assertIn("AUTO_RETRY_PAUSED_REQUEST", runner)
        self.assertIn('ACTION="--auto-retry-paused-current"', runner)
        self.assertIn("auto-retry-paused)", service)

    def test_rendered_configs_have_no_placeholders(self) -> None:
        repo_root = HERE.parents[2]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "service"
            paths = service_runtime.render(
                repo_root, Path(temp_dir) / "controller.env", output
            )
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertNotIn("@REPO_ROOT@", path.read_text(encoding="utf-8"))
            unit = (output / "vllmtkb-server-autonomous.service").read_text(
                encoding="utf-8"
            )
            self.assertIn("RestartPreventExitStatus=78", unit)
            supervisor = (output / "supervisord.conf").read_text(encoding="utf-8")
            self.assertIn("exitcodes=0,78", supervisor)
            socket_lines = [
                line.split("=", 1)[1]
                for line in supervisor.splitlines()
                if line.startswith("file=")
            ]
            self.assertEqual(len(socket_lines), 1)
            self.assertLess(len(socket_lines[0].encode()), 104)
            self.assertIn(f"serverurl=unix://{socket_lines[0]}", supervisor)
            self.assertNotIn(f"unix://{output}", supervisor)


if __name__ == "__main__":
    unittest.main()
