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


class ServiceRenderTests(unittest.TestCase):
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
