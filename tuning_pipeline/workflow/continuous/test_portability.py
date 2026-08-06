from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflow.continuous.image_identity_cli import build_documents
from workflow.continuous.session_bundle import (
    export_session,
    import_session,
    inspect_bundle,
)
from workflow.continuous.topology_profile import resolve_topology_profile


class TopologyProfileTests(unittest.TestCase):
    def test_legacy_config_keeps_exact_executor_defaults(self) -> None:
        resolved, profile = resolve_topology_profile({}, Path("."))
        self.assertEqual(profile["nodes"], 2)
        self.assertEqual(profile["tensor_parallel_size"], 16)
        self.assertEqual(resolved["topology"]["profile"], "legacy_a3_dp2_tp16")

    def test_unintegrated_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profiles.yaml").write_text(
                yaml.safe_dump(
                    {
                        "default_profile": "future",
                        "profiles": {"future": {"status": "planned"}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not integrated"):
                resolve_topology_profile(
                    {"topology": {"profiles_file": "profiles.yaml"}}, root
                )


class SessionBundleTests(unittest.TestCase):
    def test_round_trip_is_verified_and_non_activating_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_runtime = root / "source"
            session = source_runtime / "experiments" / "session-1"
            session.mkdir(parents=True)
            (session / "session_config.yaml").write_text("answer: 42\n", encoding="utf-8")
            (source_runtime / "state.json").write_text(
                json.dumps(
                    {
                        "session_id": "session-1",
                        "session_dir": str(session),
                        "status": "tuning_complete",
                    }
                ),
                encoding="utf-8",
            )
            bundle = root / "session.zip"
            export_session(source_runtime, bundle)
            manifest, _ = inspect_bundle(bundle)
            self.assertEqual(manifest["schema"], "vllmtkb-session-bundle/v1")
            target_runtime = root / "target"
            restored = import_session(bundle, target_runtime)
            self.assertEqual(
                (restored / "session_config.yaml").read_text(encoding="utf-8"),
                "answer: 42\n",
            )
            self.assertFalse((target_runtime / "state.json").exists())

    def test_active_export_requires_explicit_snapshot_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            session = runtime / "experiments" / "live"
            session.mkdir(parents=True)
            (runtime / "state.json").write_text(
                json.dumps(
                    {"session_id": "live", "session_dir": str(session), "status": "running"}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "allow-active-snapshot"):
                export_session(runtime, runtime / "live.zip")


class ImageIdentityTests(unittest.TestCase):
    def test_probe_builds_matching_manifest_and_activation(self) -> None:
        commit_a = "a" * 40
        commit_b = "b" * 40
        existing = {
            "parameter_portrait": {
                "vllm_commit": commit_a,
                "vllm_ascend_commit": commit_b,
            }
        }
        probe = {
            "image": "registry.example/team/image:v1",
            "digest": "sha256:" + "c" * 64,
            "size_bytes": 123,
            "vllm": {"package": "1.0", "commit": commit_a},
            "vllm_ascend": {"package": "2.0", "commit": commit_b},
            "platform": {"os": "linux"},
            "evidence": {"source_commit_probe": "digest-qualified runtime probe"},
        }
        manifest, activation = build_documents(probe, existing, "tester")
        self.assertEqual(manifest["target_image"]["digest"], activation["target"]["image_digest"])
        self.assertEqual(activation["approved_by"], "tester")

    def test_probe_cannot_silently_change_portrait_commit(self) -> None:
        probe = {
            "image": "registry.example/team/image:v1",
            "digest": "sha256:" + "c" * 64,
            "size_bytes": 123,
            "vllm": {"package": "1.0", "commit": "b" * 40},
            "vllm_ascend": {"package": "2.0", "commit": "c" * 40},
            "evidence": {"source_commit_probe": "probe"},
        }
        with self.assertRaisesRegex(ValueError, "migrate the knowledge artifacts"):
            build_documents(
                probe,
                {
                    "parameter_portrait": {
                        "vllm_commit": "a" * 40,
                        "vllm_ascend_commit": "c" * 40,
                    }
                },
                "tester",
            )


if __name__ == "__main__":
    unittest.main()
