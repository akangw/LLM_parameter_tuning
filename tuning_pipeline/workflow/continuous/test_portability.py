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
from workflow.continuous.runtime_profile import (
    apply_topology_baseline_binding,
    resolve_runtime_profile,
    validate_runtime_selections,
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

    def test_executor_constraints_reject_unsupported_topology(self) -> None:
        profile = {
            "status": "integrated",
            "executor": "ktp_two_role",
            "nodes": 4,
            "npu_per_node": 8,
            "data_parallel_size": 4,
            "data_parallel_size_local": 1,
            "tensor_parallel_size": 8,
            "data_parallel_rpc_port": 12980,
            "worker_replicas": 3,
            "worker_data_parallel_start_rank": 1,
        }
        with self.assertRaisesRegex(ValueError, "incompatible with executor"):
            resolve_topology_profile(
                {"topology": {"profile": "future", "resolved_profile": profile}},
                Path(__file__).resolve().parents[2],
            )

    def test_single_node_local_dp_profile_has_no_worker_rank(self) -> None:
        root = Path(__file__).resolve().parents[2]
        resolved, profile = resolve_topology_profile(
            {
                "topology": {
                    "profile": "a3_single_16npu_dp2local_tp8",
                    "profiles_file": "workflow/continuous/topology_profiles.yaml",
                }
            },
            root,
        )
        self.assertEqual(profile["nodes"], 1)
        self.assertEqual(profile["data_parallel_size_local"], 2)
        self.assertEqual(profile["tensor_parallel_size"], 8)
        self.assertEqual(profile["worker_replicas"], 0)
        self.assertEqual(profile["worker_data_parallel_start_rank"], 0)
        self.assertEqual(
            profile["resolved_executor"]["remote_contract"],
            "single_node_local_dp_v1",
        )
        self.assertEqual(
            resolved["topology"]["profile"], "a3_single_16npu_dp2local_tp8"
        )

    def test_distributed_local_dp_profile_resolves_integrated_executor(self) -> None:
        root = Path(__file__).resolve().parents[2]
        resolved, profile = resolve_topology_profile(
            {
                "topology": {
                    "profile": "a3_dp4_tp8",
                    "profiles_file": "workflow/continuous/topology_profiles.yaml",
                }
            },
            root,
        )
        self.assertEqual(4, profile["data_parallel_size"])
        self.assertEqual(2, profile["data_parallel_size_local"])
        self.assertEqual(2, profile["worker_data_parallel_start_rank"])
        self.assertEqual(
            "distributed_local_dp_v1",
            profile["resolved_executor"]["remote_contract"],
        )
        self.assertEqual("a3_dp4_tp8", resolved["topology"]["profile"])


class RuntimeProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]

    def test_default_profile_groups_and_hashes_critical_artifacts(self) -> None:
        config = yaml.safe_load(
            (Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8")
        )
        resolved, identity = resolve_runtime_profile(config, self.project_root)
        self.assertEqual(identity["profile"], "glm52_w8a8_a3_dp4_tp8_a8_guided_v4")
        self.assertEqual(resolved["topology"]["profile"], "a3_dp4_tp8")
        self.assertFalse(resolved["topology_campaign"]["enabled"])
        self.assertEqual(
            resolved["benchmark"]["profile"], "aligned_fast_c32_v2"
        )
        self.assertEqual(
            resolved["search_space"]["profile"],
            "automatic_registry_a8_frontier_v3",
        )
        self.assertIn("scenario", identity["artifacts"])
        self.assertEqual(len(identity["artifacts"]["scenario"]["sha256"]), 64)

    def test_dp4_topology_binds_probe_baseline_instead_of_dp2_a8(self) -> None:
        config = yaml.safe_load(
            (Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8")
        )
        config["runtime"]["profile"] = "glm52_w8a8_a3_topology_campaign_v4"
        config["strategy"]["profile"] = "hierarchical_agentic_frontier_v3"
        config["benchmark"]["profile"] = "aligned_fast_c32_v1"
        config["topology_campaign"]["enabled"] = True
        resolved, _ = resolve_runtime_profile(config, self.project_root)
        resolved["topology"]["profile"] = "a3_dp4_tp8"
        resolved = apply_topology_baseline_binding(resolved)
        self.assertEqual(
            "a8_dp4_tp8_probe_v1", resolved["initial_baseline"]["label"]
        )
        self.assertTrue(
            resolved["initial_baseline"]["definition"].endswith(
                "a8_glm52_w8a8_dp4_tp8_probe_v1.yaml"
            )
        )
        validate_runtime_selections(resolved)

    def test_fixed_dp4_runtime_owns_a8_derived_baseline_and_executor(self) -> None:
        config = yaml.safe_load(
            (Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8")
        )
        config["runtime"]["profile"] = "glm52_w8a8_a3_dp4_tp8_a8_fixed_v1"
        config["strategy"]["profile"] = "hierarchical_agentic_frontier_v3"
        config["benchmark"]["profile"] = "aligned_fast_c32_v1"
        resolved, identity = resolve_runtime_profile(config, self.project_root)
        self.assertEqual("a3_dp4_tp8", resolved["topology"]["profile"])
        self.assertEqual(
            "ktp_two_role_local_dp", identity["compatibility"]["executor"]
        )
        self.assertEqual(
            "a8_dp4_tp8_fixed_v1", resolved["initial_baseline"]["label"]
        )
        self.assertTrue(
            resolved["initial_baseline"]["definition"].endswith(
                "a8_glm52_w8a8_dp4_tp8_fixed_v1.yaml"
            )
        )
        baseline = yaml.safe_load(
            (self.project_root / resolved["initial_baseline"]["definition"]).read_text(
                encoding="utf-8"
            )
        )
        parameters = baseline["reference_parameters"]
        self.assertEqual(64, parameters["max_num_seqs"])
        self.assertEqual(64000, parameters["max_model_len"])
        self.assertEqual(1, parameters["num_speculative_tokens"])
        self.assertEqual(64, parameters["max_cudagraph_capture_size"])
        validate_runtime_selections(resolved)

    def test_controller_freeze_does_not_reapply_bindings_over_explicit_choice(self) -> None:
        config = yaml.safe_load(
            (Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8")
        )
        config["benchmark"]["profile"] = "vllm_bench_public_v1"
        resolved, _ = resolve_runtime_profile(
            config, self.project_root, apply_bindings=False
        )
        self.assertEqual(resolved["benchmark"]["profile"], "vllm_bench_public_v1")
        validate_runtime_selections(resolved)

    def test_w8_runtime_rejects_w4_profiles_before_session_creation(self) -> None:
        config = yaml.safe_load(
            (Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8")
        )
        resolved, _ = resolve_runtime_profile(config, self.project_root)
        resolved["search_space"]["profile"] = "automatic_registry_glm52_w4a8c8_v1"
        with self.assertRaisesRegex(ValueError, "incompatible with runtime"):
            validate_runtime_selections(resolved)

        resolved, _ = resolve_runtime_profile(config, self.project_root)
        resolved["benchmark"]["profile"] = "aligned_l1_glm52_w4a8c8_v1"
        with self.assertRaisesRegex(ValueError, "incompatible with runtime"):
            validate_runtime_selections(resolved)

    def test_implicit_runtime_keeps_its_frozen_selections(self) -> None:
        config = {
            "benchmark": {"profile": "aligned_l1_v4"},
            "search_space": {"profile": "automatic_registry_v1"},
        }
        resolved, identity = resolve_runtime_profile(config, self.project_root)
        self.assertEqual(identity["profile"], "legacy_implicit_ascend")
        self.assertEqual(resolved["benchmark"]["profile"], "aligned_l1_v4")

    def test_planned_external_adapter_cannot_activate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "adapter.yaml"
            adapter.write_text(
                yaml.safe_dump(
                    {
                        "adapter": {
                            "name": "future",
                            "status": "planned",
                            "platform": "ascend",
                            "model_contract": {
                                "family": "glm",
                                "variant": "future",
                                "weight_format": "bf16",
                            },
                            "config": {},
                            "readiness": {"blockers": ["executor not implemented"]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "executor not implemented"):
                resolve_runtime_profile(
                    {"runtime": {"adapter_file": str(adapter)}}, self.project_root
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
