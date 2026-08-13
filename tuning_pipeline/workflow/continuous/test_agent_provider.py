from __future__ import annotations

import os
import json
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from . import agent_provider
from .agent_provider import (
    _extract_json,
    run_structured_agent,
    validate_agent_credentials,
)


class AgentProviderTests(unittest.TestCase):
    @staticmethod
    def _codex_completed(output_path: Path, content: str = '{"ok": true}'):
        def complete(*args, **kwargs):
            output_path.write_text(content, encoding="utf-8")
            return subprocess.CompletedProcess(args[0], 0, "events", "")

        return complete

    def test_extract_json_accepts_fenced_and_prefixed_output(self) -> None:
        self.assertEqual({"ok": True}, _extract_json("```json\n{\"ok\": true}\n```"))
        self.assertEqual({"ok": True}, _extract_json("result:\n{\"ok\": true}"))

    def test_api_provider_requires_credential_environment_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            schema.write_text('{"type":"object"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "credential environment"):
                run_structured_agent(
                    {"provider": "anthropic", "settings": {"api_key_env": "VLLMTKB_TEST_MISSING_KEY"}},
                    prompt="test", schema_path=schema, output_path=root / "output.json",
                    cwd=root, allowed_dir=root,
                )

    def test_startup_credential_validation_fails_before_agent_call(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TEST_PROVIDER_KEY"):
                validate_agent_credentials({
                    "provider": "openai_compatible",
                    "settings": {"api_key_env": "TEST_PROVIDER_KEY"},
                })

    def test_startup_validation_checks_codex_executable(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(agent_provider.shutil, "which", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "Codex CLI was not found"):
                validate_agent_credentials({
                    "provider": "codex", "settings": {"command": "auto"}
                })

    def test_codex_profile_is_explicit_opt_in_to_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            schema.write_text('{"type":"object"}', encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(agent_provider.shutil, "which", return_value="/bin/codex"),
                patch.object(
                    agent_provider.subprocess,
                    "run",
                    side_effect=self._codex_completed(root / "output.json"),
                ) as run,
            ):
                result = run_structured_agent(
                    {
                        "provider": "codex",
                        "settings": {
                            "command": "auto",
                            "profile": "deepseek-v4-flash",
                            "ephemeral": True,
                        },
                    },
                    prompt="test",
                    schema_path=schema,
                    output_path=root / "output.json",
                    cwd=root,
                    allowed_dir=root,
                )
            command = run.call_args.args[0]
            self.assertEqual(0, result.returncode)
            self.assertIn("--profile", command)
            self.assertIn("deepseek-v4-flash", command)
            self.assertIn("--ephemeral", command)
            self.assertNotIn("--ignore-user-config", command)
            self.assertIn("read-only", command)

    def test_codex_without_profile_still_ignores_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            schema.write_text('{"type":"object"}', encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(agent_provider.shutil, "which", return_value="/bin/codex"),
                patch.object(
                    agent_provider.subprocess,
                    "run",
                    side_effect=self._codex_completed(root / "output.json"),
                ) as run,
            ):
                run_structured_agent(
                    {"provider": "codex", "settings": {"command": "auto"}},
                    prompt="test",
                    schema_path=schema,
                    output_path=root / "output.json",
                    cwd=root,
                    allowed_dir=root,
                )
            command = run.call_args.args[0]
            self.assertIn("--ignore-user-config", command)
            self.assertNotIn("--profile", command)

    def test_codex_can_explicitly_use_server_managed_base_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codex_home = root / "codex-home"
            codex_home.mkdir()
            tmp_dir = root / "codex-tmp"
            tmp_dir.mkdir()
            schema = root / "schema.json"
            schema.write_text('{"type":"object"}', encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(agent_provider.shutil, "which", return_value="/bin/codex"),
                patch.object(
                    agent_provider.subprocess,
                    "run",
                    side_effect=self._codex_completed(root / "output.json"),
                ) as run,
            ):
                run_structured_agent(
                    {
                        "provider": "codex",
                        "settings": {
                            "command": "auto",
                            "codex_home": str(codex_home),
                            "tmp_dir": str(tmp_dir),
                            "use_user_config": True,
                            "ephemeral": True,
                        },
                    },
                    prompt="test",
                    schema_path=schema,
                    output_path=root / "output.json",
                    cwd=root,
                    allowed_dir=root,
                )
            command = run.call_args.args[0]
            environment = run.call_args.kwargs["env"]
            self.assertNotIn("--ignore-user-config", command)
            self.assertNotIn("--profile", command)
            self.assertEqual(str(codex_home), environment["CODEX_HOME"])
            self.assertEqual(str(tmp_dir), environment["TMPDIR"])

    def test_codex_normalizes_prefixed_fenced_json_and_preserves_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "decision.json"
            schema = root / "schema.json"
            schema.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}},
                        "additionalProperties": False,
                    }
                ),
                encoding="utf-8",
            )
            raw = 'Analysis follows.\n```json\n{"ok": true}\n```\n'
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(agent_provider.shutil, "which", return_value="/bin/codex"),
                patch.object(
                    agent_provider.subprocess,
                    "run",
                    side_effect=self._codex_completed(output, raw),
                ),
            ):
                result = run_structured_agent(
                    {"provider": "codex", "settings": {"command": "auto"}},
                    prompt="test",
                    schema_path=schema,
                    output_path=output,
                    cwd=root,
                    allowed_dir=root,
                )
            self.assertEqual(0, result.returncode)
            self.assertEqual({"ok": True}, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(raw, (root / "decision.json.raw.txt").read_text(encoding="utf-8"))

    def test_schema_forbidden_agent_metadata_is_pruned_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "decision.json"
            schema = root / "schema.json"
            schema.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": ["changes"],
                        "additionalProperties": False,
                        "properties": {
                            "changes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["parameter"],
                                    "additionalProperties": False,
                                    "properties": {"parameter": {"type": "string"}},
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            agent_provider._validate_and_write(
                {"changes": [{"parameter": "x", "candidate": "annotation"}]},
                schema,
                output,
            )
            self.assertEqual(
                {"changes": [{"parameter": "x"}]},
                json.loads(output.read_text(encoding="utf-8")),
            )
            audit = json.loads(
                (root / "decision.json.normalization.json").read_text(encoding="utf-8")
            )
            self.assertEqual(["$.changes[0].candidate"], audit["removed_paths"])

    def test_normalization_does_not_coerce_invalid_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            schema.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}},
                        "additionalProperties": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                agent_provider._validate_and_write(
                    {"ok": "true", "note": "drop me"}, schema, root / "output.json"
                )

    def test_codex_rejects_unchanged_stale_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "decision.json"
            output.write_text('{"ok": true}', encoding="utf-8")
            schema = root / "schema.json"
            schema.write_text('{"type":"object"}', encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(agent_provider.shutil, "which", return_value="/bin/codex"),
                patch.object(
                    agent_provider.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, "events", ""),
                ),
                patch.object(agent_provider.time, "monotonic", side_effect=[0, 2]),
                patch.object(agent_provider.time, "sleep"),
            ):
                result = run_structured_agent(
                    {
                        "provider": "codex",
                        "settings": {"command": "auto", "output_wait_seconds": 1},
                    },
                    prompt="test",
                    schema_path=schema,
                    output_path=output,
                    cwd=root,
                    allowed_dir=root,
                )
            self.assertEqual(1, result.returncode)
            self.assertIn("unchanged", result.stderr)

    def test_codex_waits_for_delayed_last_message_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "decision.json"
            schema = root / "schema.json"
            schema.write_text('{"type":"object"}', encoding="utf-8")

            def delayed_complete(*args, **kwargs):
                def write_later():
                    time.sleep(0.05)
                    output.write_text('prefix\n```json\n{"ok": true}\n```', encoding="utf-8")

                threading.Thread(target=write_later, daemon=True).start()
                return subprocess.CompletedProcess(args[0], 0, "events", "")

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(agent_provider.shutil, "which", return_value="/bin/codex"),
                patch.object(agent_provider.subprocess, "run", side_effect=delayed_complete),
            ):
                result = run_structured_agent(
                    {
                        "provider": "codex",
                        "settings": {"command": "auto", "output_wait_seconds": 2},
                    },
                    prompt="test",
                    schema_path=schema,
                    output_path=output,
                    cwd=root,
                    allowed_dir=root,
                )
            self.assertEqual(0, result.returncode)
            self.assertEqual({"ok": True}, json.loads(output.read_text(encoding="utf-8")))

    def test_codex_home_must_exist(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "codex_home"):
            validate_agent_credentials(
                {
                    "provider": "codex",
                    "settings": {
                        "command": "auto",
                        "codex_home": "/definitely/missing/codex-home",
                    },
                }
            )

    def test_codex_profile_name_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Codex profile"):
            validate_agent_credentials(
                {
                    "provider": "codex",
                    "settings": {"command": "auto", "profile": "../unsafe"},
                }
            )

    def test_command_provider_requires_a_command(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires agent.settings.command"):
            validate_agent_credentials({"provider": "command", "settings": {}})

    def test_deepseek_uses_json_mode_and_retries_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            schema.write_text(
                json.dumps(
                    {
                        "type": "object",
                        "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}},
                        "additionalProperties": False,
                    }
                ),
                encoding="utf-8",
            )
            responses = [
                {"choices": [{"message": {"content": ""}}]},
                {"choices": [{"message": {"content": '{"ok": true}'}}]},
            ]
            captured: list[dict] = []

            def fake_http(url, headers, body, timeout):
                captured.append({"url": url, "headers": headers, "body": body})
                return responses.pop(0)

            with (
                patch.dict(os.environ, {"DEEPSEEK_TEST_KEY": "secret"}, clear=True),
                patch.object(agent_provider, "_http_json", side_effect=fake_http),
                patch.object(agent_provider.time, "sleep"),
            ):
                result = run_structured_agent(
                    {
                        "provider": "deepseek",
                        "settings": {
                            "api_key_env": "DEEPSEEK_TEST_KEY",
                            "model": "deepseek-v4-flash",
                            "thinking": "enabled",
                            "max_api_retries": 1,
                        },
                    },
                    prompt="return json",
                    schema_path=schema,
                    output_path=root / "output.json",
                    cwd=root,
                    allowed_dir=root,
                )

            self.assertEqual(0, result.returncode)
            self.assertEqual({"ok": True}, json.loads((root / "output.json").read_text()))
            self.assertEqual(2, len(captured))
            self.assertEqual(
                {"type": "json_object"}, captured[0]["body"]["response_format"]
            )
            self.assertEqual(
                {"type": "enabled"}, captured[0]["body"]["thinking"]
            )
            self.assertNotIn("secret", json.dumps(captured[0]["body"]))

    def test_unknown_provider_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            schema.write_text('{"type":"object"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Unsupported Agent provider"):
                run_structured_agent(
                    {"provider": "unknown"}, prompt="test", schema_path=schema,
                    output_path=root / "output.json", cwd=root, allowed_dir=root,
                )


if __name__ == "__main__":
    unittest.main()
