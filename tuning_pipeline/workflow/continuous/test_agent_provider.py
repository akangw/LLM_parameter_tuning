from __future__ import annotations

import os
import json
import tempfile
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
