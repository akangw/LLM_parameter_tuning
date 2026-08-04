from __future__ import annotations

import os
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
