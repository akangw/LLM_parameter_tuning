import unittest
from pathlib import Path

import yaml

from workflow.continuous.runtime_profile import (
    resolve_runtime_profile,
    validate_runtime_selections,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


class RuntimeProfileSelectionOverrideTests(unittest.TestCase):
    def base_config(self) -> dict:
        return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))

    def test_allowed_automatic_search_profile_survives_adapter_defaults(self) -> None:
        config = self.base_config()
        config["search_space"]["profile"] = "automatic_registry_v1"
        resolved, _ = resolve_runtime_profile(config, PROJECT_ROOT)
        self.assertEqual(
            resolved["search_space"]["profile"], "automatic_registry_v1"
        )
        validate_runtime_selections(resolved)

    def test_incompatible_selection_survives_until_fail_closed_validation(self) -> None:
        config = self.base_config()
        config["search_space"]["profile"] = "automatic_registry_glm52_w4a8c8_v1"
        resolved, _ = resolve_runtime_profile(config, PROJECT_ROOT)
        with self.assertRaisesRegex(ValueError, "incompatible with runtime"):
            validate_runtime_selections(resolved)


if __name__ == "__main__":
    unittest.main()
