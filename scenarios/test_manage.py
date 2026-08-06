import unittest

from scenarios import manage


class ScenarioCatalogTests(unittest.TestCase):
    def test_all_packages_validate(self) -> None:
        discovered = manage.discover()
        self.assertGreaterEqual(len(discovered), 2)
        for package, scenario in discovered.values():
            self.assertEqual(manage.validate_one(package, scenario), [])

    def test_w8_and_w4_are_sibling_isolated_scenarios(self) -> None:
        discovered = manage.discover()
        w8_package, w8 = discovered["glm52-w8a8-a3-2n-dp2-tp16"]
        w4_package, w4 = discovered["glm52-w4a8c8-a3-1n-dp2-tp8"]
        self.assertEqual(w8_package.parent, w4_package.parent)
        self.assertNotEqual(w8["entry"]["runtime_root"], w4["entry"]["runtime_root"])
        self.assertNotEqual(w8["artifacts"]["baseline"], w4["artifacts"]["baseline"])
        self.assertEqual(w8["status"], "integrated")
        self.assertEqual(w4["status"], "planned")

    def test_artifact_report_separates_shared_and_runtime_outputs(self) -> None:
        discovered = manage.discover()
        package, scenario = discovered["glm52-w8a8-a3-2n-dp2-tp16"]
        report = manage.artifact_report(scenario["id"], package, scenario)
        self.assertGreater(report["shared_knowledge"]["parameter_portraits"]["yaml_count"], 0)
        self.assertGreater(report["shared_knowledge"]["parameter_tags"]["yaml_count"], 0)
        self.assertIn("baseline", report["fixed_artifacts"])
        self.assertIn("sessions", report["runtime"])
        self.assertTrue(report["artifact_index"].endswith("ARTIFACTS.md"))


if __name__ == "__main__":
    unittest.main()
