import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

recipe_conformance = importlib.import_module("recipe_conformance")
recipe_coverage = importlib.import_module("recipe_conformance_coverage")
FULL_COVERAGE_PERCENT = 100.0


def write_recipe_contract(path: Path) -> None:
    for filename in recipe_conformance.REQUIRED_RECIPE_FILES:
        (path / filename).write_text("", encoding="utf-8")


class RecipeConformanceTest(unittest.TestCase):
    def test_probe_schema_matches_parser_field_inventory(self) -> None:
        schema_path = (
            recipe_conformance.REPO_ROOT
            / "tools"
            / "agent"
            / "schemas"
            / "recipe-probes-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        manifest_module = importlib.import_module("router_calibration_manifest")

        self.assertEqual(
            set(schema["properties"]),
            set(manifest_module.TOP_LEVEL_FIELDS),
        )
        self.assertEqual(
            set(schema["$defs"]["decision"]["properties"]),
            set(manifest_module.DECISION_FIELDS),
        )
        self.assertEqual(
            set(schema["$defs"]["variant"]["properties"]),
            set(manifest_module.VARIANT_FIELDS),
        )

    def test_repository_catalog_is_discovered_and_valid(self) -> None:
        inventory = recipe_conformance.discover_inventory(
            recipe_conformance.DEFAULT_RECIPE_ROOT
        )
        recipe_conformance.validate_catalog_readme(
            recipe_conformance.DEFAULT_RECIPE_ROOT, inventory
        )

        self.assertGreaterEqual(len(inventory), 7)
        self.assertGreaterEqual(sum(recipe.variants for recipe in inventory), 275)
        self.assertTrue(all(recipe.decisions for recipe in inventory))
        by_name = {recipe.name: recipe for recipe in inventory}
        self.assertEqual(
            by_name["accuracy"].auto_entrypoints,
            ("vllm-sr/auto",),
        )
        self.assertEqual(len(by_name["accuracy"].entrypoints), 1)
        self.assertEqual(by_name["multi-objective"].auto_entrypoints, ())
        self.assertEqual(len(by_name["multi-objective"].entrypoints), 5)
        self.assertTrue(all(recipe.coverage["passed"] for recipe in inventory))
        self.assertTrue(
            all(
                recipe.coverage["algorithms"]["percent"] == FULL_COVERAGE_PERCENT
                for recipe in inventory
            )
        )
        self.assertTrue(
            all(
                recipe.coverage["plugins"]["percent"] == FULL_COVERAGE_PERCENT
                for recipe in inventory
            )
        )

    def test_json_schema_rejects_invalid_manifest_types(self) -> None:
        manifest_module = importlib.import_module("router_calibration_manifest")
        source = (
            recipe_conformance.DEFAULT_RECIPE_ROOT / "accuracy" / "probes.yaml"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tempdir:
            manifest = Path(tempdir) / "probes.yaml"
            manifest.write_text(
                source.replace("concurrency: 4", "concurrency: 0"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "recipe probe schema"):
                manifest_module.load_probe_manifest(manifest)

    def test_live_tag_policy_is_a_real_gate(self) -> None:
        evaluation = {
            "results": [
                {
                    "matched": False,
                    "tags": ["language:en"],
                    "messages": [],
                    "tools": [],
                }
            ]
        }

        result = recipe_coverage.evaluate_live_tag_policy(
            evaluation,
            {"min_tag_pass_rate": {"language": 100.0}},
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["categories"]["language"]["pass_rate"], 0.0)

    def test_static_coverage_policy_blocks_regression(self) -> None:
        coverage = {
            "signals": {"percent": 50.0},
            "projections": {"percent": 100.0},
            "algorithms": {"percent": 100.0},
            "plugins": {"percent": 100.0},
            "request_shapes": ["text"],
            "tag_counts": {"negative": 1},
        }
        policy = {
            "min_signal_assertion_percent": 51.0,
            "required_request_shapes": ["text", "tools"],
            "min_tag_counts": {"negative": 2},
        }

        errors = recipe_coverage.static_policy_errors(coverage, policy)

        self.assertEqual(len(errors), 3)

    def test_sharding_is_deterministic_and_complete(self) -> None:
        inventory = recipe_conformance.discover_inventory(
            recipe_conformance.DEFAULT_RECIPE_ROOT
        )

        first = recipe_conformance.matrix_payload(inventory, 3)
        second = recipe_conformance.matrix_payload(inventory, 3)

        self.assertEqual(first, second)
        planned = {
            name for shard in first["include"] for name in shard["recipes"].split(",")
        }
        self.assertEqual(planned, {recipe.name for recipe in inventory})

    def test_incomplete_recipe_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recipe = Path(tempdir) / "incomplete"
            recipe.mkdir()
            (recipe / "config.yaml").write_text("version: v0.3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "four-file contract"):
                recipe_conformance.validate_recipe_directory(recipe)

    def test_runtime_recipe_directory_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recipe = Path(tempdir) / "complete"
            recipe.mkdir()
            write_recipe_contract(recipe)
            (recipe / ".vllm-sr").mkdir()

            recipe_conformance.validate_recipe_directory(recipe)

    def test_unexpected_recipe_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recipe = Path(tempdir) / "complete"
            recipe.mkdir()
            write_recipe_contract(recipe)
            (recipe / "unexpected").mkdir()

            with self.assertRaisesRegex(ValueError, "extra=\\['unexpected'\\]"):
                recipe_conformance.validate_recipe_directory(recipe)

    def test_default_entrypoints_are_bound_round_robin(self) -> None:
        config = {
            "global": {
                "router": {
                    "auto_model_name": "custom-auto",
                }
            }
        }
        probes = [
            recipe_conformance.Probe(
                decision_id="route",
                variant_id=str(index),
                probe_id=f"route:{index}",
                expected_decision="route",
            )
            for index in range(4)
        ]

        bound = recipe_conformance.bind_default_entrypoints(config, probes)

        self.assertEqual(
            [probe.model for probe in bound],
            ["vllm-sr/auto", "auto", "custom-auto", "vllm-sr/auto"],
        )

    def test_consolidated_report_marks_missing_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            report_root = Path(tempdir)
            (report_root / "inventory.json").write_text(
                json.dumps(
                    {
                        "schema_version": "v1",
                        "recipes": [{"name": "alpha"}, {"name": "beta"}],
                    }
                ),
                encoding="utf-8",
            )
            alpha = report_root / "alpha"
            alpha.mkdir()
            (alpha / "eval-report.json").write_text(
                json.dumps(
                    {
                        "inventory": {"name": "alpha"},
                        "evaluation": {
                            "matched": 3,
                            "total": 3,
                            "passed": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = recipe_conformance.build_consolidated_report(report_root)

            self.assertEqual(report["summary"]["passed_recipes"], 1)
            self.assertEqual(report["summary"]["missing_recipes"], 1)
            self.assertFalse(report["summary"]["passed"])


if __name__ == "__main__":
    unittest.main()
