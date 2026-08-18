import importlib
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

recipe_conformance = importlib.import_module("recipe_conformance")
recipe_sources = importlib.import_module("recipe_conformance_sources")


class BuiltInRecipeConformanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = (
            recipe_conformance.REPO_ROOT / "config" / "recipes" / "built-in" / "latest"
        )

    def test_latest_bundles_use_the_same_five_file_contract(self) -> None:
        inventory = recipe_conformance.discover_inventory(self.root)

        by_name = {recipe.name: recipe for recipe in inventory}
        self.assertIn("mom-v1", by_name)
        self.assertTrue(all(recipe.coverage["passed"] for recipe in inventory))
        mom = by_name["mom-v1"]
        self.assertEqual(len(mom.entrypoints), 5)
        self.assertEqual(len(mom.decisions), 43)
        self.assertEqual(mom.variants, 222)
        self.assertTrue(mom.coverage["passed"])

    def test_live_plan_covers_every_recipe_from_every_live_source(self) -> None:
        inventories = recipe_sources.discover_source_inventories(
            recipe_conformance.DEFAULT_RECIPE_ROOT,
            recipe_conformance.discover_inventory,
        )

        first = recipe_sources.source_matrix_payload(
            inventories, 3, recipe_conformance.REPO_ROOT
        )
        second = recipe_sources.source_matrix_payload(
            inventories, 3, recipe_conformance.REPO_ROOT
        )

        self.assertEqual(first, second)
        expected = {
            (source_inventory.source.name, recipe.name)
            for source_inventory in inventories
            for recipe in source_inventory.recipes
        }
        planned = {
            (row["source"], recipe)
            for row in first["include"]
            for recipe in row["recipes"].split(",")
        }
        self.assertEqual(planned, expected)
        built_in = [
            row for row in first["include"] if row["source"] == "built-in-latest"
        ]
        mom_row = next(row for row in built_in if "mom-v1" in row["recipes"].split(","))
        self.assertEqual(mom_row["recipes_root"], "config/recipes/built-in/latest")
        self.assertEqual(mom_row["report_dir"], "built-in/latest")

    def test_source_discovery_excludes_immutable_catalog_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "built-in" / "latest").mkdir(parents=True)
            (root / "built-in" / "v0.4").mkdir()

            sources = recipe_conformance.discover_recipe_sources(root)

        self.assertEqual(
            [source.name for source in sources],
            ["standalone", "built-in-latest"],
        )
        self.assertEqual(
            sources[1].recipes_root,
            root / "built-in" / "latest",
        )

    def test_built_in_management_auth_exposes_readiness_credential(self) -> None:
        config = recipe_conformance.load_yaml_mapping(self.root / "mom-v1/config.yaml")

        bindings = recipe_conformance.management_auth_bindings(config)

        self.assertEqual(
            bindings,
            [("VLLM_SR_DASHBOARD_RECIPE_TOKEN", True)],
        )


if __name__ == "__main__":
    unittest.main()
