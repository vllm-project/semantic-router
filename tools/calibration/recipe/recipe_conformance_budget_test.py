"""Disposable conformance deployments honor authored evaluation budgets."""

import argparse
import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
conformance = importlib.import_module("recipe_conformance")
runtime = importlib.import_module("recipe_conformance_runtime")


class RuntimePreviewBudgetTest(unittest.TestCase):
    def test_budget_only_fills_omitted_preview_leaf(self):
        authored = {
            "global": {
                "services": {"api": {"routing_preview": {"max_concurrency": 3}}},
                "model_catalog": {"signal_timeout_ms": 1700},
            }
        }
        original = copy.deepcopy(authored)
        for timeout, expected in ((300, 300), (300.9, 300), (1, 1), (1200, 1200)):
            with self.subTest(timeout=timeout):
                actual = runtime.with_runtime_preview_budget(
                    authored, {"evaluation": {"request_timeout_seconds": timeout}}
                )
                wanted = copy.deepcopy(authored)
                wanted["global"]["services"]["api"]["routing_preview"][
                    "request_timeout_seconds"
                ] = expected
                self.assertEqual(actual, wanted)
                self.assertEqual(authored, original)

    def test_explicit_preview_budget_and_omitted_manifest_preserve_defaults(self):
        for timeout in (1, 100, 500):
            authored = {
                "global": {
                    "services": {
                        "api": {"routing_preview": {"request_timeout_seconds": timeout}}
                    }
                }
            }
            self.assertEqual(
                runtime.with_runtime_preview_budget(
                    authored, {"evaluation": {"request_timeout_seconds": 300}}
                ),
                authored,
            )
        self.assertEqual(runtime.with_runtime_preview_budget({}, {}), {})
        self.assertEqual(
            runtime.with_runtime_preview_budget({}, {"evaluation": {"concurrency": 4}}),
            {},
        )

    def test_invalid_budget_does_not_write_runtime_config(self):
        recipe = conformance.DEFAULT_RECIPE_ROOT / "built-in/latest/mom-v1"
        for timeout in (0, 1201, "invalid", float("nan"), float("inf"), float("-inf")):
            with tempfile.TemporaryDirectory() as directory, self.subTest(
                timeout=timeout
            ):
                output = Path(directory) / "runtime/config.yaml"
                with self.assertRaisesRegex(ValueError, "request_timeout_seconds|NaN"):
                    runtime.prepare_builtin_runtime(
                        recipe,
                        output,
                        conformance.REPO_ROOT,
                        {"evaluation": {"request_timeout_seconds": timeout}},
                    )
                self.assertFalse(output.exists())
                self.assertFalse(output.parent.exists())

    def test_standalone_prepare_applies_budget_without_changing_source(self):
        recipe = conformance.DEFAULT_RECIPE_ROOT / "accuracy"
        source = {
            path.name: path.read_bytes() for path in recipe.iterdir() if path.is_file()
        }
        authored = conformance.load_yaml_mapping(recipe / "config.yaml")
        manifest, _probes = conformance.load_probe_manifest(recipe / "probes.yaml")
        for explicit in (None, 100):
            with tempfile.TemporaryDirectory() as directory, self.subTest(
                explicit=explicit
            ):
                config = copy.deepcopy(authored)
                if explicit is not None:
                    config.setdefault("global", {}).setdefault(
                        "services", {}
                    ).setdefault("api", {})["routing_preview"] = {
                        "request_timeout_seconds": explicit,
                        "max_concurrency": 2,
                    }
                output = Path(directory) / "runtime.yaml"
                args = argparse.Namespace(
                    recipes_root=recipe.parent, recipe=recipe.name, config=output
                )
                # Preserve real manifest/inventory validation and filesystem output;
                # only supply the alternative authored Preview policy in this case.
                with patch.object(
                    conformance, "load_yaml_mapping", return_value=config
                ):
                    self.assertEqual(conformance.command_prepare_runtime(args), 0)
                actual = yaml.safe_load(output.read_text())
                expected = copy.deepcopy(config)
                preview = (
                    expected.setdefault("global", {})
                    .setdefault("services", {})
                    .setdefault("api", {})
                    .setdefault("routing_preview", {})
                )
                preview.setdefault(
                    "request_timeout_seconds",
                    int(manifest["evaluation"]["request_timeout_seconds"]),
                )
                self.assertEqual(actual, expected)
                self.assertEqual(
                    source,
                    {
                        path.name: path.read_bytes()
                        for path in recipe.iterdir()
                        if path.is_file()
                    },
                )

    def test_invalid_standalone_budget_leaves_no_output(self):
        source = conformance.DEFAULT_RECIPE_ROOT / "accuracy"
        for timeout in (0, float("nan"), float("inf")):
            with tempfile.TemporaryDirectory() as directory, self.subTest(
                timeout=timeout
            ):
                root = Path(directory)
                recipe = root / "recipes" / "accuracy"
                recipe.mkdir(parents=True)
                for path in source.iterdir():
                    if path.is_file():
                        (recipe / path.name).write_bytes(path.read_bytes())
                manifest = yaml.safe_load((recipe / "probes.yaml").read_text())
                manifest["evaluation"]["request_timeout_seconds"] = timeout
                (recipe / "probes.yaml").write_text(yaml.safe_dump(manifest))
                output = root / "runtime" / "config.yaml"
                args = argparse.Namespace(
                    recipes_root=recipe.parent, recipe=recipe.name, config=output
                )
                with self.assertRaises(ValueError):
                    conformance.command_prepare_runtime(args)
                self.assertFalse(output.exists())
                self.assertFalse(output.parent.exists())


if __name__ == "__main__":
    unittest.main()
