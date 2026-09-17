import importlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

recipe_conformance = importlib.import_module("recipe_conformance")
recipe_coverage = importlib.import_module("recipe_conformance_coverage")
recipe_metadata_schema = importlib.import_module("recipe_metadata_schema")
FULL_COVERAGE_PERCENT = 100.0


def write_recipe_contract(path: Path) -> None:
    for filename in recipe_conformance.REQUIRED_RECIPE_FILES:
        (path / filename).write_text("", encoding="utf-8")


class RecipeConformanceTest(unittest.TestCase):
    def test_latest_built_in_bundle_uses_the_same_five_file_contract(self) -> None:
        root = (
            recipe_conformance.REPO_ROOT / "config" / "recipes" / "built-in" / "latest"
        )

        inventory = recipe_conformance.discover_inventory(root)

        self.assertEqual([recipe.name for recipe in inventory], ["mom-v1"])
        mom = inventory[0]
        # Built-in packages describe reusable routing policy. Entrypoints and
        # model assignments are composed by the Dashboard when a Mixture is
        # created, so the package itself stays model- and deployment-neutral.
        self.assertEqual(len(mom.entrypoints), 0)
        self.assertEqual(
            set(mom.decisions),
            {
                "balance:simple",
                "balance:medium",
                "balance:reasoning",
                "speed:fast",
                "speed:tools",
                "speed:reasoning",
                "cost:economy",
                "cost:tools",
                "cost:reasoning",
                "accuracy:simple",
                "accuracy:reasoning",
                "accuracy:review",
                "accuracy:agent",
                "vault:guard",
                "vault:sensitive",
                "vault:private",
            },
        )
        self.assertEqual(len(mom.decisions), 16)
        self.assertEqual(mom.variants, 315)
        self.assertEqual(
            mom.coverage["signals"]["by_family"]["classifier"]["asserted"],
            ["classifier:content-risk"],
        )
        self.assertEqual(set(mom.algorithms), {"multi_factor", "fusion", "workflows"})
        self.assertTrue(mom.coverage["passed"])

    def test_default_discovery_skips_the_nested_built_in_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "maintained"
            recipe.mkdir()
            write_recipe_contract(recipe)
            (root / "built-in").mkdir()

            self.assertEqual(
                recipe_conformance.discover_recipe_directories(root), [recipe]
            )

    def test_metadata_schema_has_the_required_strict_identity_fields(self) -> None:
        schema = json.loads(
            recipe_metadata_schema.SCHEMA_PATH.read_text(encoding="utf-8")
        )
        expected_fields = {
            "schema_version",
            "id",
            "name",
            "version",
            "description",
            "authors",
            "license",
            "tags",
            "links",
        }

        recipe_metadata_schema.recipe_metadata_validator()
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["author"]["additionalProperties"])
        self.assertFalse(schema["$defs"]["links"]["additionalProperties"])
        self.assertEqual(set(schema["properties"]), expected_fields)
        self.assertEqual(set(schema["required"]), expected_fields)

    def test_metadata_contract_cases_match_json_schema(self) -> None:
        contract_path = (
            recipe_conformance.REPO_ROOT
            / "config"
            / "schemas"
            / "recipe-metadata-v1.contract.yaml"
        )
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

        for case in contract["cases"]:
            with self.subTest(case=case["name"]):
                try:
                    metadata = recipe_metadata_schema.load_recipe_metadata_document(
                        case["document"]
                    )
                    recipe_metadata_schema.validate_recipe_metadata_schema(
                        metadata, contract_path
                    )
                    accepted = True
                except (TypeError, ValueError, yaml.YAMLError):
                    accepted = False
                self.assertEqual(accepted, case["valid"])

    def test_probe_schema_matches_parser_field_inventory(self) -> None:
        schema_path = (
            recipe_conformance.REPO_ROOT
            / "config"
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
        self.assertEqual(
            {recipe.identity.id for recipe in inventory},
            set(by_name),
        )
        self.assertTrue(
            all(
                recipe.identity.schema_version
                == recipe_metadata_schema.METADATA_SCHEMA_VERSION
                for recipe in inventory
            )
        )
        self.assertTrue(all(recipe.identity.version for recipe in inventory))
        self.assertEqual(
            by_name["multi-objective"].identity.name,
            "Multi-Objective Mixture-of-Models",
        )
        self.assertTrue(
            all(recipe.metadata.endswith("/metadata.yaml") for recipe in inventory)
        )
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

    def test_metadata_schema_rejects_unknown_fields(self) -> None:
        source = (
            recipe_conformance.DEFAULT_RECIPE_ROOT / "accuracy" / "metadata.yaml"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tempdir:
            metadata = Path(tempdir) / "metadata.yaml"
            metadata.write_text(source + "unknown: value\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "recipe metadata schema"):
                recipe_conformance.load_recipe_identity(metadata, "accuracy")

    def test_metadata_schema_requires_semantic_version(self) -> None:
        source = yaml.safe_load(
            (
                recipe_conformance.DEFAULT_RECIPE_ROOT / "accuracy" / "metadata.yaml"
            ).read_text(encoding="utf-8")
        )
        source["version"] = "latest"
        with tempfile.TemporaryDirectory() as tempdir:
            metadata = Path(tempdir) / "metadata.yaml"
            metadata.write_text(yaml.safe_dump(source), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "recipe metadata schema"):
                recipe_conformance.load_recipe_identity(metadata, "accuracy")

    def test_metadata_id_must_match_recipe_directory(self) -> None:
        source = (
            recipe_conformance.DEFAULT_RECIPE_ROOT / "accuracy" / "metadata.yaml"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tempdir:
            metadata = Path(tempdir) / "metadata.yaml"
            metadata.write_text(
                source.replace("id: accuracy", "id: different"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "id must match directory"):
                recipe_conformance.load_recipe_identity(metadata, "accuracy")

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

    def test_cpu_plan_keeps_complete_static_inventory_and_reports_hardware(
        self,
    ) -> None:
        inventory = recipe_conformance.discover_inventory(
            recipe_conformance.DEFAULT_RECIPE_ROOT
        )
        by_name = {recipe.name: recipe for recipe in inventory}
        self.assertEqual(by_name["vela-amd"].required_devices, ("migraphx:0", "rocm:0"))
        with tempfile.TemporaryDirectory() as directory:
            args = recipe_conformance.build_parser().parse_args(
                ["--output-dir", directory, "plan", "--shards", "3"]
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(recipe_conformance.command_plan(args), 0)
            matrix = json.loads(output.getvalue())
            planned = {
                name
                for shard in matrix["include"]
                for name in shard["recipes"].split(",")
            }
            self.assertEqual(planned, set(by_name) - {"vela-amd"})
            receipt = json.loads((Path(directory) / "cpu-eligibility.json").read_text())
            self.assertEqual(
                receipt["excluded"],
                [{"recipe": "vela-amd", "required_devices": ["migraphx:0", "rocm:0"]}],
            )

    def test_accelerator_requirements_follow_devices_not_recipe_names(self) -> None:
        deployments = {
            "local": {"provider": "candle", "device": "cpu"},
            "remote": {"provider": "http"},
            "gpu": {"provider": "candle", "device": "cuda:0"},
            "other": {"provider": "ort", "device": "migraphx:2"},
        }
        config = {"global": {"model_catalog": {"deployments": deployments}}}
        self.assertEqual(
            recipe_conformance.configured_accelerators(config), ("cuda:0", "migraphx:2")
        )
        del deployments["gpu"]
        del deployments["other"]
        self.assertEqual(recipe_conformance.configured_accelerators(config), ())

    def test_cpu_runner_rejects_hardware_before_any_stack_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "mutated"
            for name in ("vllm-sr", "docker"):
                command = root / name
                command.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 0\n")
                command.chmod(0o755)
            # Use this test process's Python, including its conformance dependencies.
            python = root / "python3"
            python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
            python.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "RECIPES": "accuracy,vela-amd",
                "ROUTER_IMAGE": "cpu-test",
            }
            result = subprocess.run(
                [
                    "bash",
                    str(
                        recipe_conformance.REPO_ROOT
                        / "e2e/testing/run_recipe_conformance.sh"
                    ),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires explicit devices", result.stderr)
            self.assertFalse(marker.exists(), result.stdout + result.stderr)
        args = recipe_conformance.build_parser().parse_args(
            ["check-cpu", "--recipes", "accuracy,knowledge"]
        )
        self.assertEqual(recipe_conformance.command_check_cpu(args), 0)

    def test_hardware_requirement_is_not_a_runtime_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inventory.json").write_text(
                json.dumps(
                    {
                        "recipes": [
                            {"name": "gpu", "required_devices": ["rocm:0"]},
                            {"name": "cpu", "required_devices": []},
                        ]
                    }
                )
            )
            report = recipe_conformance.build_consolidated_report(root)
            self.assertEqual(report["summary"]["requires_hardware_recipes"], 1)
            self.assertEqual(report["summary"]["passed_recipes"], 0)
            self.assertEqual(report["summary"]["missing_recipes"], 1)
            self.assertFalse(report["summary"]["passed"])
            gpu = report["results"][0]
            self.assertEqual(gpu["status"], "requires_hardware")
            self.assertFalse(gpu["passed"])
            self.assertEqual(gpu["total"], 0)

    def test_cpu_success_does_not_complete_catalog_with_unrun_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inventory.json").write_text(
                json.dumps(
                    {
                        "recipes": [
                            {"name": "gpu", "required_devices": ["rocm:0"]},
                            {"name": "cpu", "required_devices": []},
                        ]
                    }
                )
            )
            (root / "cpu").mkdir()
            (root / "cpu/eval-report.json").write_text(
                json.dumps(
                    {
                        "inventory": {"name": "cpu"},
                        "evaluation": {"passed": True, "matched": 2, "total": 2},
                    }
                )
            )
            summary = recipe_conformance.build_consolidated_report(root)["summary"]
            self.assertEqual(summary["passed_recipes"], 1)
            self.assertEqual(summary["requires_hardware_recipes"], 1)
            self.assertFalse(summary["complete"])
            self.assertFalse(summary["passed"])
            self.assertTrue(summary["cpu_compatible_complete"])
            self.assertTrue(summary["cpu_compatible_passed"])

    def test_incomplete_recipe_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recipe = Path(tempdir) / "incomplete"
            recipe.mkdir()
            (recipe / "config.yaml").write_text("version: v0.3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "five-file contract"):
                recipe_conformance.validate_recipe_directory(recipe)

    def test_recipe_readme_must_be_a_structured_model_card(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recipe = Path(tempdir) / "incomplete-card"
            recipe.mkdir()
            (recipe / "README.md").write_text(
                "# Routing notes\n\n## Overview\n\nA route.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "title must end with 'Model Card'"):
                recipe_conformance.validate_recipe_model_card(recipe)

    def test_recipe_model_card_sections_must_be_complete_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recipe = Path(tempdir) / "unordered-card"
            recipe.mkdir()
            sections = list(recipe_conformance.REQUIRED_MODEL_CARD_HEADINGS)
            sections[0], sections[1] = sections[1], sections[0]
            (recipe / "README.md").write_text(
                "# Example Model Card\n\n" + "\n\n".join(sections) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "documented order"):
                recipe_conformance.validate_recipe_model_card(recipe)

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

    def test_portable_recipe_defaults_do_not_create_runtime_entrypoints(self) -> None:
        config = {
            "global": {"router": {"learning": {"enabled": True}}},
            "recipes": [{"name": "balanced", "routing": {}}],
        }
        probes = [
            recipe_conformance.Probe(
                decision_id="route",
                variant_id="portable",
                probe_id="route:portable",
                expected_decision="route",
                expected_recipe="balanced",
            )
        ]

        self.assertEqual(recipe_conformance.config_entrypoints(config), {})
        self.assertIs(
            recipe_conformance.bind_default_entrypoints(config, probes), probes
        )
        self.assertIsNone(probes[0].model)

    def test_composed_recipe_keeps_runtime_entrypoint_coverage(self) -> None:
        bundle = {
            "global": {"router": {"learning": {"enabled": True}}},
            "recipes": [{"name": "balanced", "routing": {}}],
        }
        for field, value in (
            ("providers", {"models": [{"name": "backend"}]}),
            ("routing", {"decisions": [{"name": "fallback"}]}),
            ("entrypoints", [{"recipe": "balanced", "model_names": ["route"]}]),
        ):
            with self.subTest(field=field):
                config = {**bundle, field: value}
                entrypoints = recipe_conformance.config_entrypoints(config)
                self.assertEqual(
                    {
                        name: entrypoints[name]
                        for name in ("vllm-sr/auto", "auto", "MoM")
                    },
                    dict.fromkeys(("vllm-sr/auto", "auto", "MoM"), "default"),
                )
                if field == "entrypoints":
                    self.assertEqual(entrypoints["route"], "balanced")

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
