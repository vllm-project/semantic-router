import importlib
import json
import sys
import tempfile
import unittest
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
    def test_latest_built_in_recipe_bundle_uses_the_same_five_file_contract(
        self,
    ) -> None:
        root = (
            recipe_conformance.REPO_ROOT / "config" / "recipes" / "built-in" / "latest"
        )

        inventory = recipe_conformance.discover_inventory(root)

        self.assertEqual([recipe.name for recipe in inventory], ["mom-v1"])
        mom = inventory[0]
        self.assertEqual(mom.entrypoints, ())
        self.assertEqual(mom.auto_entrypoints, ())
        self.assertEqual(len(mom.decisions), 26)
        self.assertEqual(mom.variants, 226)
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

    def test_incomplete_recipe_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            recipe = Path(tempdir) / "incomplete"
            recipe.mkdir()
            (recipe / "config.yaml").write_text("version: v0.4\n", encoding="utf-8")

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
            "providers": {"models": [{"name": "frontier"}]},
            "routing": {"modelCards": [{"name": "frontier"}]},
            "recipes": [
                {
                    "name": "default",
                    "routing": {"decisions": [{"name": "route"}]},
                }
            ],
            "entrypoints": [
                {
                    "model_names": ["vllm-sr/auto", "auto", "custom-auto"],
                    "recipe": "default",
                    "assignments": {"route": {"models": [{"model": "frontier"}]}},
                }
            ],
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

        bound = recipe_conformance.bind_probe_entrypoints(config, probes)

        self.assertEqual(
            [probe.model for probe in bound],
            ["vllm-sr/auto", "auto", "custom-auto", "vllm-sr/auto"],
        )

    def test_live_entrypoints_bind_each_recipe_without_mutating_assets(self) -> None:
        config = {
            "recipes": [
                {"name": "balance", "routing": {"decisions": []}},
                {"name": "vault", "routing": {"decisions": []}},
            ]
        }
        probes = [
            recipe_conformance.Probe(
                decision_id="route",
                variant_id=recipe,
                probe_id=f"route:{recipe}",
                expected_decision="route",
                expected_recipe=recipe,
            )
            for recipe in ("balance", "vault")
        ]

        bound = recipe_conformance.bind_probe_entrypoints(
            config,
            probes,
            {"balance": ("vllm-sr/blend",), "vault": ("vllm-sr/vault",)},
        )

        self.assertEqual(
            [probe.model for probe in bound],
            ["vllm-sr/blend", "vllm-sr/vault"],
        )

    def test_live_entrypoint_arguments_are_strict_and_repeatable(self) -> None:
        self.assertEqual(
            recipe_conformance.parse_live_entrypoints(
                ["balance=vllm-sr/blend", "balance=blend"]
            ),
            {"balance": ("vllm-sr/blend", "blend")},
        )
        with self.assertRaisesRegex(ValueError, "RECIPE=MODEL"):
            recipe_conformance.parse_live_entrypoints(["balance"])

    def test_expected_models_resolve_from_current_entrypoint_assignments(self) -> None:
        config = {
            "providers": {"models": [{"name": "frontier"}]},
            "routing": {
                "modelCards": [{"name": "frontier", "aliases": ["frontier/latest"]}]
            },
            "recipes": [
                {
                    "name": "balanced",
                    "routing": {
                        "decisions": [
                            {"name": "complex"},
                        ]
                    },
                }
            ],
            "entrypoints": [
                {
                    "model_names": ["vllm-sr/mixture", "mixture"],
                    "recipe": "balanced",
                    "assignments": {"complex": {"models": [{"model": "frontier"}]}},
                }
            ],
        }

        assignments = recipe_conformance.config_assigned_models(config)

        self.assertEqual(
            assignments[("vllm-sr/mixture", "balanced", "complex")],
            frozenset({"frontier", "frontier/latest"}),
        )
        self.assertEqual(
            assignments[("mixture", "balanced", "complex")],
            frozenset({"frontier", "frontier/latest"}),
        )

    def test_dynamic_entrypoint_rules_are_not_a_config_authoring_shape(self) -> None:
        config = {
            "providers": {"models": [{"name": "frontier"}, {"name": "fast"}]},
            "routing": {
                "modelCards": [
                    {"name": "frontier", "aliases": ["frontier/latest"]},
                    {"name": "fast"},
                ]
            },
            "recipes": [
                {
                    "name": "balanced",
                    "routing": {"decisions": [{"name": "answer"}]},
                }
            ],
            "entrypoints": [
                {
                    "model_names": ["vllm-sr/conditional", "conditional"],
                    "rules": [
                        {
                            "name": "premium",
                            "recipe": "balanced",
                            "assignments": {
                                "answer": {"models": [{"model": "frontier"}]}
                            },
                        },
                        {
                            "name": "standard",
                            "recipe": "balanced",
                            "assignments": {"answer": {"models": [{"model": "fast"}]}},
                        },
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "unsupported generated fields: rules"):
            recipe_conformance.config_entrypoints(config)

    def test_generated_identity_fields_are_not_supported(self) -> None:
        def human_config() -> dict:
            return {
                "providers": {"models": [{"name": "frontier"}]},
                "routing": {"modelCards": [{"name": "frontier"}]},
                "recipes": [
                    {
                        "name": "default",
                        "routing": {"decisions": [{"name": "route"}]},
                    }
                ],
                "entrypoints": [
                    {
                        "model_names": ["vllm-sr/auto"],
                        "recipe": "default",
                        "assignments": {"route": {"models": [{"model": "frontier"}]}},
                    }
                ],
            }

        recipe_identity = human_config()
        recipe_identity["recipes"][0]["id"] = "rcp_default"
        decision_identity = human_config()
        decision_identity["recipes"][0]["routing"]["decisions"][0][
            "decision_id"
        ] = "dec_route"
        entrypoint_identity = human_config()
        entrypoint_identity["entrypoints"][0]["id"] = "ep_auto"
        entrypoint_action = human_config()
        entrypoint_action["entrypoints"] = [
            {
                "model_names": ["vllm-sr/auto"],
                "rules": [
                    {
                        "name": "default",
                        "action": {"recipe_id": "rcp_default"},
                    }
                ],
            }
        ]
        model_identity = human_config()
        model_identity["providers"]["models"][0]["id"] = "mdl_frontier"
        assignment_identity = human_config()
        assignment_identity["entrypoints"][0]["assignments"]["route"]["models"] = [
            {"model_id": "mdl_frontier"}
        ]

        cases = (
            ("recipe id", recipe_identity, recipe_conformance.recipe_profiles),
            ("decision id", decision_identity, recipe_conformance.recipe_profiles),
            (
                "entrypoint id",
                entrypoint_identity,
                recipe_conformance.config_entrypoints,
            ),
            (
                "entrypoint runtime action",
                entrypoint_action,
                recipe_conformance.config_entrypoints,
            ),
            (
                "model id",
                model_identity,
                recipe_conformance.config_assigned_models,
            ),
            (
                "assignment model id",
                assignment_identity,
                recipe_conformance.config_assigned_models,
            ),
        )

        for name, config, parser in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "unsupported generated fields"
            ):
                parser(config)

    def test_config_entrypoint_requires_one_recipe(self) -> None:
        config = {
            "recipes": [
                {
                    "name": "fast",
                    "routing": {"decisions": [{"name": "answer"}]},
                },
                {
                    "name": "deep",
                    "routing": {"decisions": [{"name": "answer"}]},
                },
            ],
            "entrypoints": [
                {
                    "model_names": ["vllm-sr/missing"],
                    "assignments": {},
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "has no recipe"):
            recipe_conformance.config_entrypoints(config)

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
