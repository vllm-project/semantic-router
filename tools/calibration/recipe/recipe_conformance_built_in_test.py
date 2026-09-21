"""The portable built-in bundle must execute through real CLI composition."""

import argparse
import copy
import importlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
conformance = importlib.import_module("recipe_conformance")
runtime = importlib.import_module("recipe_conformance_runtime")
sources = importlib.import_module("recipe_conformance_sources")
http = importlib.import_module("router_calibration_http")
EXPECTED_BUILTIN_PROBES = 315


class BuiltInRecipeConformanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe_root = conformance.DEFAULT_RECIPE_ROOT / "built-in/latest"
        cls.recipe = cls.recipe_root / "mom-v1"
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.config_path = Path(cls.temporary.name) / "runtime/config.yaml"
        cls.original = {
            path.name: path.read_bytes()
            for path in cls.recipe.iterdir()
            if path.is_file()
        }
        cls.config = runtime.prepare_builtin_runtime(
            cls.recipe, cls.config_path, conformance.REPO_ROOT
        )
        cls.authored = conformance.load_yaml_mapping(cls.recipe / "config.yaml")
        cls.manifest, cls.probes = conformance.load_probe_manifest(
            cls.recipe / "probes.yaml"
        )

    def test_real_cli_composition_covers_every_entrypoint_and_authored_probe(self):
        bound = runtime.bind_runtime_entrypoints(self.config, self.probes)
        runtime.verify_composed_policy(self.authored, self.config)
        conformance.validate_probe_references(
            self.recipe / "probes.yaml", self.config, self.manifest, bound
        )
        self.assertEqual(len(bound), 315)
        self.assertEqual(
            {probe.model for probe in bound},
            {
                "vllm-sr/mom-v1-blend",
                "vllm-sr/mom-v1-flash",
                "vllm-sr/mom-v1-lite",
                "vllm-sr/mom-v1-ultra",
                "vllm-sr/mom-v1-vault",
            },
        )
        self.assertEqual(
            Counter(probe.expected_recipe for probe in bound),
            Counter(probe.expected_recipe for probe in self.probes),
        )
        for original, actual in zip(self.probes, bound, strict=True):
            self.assertEqual(actual, conformance.replace(original, model=actual.model))
        self.assertEqual(
            self.original,
            {
                path.name: path.read_bytes()
                for path in self.recipe.iterdir()
                if path.is_file()
            },
        )
        decisions = [
            decision
            for recipe in self.config["recipes"]
            for decision in recipe["routing"]["decisions"]
        ]
        for decision in decisions:
            minimum = decision.get("algorithm", {}).get("minimum_candidates", 0)
            self.assertGreaterEqual(len(decision.get("modelRefs", [])), minimum)

    def test_policy_mutation_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["recipes"][0]["routing"]["decisions"][0]["priority"] += 1
        with self.assertRaisesRegex(ValueError, "changed authored routing policy"):
            runtime.verify_composed_policy(self.authored, changed)

    def test_authored_global_learning_loss_or_change_is_rejected(self):
        for path, replacement in (
            (("global",), None),
            (("global", "router", "learning"), None),
            (("global", "router", "learning", "enabled"), False),
            (("global", "router", "learning", "adaptation", "enabled"), True),
            (("global", "router", "learning", "protection", "scope"), "session"),
        ):
            with self.subTest(path=path):
                changed = copy.deepcopy(self.config)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                if replacement is None:
                    del target[path[-1]]
                else:
                    target[path[-1]] = replacement
                with self.assertRaisesRegex(
                    ValueError, "changed authored global policy"
                ):
                    runtime.verify_composed_policy(self.authored, changed)

    def test_global_overrides_are_limited_to_fixture_deployment_fields(self):
        authored = copy.deepcopy(self.authored)
        authored["global"]["router"]["auto_model_names"] = ["MoM"]
        authored["global"].setdefault("services", {})["management_api"] = {
            "bind_address": "127.0.0.1",
            "auth": {"mode": "disabled"},
            "port": 9080,
        }
        composed = copy.deepcopy(self.config)
        composed["global"]["services"]["management_api"]["port"] = 9080
        runtime.verify_composed_policy(authored, composed)
        self.assertEqual(authored["global"]["router"]["auto_model_names"], ["MoM"])

        # The fixture owns bind/auth, but cannot discard another authored field
        # merely because it shares the management_api parent mapping.
        del composed["global"]["services"]["management_api"]["port"]
        with self.assertRaisesRegex(ValueError, "changed authored global policy"):
            runtime.verify_composed_policy(authored, composed)

    def test_composed_management_listener_is_reachable_in_split_runtime(self):
        parser = importlib.import_module("cli.parser")
        listener = importlib.import_module("cli.container_management_listener")
        stack = importlib.import_module("cli.runtime_stack").resolve_runtime_stack(
            stack_name="recipe-conformance-contract", port_offset=1000
        )
        config = parser.parse_user_config(str(self.config_path), log_summary=False)
        self.assertEqual(
            listener.resolve_managed_management_listener(config, stack),
            {"bind_address": "0.0.0.0", "port": 8080, "host_port": 9080},
        )

    def test_declared_auth_reaches_readiness_and_evaluation_transport(self):
        self.assertEqual(
            runtime.management_auth_bindings(self.config),
            [("VLLM_SR_CONFORMANCE_TOKEN", True)],
        )
        with (
            patch.dict("os.environ", {"VSR_MGMT_TOKEN": "test-token"}),
            patch("urllib.request.urlopen") as urlopen,
        ):
            response = urlopen.return_value.__enter__.return_value
            response.getcode.return_value = 200
            response.read.return_value = b"{}"
            http.http_json("POST", "http://localhost/api/v1/routing/preview", {})
            self.assertEqual(
                urlopen.call_args.args[0].get_header("Authorization"),
                "Bearer test-token",
            )
        with self.assertRaisesRegex(ValueError, "ready.read"):
            runtime.management_auth_bindings(
                {
                    "global": {
                        "services": {
                            "management_api": {
                                "auth": {
                                    "mode": "bearer",
                                    "roles": {"reader": ["config.read"]},
                                    "tokens": [{"env": "TOKEN", "role": "reader"}],
                                }
                            }
                        }
                    }
                }
            )

    def test_cpu_plan_covers_both_sources_without_hardware_or_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                recipes_root=conformance.DEFAULT_RECIPE_ROOT,
                output_dir=Path(directory),
                shards=3,
                github_output=None,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(conformance.command_plan_all(args), 0)
            matrix = json.loads(output.getvalue())["include"]
            self.assertEqual(len({row["shard"] for row in matrix}), len(matrix))
            built_in = [row for row in matrix if row["source"] == "built-in-latest"]
            self.assertEqual(
                built_in,
                [
                    {
                        "source": "built-in-latest",
                        "shard": "built-in-latest-0",
                        "recipes": "mom-v1",
                        "variants": 315,
                        "recipes_root": "config/recipes/built-in/latest",
                        "report_dir": "built-in/latest",
                    }
                ],
            )
            self.assertTrue(
                all("vela-amd" not in row["recipes"].split(",") for row in matrix)
            )
            receipt = json.loads((Path(directory) / "cpu-eligibility.json").read_text())
            self.assertEqual(receipt["excluded"][0]["recipe"], "vela-amd")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "built-in/latest").mkdir(parents=True)
            (root / "built-in/v0.4").mkdir()
            self.assertEqual(
                [source.name for source in sources.discover_recipe_sources(root)],
                ["standalone", "built-in-latest"],
            )

    def test_eval_rejects_zero_or_partial_probe_execution(self):
        for count in (0, 314, 315):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                probes = runtime.bind_runtime_entrypoints(self.config, self.probes)
                evaluation = {
                    "passed": True,
                    "matched": count,
                    "total": count,
                    "results": [
                        {"id": probe.probe_id, "model": probe.model}
                        for probe in probes[:count]
                    ],
                }
                args = argparse.Namespace(
                    recipes_root=self.recipe_root,
                    recipe="mom-v1",
                    output_dir=Path(directory),
                    runtime_config=self.config_path,
                    router_url="http://unused",
                    scope="deployment",
                )
                with (
                    patch.object(
                        conformance, "evaluate_probes", return_value=evaluation
                    ),
                    patch.object(
                        conformance,
                        "evaluate_live_tag_policy",
                        return_value={"passed": True},
                    ),
                    patch.object(
                        conformance, "render_markdown_summary", return_value="summary"
                    ),
                    patch.object(
                        conformance,
                        "render_tag_acceptance_markdown",
                        return_value="tags",
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    result = conformance.command_eval(args)
                self.assertEqual(result, 0 if count == EXPECTED_BUILTIN_PROBES else 1)
                receipt = json.loads(
                    (Path(directory) / "mom-v1/eval-report.json").read_text()
                )
                self.assertEqual(
                    receipt["evaluation"]["execution"]["complete"],
                    count == EXPECTED_BUILTIN_PROBES,
                )

    def test_consolidation_fails_missing_partial_or_empty_results(self):
        for count in (None, 0, 314, 315):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "inventory.json").write_text(
                    json.dumps({"recipes": [{"name": "mom-v1", "variants": 315}]})
                )
                if count is not None:
                    (root / "mom-v1").mkdir()
                    (root / "mom-v1/eval-report.json").write_text(
                        json.dumps(
                            {
                                "inventory": {"name": "mom-v1"},
                                "evaluation": {
                                    "passed": True,
                                    "matched": count,
                                    "total": count,
                                    "execution": {"complete": True},
                                },
                            }
                        )
                    )
                summary = conformance.build_consolidated_report(root)["summary"]
                self.assertEqual(
                    summary["cpu_compatible_passed"], count == EXPECTED_BUILTIN_PROBES
                )


if __name__ == "__main__":
    unittest.main()
