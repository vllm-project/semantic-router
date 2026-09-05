from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_model_catalog.py"
SPEC = importlib.util.spec_from_file_location("generate_model_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)


class ModelCatalogValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocols = [
            {
                "id": "openai/chat-completions@1",
                "operations": [
                    {"id": "create", "method": "POST", "path": "/v1/chat/completions"},
                    {"id": "list_models", "method": "GET", "path": "/v1/models"},
                ],
            }
        ]

    def test_virtual_model_pool_can_reference_operator_defined_models(self) -> None:
        catalog._validate_virtual_model_role(
            {
                "name": "private",
                "required": True,
                "minimum_candidates": 1,
                "traits": ["local_only"],
                "recommended_pool": ["operator/private-model"],
            },
            "models[0].roles[0]",
        )

    def test_creator_inventory_policy_ignores_virtual_recipe_publishers(self) -> None:
        manifest = {
            "inventory": {
                "physical": {
                    "strategy": "curated_creator_companies",
                    "default_min_representatives": 2,
                    "creators": [
                        {
                            "publisher": "Acme Models",
                            "representative_models": ["acme/one", "acme/two"],
                        },
                        {
                            "publisher": "New Lab",
                            "min_representatives": 1,
                            "representative_models": ["new/one"],
                        },
                    ],
                }
            }
        }
        models = [
            {
                "id": "acme/one",
                "kind": "physical",
                "publisher": "Acme Models",
                "lifecycle": "active",
            },
            {
                "id": "acme/two",
                "kind": "physical",
                "publisher": "Acme Models",
                "lifecycle": "experimental",
            },
            {
                "id": "new/one",
                "kind": "physical",
                "publisher": "New Lab",
                "lifecycle": "active",
            },
            {
                "id": "vllm-sr/virtual",
                "kind": "virtual",
                "publisher": "vllm-sr.ai",
                "lifecycle": "active",
            },
        ]

        catalog._validate_inventory_policy(manifest, models)

    def test_creator_inventory_policy_rejects_long_tail_and_shallow_entries(
        self,
    ) -> None:
        manifest = {
            "inventory": {
                "physical": {
                    "strategy": "curated_creator_companies",
                    "default_min_representatives": 2,
                    "creators": [
                        {
                            "publisher": "Acme Models",
                            "representative_models": ["acme/one", "acme/two"],
                        }
                    ],
                }
            }
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "unlisted physical creators: Obscure Lab"
        ):
            catalog._validate_inventory_policy(
                manifest,
                [
                    {
                        "id": "acme/one",
                        "kind": "physical",
                        "publisher": "Acme Models",
                        "lifecycle": "active",
                    },
                    {
                        "id": "obscure/one",
                        "kind": "physical",
                        "publisher": "Obscure Lab",
                        "lifecycle": "active",
                    },
                ],
            )
        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            r"representative model 'acme/two' is missing or removed",
        ):
            catalog._validate_inventory_policy(
                manifest,
                [
                    {
                        "id": "acme/one",
                        "kind": "physical",
                        "publisher": "Acme Models",
                        "lifecycle": "active",
                    }
                ],
            )

        shallow_manifest = {
            "inventory": {
                "physical": {
                    "strategy": "curated_creator_companies",
                    "default_min_representatives": 2,
                    "creators": [
                        {
                            "publisher": "Acme Models",
                            "representative_models": ["acme/one"],
                        }
                    ],
                }
            }
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            r"representative_models has 1 models \(minimum 2\)",
        ):
            catalog._validate_inventory_policy(
                shallow_manifest,
                [
                    {
                        "id": "acme/one",
                        "kind": "physical",
                        "publisher": "Acme Models",
                        "lifecycle": "active",
                    }
                ],
            )

    def test_creator_inventory_policy_rejects_wrong_or_stale_representatives(
        self,
    ) -> None:
        manifest = {
            "inventory": {
                "physical": {
                    "strategy": "curated_creator_companies",
                    "default_min_representatives": 1,
                    "creators": [
                        {
                            "publisher": "Acme Models",
                            "representative_models": ["new/one"],
                        },
                        {
                            "publisher": "New Lab",
                            "representative_models": ["new/one"],
                        },
                    ],
                }
            }
        }
        models = [
            {
                "id": "acme/old",
                "kind": "physical",
                "publisher": "Acme Models",
                "lifecycle": "active",
            },
            {
                "id": "new/one",
                "kind": "physical",
                "publisher": "New Lab",
                "lifecycle": "active",
            },
        ]
        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            r"creator 'Acme Models' cannot claim representative model 'new/one'",
        ):
            catalog._validate_inventory_policy(manifest, models)

        manifest["inventory"]["physical"]["creators"][0]["representative_models"] = [
            "acme/old"
        ]
        models[0]["lifecycle"] = "deprecated"
        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            r"representative model 'acme/old' must be current",
        ):
            catalog._validate_inventory_policy(manifest, models)

    def test_model_layout_stays_inside_catalog_and_uses_one_file_per_creator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            source_root = repo_root / "config" / "catalog"
            model_root = source_root / "resources" / "models"
            single_root = model_root / "single"
            virtual_root = model_root / "virtual"
            single_root.mkdir(parents=True)
            virtual_root.mkdir(parents=True)
            (single_root / "acme-a.yaml").write_text(
                "kind: physical\npublisher: Acme Models\n", encoding="utf-8"
            )
            (single_root / "acme-b.yaml").write_text(
                "kind: physical\npublisher: Acme Models\n", encoding="utf-8"
            )
            (virtual_root / "router.yaml").write_text(
                "kind: virtual\n", encoding="utf-8"
            )
            manifest = {"resources": {"models": "resources/models"}}

            with self.assertRaisesRegex(
                catalog.CatalogBuildError,
                r"physical model creator 'Acme Models' must use one file",
            ):
                catalog._validate_model_resource_layout_impl(
                    manifest, source_root, repo_root
                )

            escaped = {"resources": {"models": "../../outside-models"}}
            with self.assertRaisesRegex(
                catalog.CatalogBuildError, "resources.models escapes config/catalog"
            ):
                catalog._validate_model_resource_layout_impl(
                    escaped, source_root, repo_root
                )

    def test_evaluation_layout_matches_model_kind_and_creator_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            source_root = repo_root / "config" / "catalog"
            model_root = source_root / "resources" / "models"
            evaluation_root = source_root / "resources" / "evaluations"
            for root in (
                model_root / "single",
                model_root / "virtual",
                evaluation_root / "single",
                evaluation_root / "virtual",
            ):
                root.mkdir(parents=True)
            (model_root / "single" / "acme.yaml").write_text(
                "id: acme/one\nkind: physical\npublisher: Acme Models\n",
                encoding="utf-8",
            )
            (model_root / "virtual" / "vllm-sr.yaml").write_text(
                "id: vllm-sr/router\nkind: virtual\npublisher: vllm-sr.ai\n",
                encoding="utf-8",
            )
            misplaced = evaluation_root / "single" / "other.yaml"
            misplaced.write_text("model: acme/one\n", encoding="utf-8")
            manifest = {
                "resources": {
                    "models": "resources/models",
                    "evaluations": "resources/evaluations",
                }
            }

            with self.assertRaisesRegex(
                catalog.CatalogBuildError,
                r"evaluation for 'acme/one' must live in evaluations/single/acme.yaml",
            ):
                catalog._validate_evaluation_resource_layout_impl(
                    manifest, source_root, repo_root
                )

            misplaced.unlink()
            (evaluation_root / "single" / "vllm-sr.yaml").write_text(
                "model: vllm-sr/router\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                catalog.CatalogBuildError,
                r"evaluation for 'vllm-sr/router' must live in "
                r"evaluations/virtual/vllm-sr.yaml",
            ):
                catalog._validate_evaluation_resource_layout_impl(
                    manifest, source_root, repo_root
                )

    def test_schema_failure_reports_the_resource_path(self) -> None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["id"],
            "properties": {"id": {"type": "string", "minLength": 1}},
        }
        with self.assertRaisesRegex(catalog.CatalogBuildError, r"provider\.id"):
            catalog._validate_schema({"id": ""}, schema, "provider")

    def test_provider_reasoning_transport_is_a_known_semantic_adapter(self) -> None:
        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#create"],
            "reasoning_transport": "hostname_switch",
            "auth": {
                "strategy": "bearer",
                "header": "Authorization",
                "prefix": "Bearer",
            },
            "presentation": {"logo": "monogram", "monogram": "E", "monochrome": True},
            "conformance": {"status": "unverified"},
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "reasoning_transport is unsupported"
        ):
            catalog._validate_providers([provider], self.protocols)

        provider["reasoning_transport"] = "reasoning_object"
        catalog._validate_providers([provider], self.protocols)

    def test_featured_is_repository_owned_provider_presentation(self) -> None:
        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#create"],
            "auth": {
                "strategy": "bearer",
                "header": "Authorization",
                "prefix": "Bearer",
            },
            "presentation": {
                "logo": "monogram",
                "monogram": "E",
                "monochrome": True,
                "featured": True,
            },
            "conformance": {"status": "unverified"},
        }
        catalog._validate_providers([provider], self.protocols)

        provider["presentation"]["featured"] = "yes"
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "featured must be a boolean"
        ):
            catalog._validate_providers([provider], self.protocols)

        provider["presentation"]["featured"] = True
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "unknown fields: featured"
        ):
            catalog._validate_provider_presentation(provider, "models[0]")

    def test_provider_operations_are_explicit_protocol_subsets(self) -> None:
        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#delete_model"],
            "auth": {
                "strategy": "bearer",
                "header": "Authorization",
                "prefix": "Bearer",
            },
            "presentation": {"logo": "monogram", "monogram": "E", "monochrome": True},
            "conformance": {"status": "unverified"},
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "unknown or duplicate operation"
        ):
            catalog._validate_providers([provider], self.protocols)

    def test_provider_catalog_model_protocol_must_be_supported(self) -> None:
        providers = {
            "example": {
                "protocols": ["openai/chat-completions@1"],
                "supported_operations": ["openai/chat-completions@1#create"],
                "models": [
                    {
                        "catalog": "example/model",
                        "relationship": "first_party",
                        "id": "native-model",
                        "protocols": ["openai/chat-completions@1"],
                    }
                ],
            }
        }
        models = {"example/model": {"kind": "physical", "lifecycle": "active"}}
        catalog._validate_provider_bindings(
            providers, models, {"openai/chat-completions@1"}
        )

    def test_physical_model_requires_a_provider_binding(self) -> None:
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "require at least one provider binding"
        ):
            catalog._validate_provider_bindings(
                {},
                {
                    "example/model": {
                        "kind": "physical",
                        "lifecycle": "active",
                    }
                },
                {"openai/chat-completions@1"},
            )

    def test_provider_model_id_kind_policy_is_closed(self) -> None:
        binding = {
            "catalog": "example/model",
            "relationship": "first_party",
            "id": "native-model",
            "protocols": ["openai/chat-completions@1"],
            "restrictions": {
                "provider_model_id_kind": "deployment_name",
                "catalog_model_name": "Example Model",
            },
        }
        providers = {
            "example": {
                "protocols": ["openai/chat-completions@1"],
                "supported_operations": ["openai/chat-completions@1#create"],
                "models": [binding],
            }
        }
        models = {"example/model": {"kind": "physical", "lifecycle": "active"}}
        catalog._validate_provider_bindings(
            providers, models, {"openai/chat-completions@1"}
        )

        binding["restrictions"]["provider_model_id_kind"] = "opaque_alias"
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "provider_model_id_kind is unsupported"
        ):
            catalog._validate_provider_bindings(
                providers, models, {"openai/chat-completions@1"}
            )

        binding["restrictions"] = {"provider_model_id_kind": "deployment_name"}
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "catalog_model_name must be a non-empty string"
        ):
            catalog._validate_provider_bindings(
                providers, models, {"openai/chat-completions@1"}
            )

    def test_provider_binding_relationship_is_required_and_closed(self) -> None:
        binding = {
            "catalog": "example/model",
            "relationship": "first_party",
            "id": "native-model",
            "protocols": ["openai/chat-completions@1"],
        }
        providers = {
            "example": {
                "protocols": ["openai/chat-completions@1"],
                "supported_operations": ["openai/chat-completions@1#create"],
                "models": [binding],
            }
        }
        models = {"example/model": {"kind": "physical", "lifecycle": "active"}}

        catalog._validate_provider_bindings(
            providers, models, {"openai/chat-completions@1"}
        )
        del binding["relationship"]
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "relationship must be a non-empty string"
        ):
            catalog._validate_provider_bindings(
                providers, models, {"openai/chat-completions@1"}
            )
        binding["relationship"] = "brokered"
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "relationship is unsupported"
        ):
            catalog._validate_provider_bindings(
                providers, models, {"openai/chat-completions@1"}
            )

    def test_available_evaluation_metric_is_unambiguous(self) -> None:
        records = [
            {
                "id": f"example/run-{index}@1.0.0",
                "model": "example/model",
                "benchmark": "example/bench@1.0.0",
                "benchmark_profile": "standard",
                "reasoning_effort": "default",
                "subject": {},
                "metrics": {"score": value},
                "status": "available",
                "observed_at": "2026-09-06",
                "evidence": {
                    "provenance": "operator",
                    "verification": "claimed",
                    "redistributable": True,
                },
            }
            for index, value in enumerate((0.7, 0.8), start=1)
        ]
        metrics = {
            "example/bench@1.0.0#score": {
                "range": [0, 1],
                "direction": "higher_is_better",
                "profiles": {"standard"},
            }
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "one available value is allowed"
        ):
            catalog._validate_evaluations(
                records,
                {"example/model": {"id": "example/model"}},
                {},
                metrics,
            )

    def test_benchmark_scoped_subject_keys_are_rejected_elsewhere(self) -> None:
        metrics = {
            "example/bench@1.0.0#score": {
                "range": [0, 1],
                "direction": "higher_is_better",
                "profiles": {"standard"},
            }
        }
        for subject_key in (
            "hle_judge",
            "hle_mode",
            "terminal_harness",
            "swe_harness",
        ):
            with self.subTest(subject_key=subject_key), self.assertRaisesRegex(
                catalog.CatalogBuildError,
                rf"subject\.{subject_key} is only valid for benchmark families",
            ):
                catalog._validate_evaluations(
                    [
                        {
                            "id": f"example/{subject_key}@1.0.0",
                            "model": "example/model",
                            "benchmark": "example/bench@1.0.0",
                            "benchmark_profile": "standard",
                            "reasoning_effort": "default",
                            "subject": {subject_key: "configured"},
                            "metrics": {"score": 0.8},
                            "status": "available",
                            "observed_at": "2026-09-06",
                            "evidence": {
                                "provenance": "operator",
                                "verification": "claimed",
                                "redistributable": True,
                            },
                        }
                    ],
                    {"example/model": {"id": "example/model"}},
                    {},
                    metrics,
                )

    def test_benchmark_scoped_subject_keys_accept_their_benchmark_family(
        self,
    ) -> None:
        allowed = {
            "hle_judge": "cais/humanitys-last-exam@1.0.0",
            "hle_mode": "cais/humanitys-last-exam-verified@1.0.0",
            "terminal_harness": "harbor/terminal-bench@2.1.0",
            "swe_harness": "swe-bench/verified@1.0.0",
        }
        for subject_key, benchmark in allowed.items():
            with self.subTest(subject_key=subject_key, benchmark=benchmark):
                catalog._validate_evaluations(
                    [
                        {
                            "id": f"example/{subject_key}@1.0.0",
                            "model": "example/model",
                            "benchmark": benchmark,
                            "benchmark_profile": "standard",
                            "reasoning_effort": "default",
                            "subject": {subject_key: "configured"},
                            "metrics": {"score": 0.8},
                            "status": "available",
                            "observed_at": "2026-09-06",
                            "evidence": {
                                "provenance": "operator",
                                "verification": "claimed",
                                "redistributable": True,
                            },
                        }
                    ],
                    {"example/model": {"id": "example/model"}},
                    {},
                    {
                        f"{benchmark}#score": {
                            "range": [0, 1],
                            "direction": "higher_is_better",
                            "profiles": {"standard"},
                        }
                    },
                )

    def test_available_evaluation_requires_calendar_anchor(self) -> None:
        record = {
            "id": "example/run@1.0.0",
            "model": "example/model",
            "benchmark": "example/bench@1.0.0",
            "benchmark_profile": "standard",
            "reasoning_effort": "default",
            "subject": {},
            "metrics": {"score": 0.8},
            "status": "available",
            "evidence": {
                "provenance": "operator",
                "verification": "claimed",
                "redistributable": True,
            },
        }
        models = {"example/model": {"id": "example/model"}}
        metrics = {
            "example/bench@1.0.0#score": {
                "range": [0, 1],
                "direction": "higher_is_better",
                "profiles": {"standard"},
            }
        }
        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            "must define measured_at or observed_at",
        ):
            catalog._validate_evaluations([record], models, {}, metrics)

        record["observed_at"] = "2026-09-06"
        catalog._validate_evaluations([record], models, {}, metrics)
        record["observed_at"] = "2026-09-31"
        with self.assertRaisesRegex(
            catalog.CatalogBuildError,
            "observed_at must use YYYY-MM-DD",
        ):
            catalog._validate_evaluations([record], models, {}, metrics)

    def test_missing_index_components_remain_unavailable(self) -> None:
        resources = {
            "models": [{"id": "example/model", "kind": "physical"}],
            "reasoning_families": [],
            "benchmarks": [
                {
                    "id": "example/bench@1.0.0",
                    "domain": "reasoning",
                    "default_profile": "standard",
                    "profiles": [{"id": "standard"}],
                    "metrics": [{"id": "score"}],
                }
            ],
            "evaluations": [],
            "indices": [
                {
                    "id": "example/index@1.0.0",
                    "scale": [0, 100],
                    "missing": {"policy": "require_all"},
                    "components": [
                        {
                            "benchmark": "example/bench@1.0.0",
                            "metric": "score",
                            "benchmark_profile": "standard",
                            "weight": 1.0,
                            "normalization": {"type": "identity"},
                        }
                    ],
                }
            ],
        }
        self.assertEqual(
            catalog._index_results(resources),
            [
                {
                    "model": "example/model",
                    "reasoning_effort": "default",
                    "index": "example/index@1.0.0",
                    "status": "missing",
                    "score": None,
                    "coverage": 0.0,
                    "components": [
                        {
                            "benchmark": "example/bench@1.0.0",
                            "metric": "score",
                            "benchmark_profile": "standard",
                            "weight": 1.0,
                            "status": "missing",
                            "value": None,
                            "normalized": None,
                        }
                    ],
                    "provenance": [],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
