from __future__ import annotations

import importlib.util
import sys
import unittest
from copy import deepcopy
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_model_catalog.py"
SPEC = importlib.util.spec_from_file_location("generate_model_catalog", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = catalog
SPEC.loader.exec_module(catalog)


class ModelCatalogSecurityValidationTests(unittest.TestCase):
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

    def test_security_validation_rejects_secret_fields_and_literals(self) -> None:
        with self.assertRaisesRegex(catalog.CatalogBuildError, "secret-like field"):
            catalog._validate_security({"provider": {"api_key": "not-published"}})
        with self.assertRaisesRegex(
            catalog.CatalogBuildError, "credential-like literal"
        ):
            catalog._validate_security({"source": "Bearer abcdefghijklmnop"})

    def test_provider_default_headers_cannot_store_credentials(self) -> None:
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
            "presentation": {"logo": "monogram", "monogram": "E", "monochrome": True},
            "conformance": {"status": "unverified"},
            "default_headers": {"Authorization": "not-a-secret"},
        }
        with self.assertRaisesRegex(catalog.CatalogBuildError, "credential headers"):
            catalog._validate_providers([provider], self.protocols)

    def test_provider_default_base_url_rejects_document_fragments(self) -> None:
        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "default_base_url": "https://api.example.test/v1#documentation",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#create"],
            "auth": {"strategy": "none", "header": "", "prefix": ""},
            "presentation": {"logo": "monogram", "monogram": "E", "monochrome": True},
            "conformance": {"status": "unverified"},
        }

        with self.assertRaisesRegex(catalog.CatalogBuildError, "transport URL"):
            catalog._validate_providers([provider], self.protocols)

        for malformed in (
            "https://:443/v1",
            "https://example.test:bad/v1",
            "https://example.test:65536/v1",
            "https://example.test:0/v1",
            "\nhttps://example.test/v1",
            "https://example.test/v1 ",
        ):
            with self.subTest(malformed=malformed):
                provider["default_base_url"] = malformed
                with self.assertRaisesRegex(catalog.CatalogBuildError, "transport URL"):
                    catalog._validate_providers([provider], self.protocols)

    def test_public_document_urls_are_validated_at_every_source_surface(self) -> None:
        _, resources, _ = catalog.load_and_validate()
        malformed = "javascript:alert(1)"
        models = {item["id"]: item for item in resources["models"]}
        reasoning = {item["id"]: item for item in resources["reasoning_families"]}
        reasoning_ids = set(reasoning)
        metrics = catalog._metric_catalog(resources["benchmarks"])

        physical = deepcopy(
            next(item for item in resources["models"] if item["kind"] == "physical")
        )
        physical["distribution"]["source"] = malformed
        with self.assertRaisesRegex(catalog.CatalogBuildError, "HTTPS document URL"):
            catalog._validate_models([physical], set(), reasoning_ids)

        physical = deepcopy(
            next(item for item in resources["models"] if item["kind"] == "physical")
        )
        physical["verification"]["source"] = "https://:443"
        with self.assertRaisesRegex(catalog.CatalogBuildError, "HTTPS document URL"):
            catalog._validate_models([physical], set(), reasoning_ids)

        providers = deepcopy(resources["providers"])
        binding = next(
            item
            for provider in providers
            for item in provider.get("models", [])
            if item.get("verification", {}).get("source")
        )
        binding["verification"]["source"] = malformed
        with self.assertRaisesRegex(catalog.CatalogBuildError, "HTTPS document URL"):
            catalog._validate_provider_bindings(
                {item["id"]: item for item in providers},
                models,
                {item["id"] for item in resources["protocols"]},
            )

        benchmark = deepcopy(resources["benchmarks"][0])
        benchmark["source"] = malformed
        with self.assertRaisesRegex(catalog.CatalogBuildError, "HTTPS document URL"):
            catalog._metric_catalog([benchmark])

        evaluation = deepcopy(resources["evaluations"][0])
        evaluation["evidence"]["source"] = malformed
        with self.assertRaisesRegex(catalog.CatalogBuildError, "HTTPS document URL"):
            catalog._validate_evaluations([evaluation], models, reasoning, metrics)

        index = deepcopy(resources["indices"][0])
        index["methodology"] = malformed
        with self.assertRaisesRegex(catalog.CatalogBuildError, "HTTPS document URL"):
            catalog._validate_indices([index], metrics)

        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#create"],
            "auth": {"strategy": "none", "header": "", "prefix": ""},
            "presentation": {
                "logo": f"url:{malformed}",
                "monogram": "E",
                "monochrome": True,
            },
            "conformance": {"status": "unverified"},
        }
        with self.assertRaisesRegex(catalog.CatalogBuildError, "external URLs"):
            catalog._validate_providers([provider], self.protocols)

        catalog._validate_https_url(
            "https://example.test/model-card#benchmarks", "document"
        )

    def test_provider_paths_and_headers_are_safe_for_runtime_materialization(
        self,
    ) -> None:
        provider = {
            "id": "example",
            "category": "model_api",
            "support_tier": "compatible",
            "protocols": ["openai/chat-completions@1"],
            "default_protocol": "openai/chat-completions@1",
            "supported_operations": ["openai/chat-completions@1#create"],
            "auth": {"strategy": "none", "header": "", "prefix": ""},
            "presentation": {"logo": "monogram", "monogram": "E", "monochrome": True},
            "conformance": {"status": "unverified"},
        }

        provider["path_overrides"] = {
            "openai/chat-completions@1#create": "relative/path"
        }
        with self.assertRaisesRegex(catalog.CatalogBuildError, "absolute path"):
            catalog._validate_providers([provider], self.protocols)

        provider.pop("path_overrides")
        for value in ("unsafe\x00value", "unsafe\x7fvalue"):
            with self.subTest(value=value):
                provider["default_headers"] = {"X-Catalog-Test": value}
                with self.assertRaisesRegex(
                    catalog.CatalogBuildError, "invalid header value"
                ):
                    catalog._validate_providers([provider], self.protocols)

        provider["default_headers"] = {"X-Catalog-Test": "one", "x-catalog-test": "two"}
        with self.assertRaisesRegex(catalog.CatalogBuildError, "duplicate"):
            catalog._validate_providers([provider], self.protocols)


if __name__ == "__main__":
    unittest.main()
