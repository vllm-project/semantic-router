"""The same fixtures are consumed by Go integration tests and the Console client."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from jsonschema import Draft202012Validator, ValidationError

from src.training.model_eval.provenance.crossref import (
    artifact_identity_digest,
    file_digest,
)
from src.training.model_eval.provenance.manifest import ManifestError, load_manifest
from src.training.model_eval.test_provenance import (
    artifact_manifest,
    dataset_manifest,
    evaluation_manifest,
    run_manifest,
)

from .contracts import CONTRACT_ROOT, SCHEMA, validate
from .provenance import validate_classifier_provenance


class ContractTests(unittest.TestCase):
    def test_api_error_accepts_unknown_codes(self):
        for code in ("unavailable", "rate_limited"):
            with self.subTest(code=code):
                validate({"code": code, "message": "Try again later"}, "APIError")

    def test_shared_fixtures(self):
        for name in ("selector", "neural"):
            with self.subTest(name=name):
                fixture = json.loads(
                    (CONTRACT_ROOT / f"testdata/{name}.json").read_text()
                )
                validate(fixture, "Fixture")
                self.assertEqual(
                    fixture["variant"]["id"],
                    fixture["evaluate_result"]["evaluations"][0]["variant_id"],
                )
                self.assertNotIn("artifacts", fixture["evaluate_result"])

    def test_shared_invalid_inputs(self):
        cases = json.loads((CONTRACT_ROOT / "testdata/invalid.json").read_text())
        for case in cases:
            with self.subTest(name=case["name"]), self.assertRaises(ValidationError):
                validate(case["value"], case["definition"])

    def test_schema_and_openapi_references(self):
        Draft202012Validator.check_schema(SCHEMA)
        api = yaml.safe_load((CONTRACT_ROOT / "training-v2.openapi.yaml").read_text())
        operation_ids = []
        for path, methods in api["paths"].items():
            for method, operation in methods.items():
                operation_ids.append(operation["operationId"])
                self.assertIn(method, ("get", "post"))
                if "{id}" in path:
                    self.assertIn("id", [p["name"] for p in operation["parameters"]])
        self.assertEqual(len(operation_ids), len(set(operation_ids)))
        self.assertEqual(set(api["paths"]["/data-snapshots/{id}"]), {"get"})

        def check_refs(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    ref = value["$ref"]
                    if ref.startswith("./training-v2.schema.json#/$defs/"):
                        self.assertIn(ref.rsplit("/", 1)[1], SCHEMA["$defs"])
                    else:
                        resolved = api
                        for part in ref.removeprefix("#/").split("/"):
                            resolved = resolved[part]
                for child in value.values():
                    check_refs(child)
            elif isinstance(value, list):
                for child in value:
                    check_refs(child)

        check_refs(api)

    def test_resource_and_worker_states(self):
        fixture = json.loads((CONTRACT_ROOT / "testdata/selector.json").read_text())
        result = fixture["train_result"]
        result["status"] = "pending"
        with self.assertRaises(ValidationError):
            validate(result, "WorkerResult")
        result["status"] = "failed"
        with self.assertRaises(ValidationError):
            validate(result, "WorkerResult")
        run = fixture["graph"]["run"]
        run["status"] = "skipped"
        with self.assertRaises(ValidationError):
            validate(run, "TrainingRun")

    def test_named_files_in_published_variants_and_worker_messages(self):
        fixture = json.loads((CONTRACT_ROOT / "testdata/neural.json").read_text())
        del fixture["artifact"]["provenance"]["manifest_bundle"]
        del fixture["train_result"]["artifacts"][0]["manifest_bundle"]
        validate(fixture, "Fixture")
        for variant, message, definition in (
            (fixture["variant"], fixture["variant"], "ArtifactVariant"),
            (
                fixture["train_result"]["artifacts"][0]["variants"][0],
                fixture["train_result"],
                "WorkerResult",
            ),
            (
                fixture["evaluate_request"]["inputs"][0],
                fixture["evaluate_request"],
                "WorkerRequest",
            ),
        ):
            with self.subTest(definition=definition):
                variant["files"]["../config.json"] = variant["files"].pop("config.json")
                with self.assertRaises(ValidationError):
                    validate(message, definition)

    def test_span_profile(self):
        validate(
            {
                "target_contract": "signal.spans/v1",
                "spans": {
                    "labels": ["person", "location"],
                    "offset_unit": "unicode-codepoint",
                },
            },
            "Profile",
        )

    def test_capabilities_fixture(self):
        fixture = json.loads((CONTRACT_ROOT / "testdata/capabilities.json").read_text())
        validate(fixture, "CapabilityCatalog")

    def test_capability_catalog_schema(self):
        catalog = {
            "schema_version": "semantic-router.training/v2",
            "targets": [
                {
                    "id": "target/selector.model-choice@v1",
                    "target_contract": "selector.model-choice/v1",
                    "display_name": "Model Selector",
                }
            ],
            "trainers": [
                {
                    "id": "trainer/selector@v1",
                    "component": {"name": "selector", "version": "1"},
                    "display_name": "Model Selector",
                    "supported_targets": ["selector.model-choice/v1"],
                    "supported_executors": ["executor/train@v1"],
                    "supported_hardware": ["hardware/cpu@v1"],
                    "supported_precisions": ["precision/fp32@v1"],
                    "produced_formats": ["format/selector-v2@v1"],
                }
            ],
            "architectures": [
                {
                    "id": "architecture/selector-tabular@v1",
                    "family": "selector",
                    "display_name": "Selector Tabular",
                    "supported_targets": ["selector.model-choice/v1"],
                    "supported_formats": ["format/selector-v2@v1"],
                    "supported_runtimes": ["runtime/native@v1"],
                }
            ],
            "executors": [
                {
                    "id": "executor/train@v1",
                    "component": {"name": "train", "version": "1"},
                    "display_name": "Standard Trainer",
                    "supported_hardware": ["hardware/cpu@v1"],
                    "isolation_level": "container",
                }
            ],
            "formats": [
                {
                    "id": "format/selector-v2@v1",
                    "component": {"name": "selector-v2", "version": "1"},
                    "display_name": "Selector V2",
                    "file_extensions": [".json"],
                    "direct_runtimes": ["runtime/native@v1"],
                }
            ],
            "runtimes": [
                {
                    "id": "runtime/native@v1",
                    "component": {"name": "native", "version": "1"},
                    "display_name": "Native Runtime",
                    "supported_targets": ["selector.model-choice/v1"],
                    "accepted_formats": ["format/selector-v2@v1"],
                    "supported_hardware": ["hardware/cpu@v1"],
                    "supported_precisions": ["precision/fp32@v1"],
                    "connector": "sr.native.embedded.v1",
                }
            ],
            "precisions": [
                {
                    "id": "precision/fp32@v1",
                    "name": "float32",
                    "bits_per_element": 32,
                }
            ],
            "hardware": [
                {
                    "id": "hardware/cpu@v1",
                    "provider": "cpu",
                    "device_type": "cpu",
                    "display_name": "Host CPU",
                    "supported_precisions": ["precision/fp32@v1"],
                }
            ],
        }
        validate(catalog, "CapabilityCatalog")

    def test_training_plan_request_and_response_schema(self):
        plan_req = {
            "schema_version": "semantic-router.training/v2",
            "target_contract": "selector.model-choice/v1",
            "trainer": "trainer/selector@v1",
            "training_hardware": "hardware/cpu@v1",
            "training_precision": "precision/fp32@v1",
            "parameters": {"seed": 42},
            "qualification_targets": [
                {
                    "key": "native",
                    "runtime": "runtime/native@v1",
                    "hardware": "hardware/cpu@v1",
                    "precision": "precision/fp32@v1",
                }
            ],
        }
        validate(plan_req, "TrainingPlanRequest")

        plan_resp = {
            "schema_version": "semantic-router.training/v2",
            "valid": True,
            "plan": {
                "target_contract": "selector.model-choice/v1",
                "trainer": "trainer/selector@v1",
                "architecture": "architecture/selector-tabular@v1",
                "executor": "executor/train@v1",
                "training_hardware": "hardware/cpu@v1",
                "training_precision": "precision/fp32@v1",
                "tasks": [
                    {"key": "train", "executor": {"name": "train", "version": "1"}},
                    {
                        "key": "evaluate",
                        "depends_on": ["train"],
                        "executor": {"name": "evaluate", "version": "1"},
                    },
                    {
                        "key": "qualify-native",
                        "depends_on": ["evaluate"],
                        "executor": {"name": "qualify", "version": "1"},
                    },
                ],
                "artifact_variants": [
                    {
                        "key": "primary",
                        "format": "format/selector-v2@v1",
                        "producing_task": "train",
                        "qualifications": [
                            {
                                "key": "native",
                                "task_key": "qualify-native",
                                "runtime": "runtime/native@v1",
                                "hardware": "hardware/cpu@v1",
                                "precision": "precision/fp32@v1",
                                "connector": "sr.native.embedded.v1",
                            }
                        ],
                    }
                ],
                "resolved_parameters": {"seed": 42, "normalize": True},
            },
            "diagnostics": [],
        }
        validate(plan_resp, "TrainingPlanResponse")

        plan_reject = {
            "schema_version": "semantic-router.training/v2",
            "valid": False,
            "diagnostics": [
                {
                    "code": "INCOMPATIBLE_HARDWARE",
                    "severity": "error",
                    "field": "training_hardware",
                    "message": "trainer does not support hardware/cuda@v1",
                    "remediation": "Select hardware/cpu@v1",
                }
            ],
        }
        validate(plan_reject, "TrainingPlanResponse")


class ClassifierProvenanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.run_manifest = run_manifest()
        self.profile = {
            "target_contract": "signal.label-scores/v1",
            "classifier": {"label_mapping": self.run_manifest["label_mapping"]},
        }
        self.base_model = {
            "repository": self.run_manifest["base_model"]["repo"],
            "revision": self.run_manifest["base_model"]["revision"],
        }
        # Owned storage paths deliberately differ from the manifest's logical names.
        self.owned_path = self.directory / "owned-config"
        self.owned_path.write_bytes(b'{"model": 1}')
        self.owned_files = {"file_config": self.owned_path}
        entry = {
            "path": "config.json",
            "size_bytes": self.owned_path.stat().st_size,
            "digest": file_digest(self.owned_path),
        }
        self.artifact = artifact_manifest(files=[entry])
        self.artifact["identity"]["digest"] = artifact_identity_digest([entry])
        evaluation = evaluation_manifest()
        evaluation["artifact_ref"]["digest"] = self.artifact["identity"]["digest"]
        self.variant = {
            "format": {"name": "safetensors", "version": "1"},
            "files": {
                entry["path"]: {
                    "handle": "file_config",
                    "digest": entry["digest"],
                    "size_bytes": entry["size_bytes"],
                }
            },
        }
        for manifest in (
            dataset_manifest(),
            self.run_manifest,
            self.artifact,
            evaluation,
        ):
            (self.directory / f"{manifest['kind']}.manifest.yaml").write_text(
                yaml.safe_dump(manifest)
            )

    def validate_provenance(self):
        return validate_classifier_provenance(
            self.directory,
            self.profile,
            self.base_model,
            self.variant,
            self.owned_files.__getitem__,
        )

    def test_existing_bundle_guarantees_and_profile_binding(self):
        with patch(
            "src.training.model_eval.provenance.manifest.load_manifest",
            wraps=load_manifest,
        ) as load:
            self.validate_provenance()
            self.assertEqual(load.call_count, 4)
        self.profile["classifier"]["label_mapping"] = {"jailbreak": 0, "benign": 1}
        with self.assertRaisesRegex(ManifestError, "label_mapping differ"):
            self.validate_provenance()
        self.profile["classifier"]["label_mapping"] = self.run_manifest["label_mapping"]
        self.base_model["revision"] = "e" * 40
        with self.assertRaisesRegex(ManifestError, "base_model differ"):
            self.validate_provenance()
        self.base_model["revision"] = self.run_manifest["base_model"]["revision"]
        self.run_manifest["base_model"]["revision"] = "main"
        (self.directory / "run.manifest.yaml").write_text(
            yaml.safe_dump(self.run_manifest)
        )
        with self.assertRaises(ManifestError):
            self.validate_provenance()

    def test_worker_result_for_different_bytes_is_not_qualified(self):
        other = self.directory / "other-config"
        other.write_bytes(b'{"model": 2}')
        self.owned_files["file_other"] = other
        self.variant["files"]["config.json"] = {
            "handle": "file_other",
            "digest": file_digest(other),
            "size_bytes": other.stat().st_size,
        }
        validate(
            {
                "schema_version": "semantic-router.training/v2",
                "status": "succeeded",
                "artifacts": [{"profile": self.profile, "variants": [self.variant]}],
            },
            "WorkerResult",
        )
        with self.assertRaisesRegex(ManifestError, "do not match"):
            self.validate_provenance()

    def test_variant_file_inventory_must_match(self):
        original = copy.deepcopy(self.variant["files"])
        for change in ("missing", "extra", "renamed", "size"):
            with self.subTest(change=change):
                files = copy.deepcopy(original)
                if change == "missing":
                    del files["config.json"]
                elif change == "extra":
                    files["extra.json"] = files["config.json"]
                elif change == "renamed":
                    files["other.json"] = files.pop("config.json")
                else:
                    files["config.json"]["size_bytes"] += 1
                self.variant["files"] = files
                with self.assertRaisesRegex(ManifestError, "do not match"):
                    self.validate_provenance()

    def test_owned_bytes_must_match_even_when_declared_digest_matches(self):
        for content in (b'{"model": 2}', b"truncated"):
            with self.subTest(content=content):
                self.owned_path.write_bytes(content)
                with self.assertRaisesRegex(ManifestError, "differs from its manifest"):
                    self.validate_provenance()

    def test_checks_the_variant_handle_not_a_same_named_local_file(self):
        other = self.directory / "other-config"
        other.write_bytes(b'{"model": 2}')
        self.owned_files["file_other"] = other
        self.variant["files"]["config.json"]["handle"] = "file_other"
        (self.directory / "config.json").write_bytes(self.owned_path.read_bytes())
        with self.assertRaisesRegex(ManifestError, "differs from its manifest"):
            self.validate_provenance()

    def test_missing_owned_file_is_rejected(self):
        self.owned_path.unlink()
        with self.assertRaisesRegex(ManifestError, "is missing"):
            self.validate_provenance()

    def test_matching_artifact_must_have_evaluation(self):
        unevaluated = copy.deepcopy(self.artifact)
        unevaluated["id"] = "unevaluated-model"
        self.artifact["files"][0]["path"] = "other.json"
        self.artifact["identity"]["digest"] = artifact_identity_digest(
            self.artifact["files"]
        )
        evaluation = evaluation_manifest()
        evaluation["artifact_ref"]["digest"] = self.artifact["identity"]["digest"]
        for name, manifest in (
            ("artifact", self.artifact),
            ("evaluation", evaluation),
            ("unevaluated", unevaluated),
        ):
            (self.directory / f"{name}.manifest.yaml").write_text(
                yaml.safe_dump(manifest)
            )
        with self.assertRaisesRegex(ManifestError, "do not match"):
            self.validate_provenance()


if __name__ == "__main__":
    unittest.main()
