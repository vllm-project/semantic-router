"""The same fixtures are consumed by Go integration tests and the Console client."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from jsonschema import Draft202012Validator, ValidationError

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
        api = yaml.safe_load((CONTRACT_ROOT / "training-v1.openapi.yaml").read_text())
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
                    if ref.startswith("./training-v1.schema.json#/$defs/"):
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


class ClassifierProvenanceTests(unittest.TestCase):
    def test_existing_bundle_guarantees_and_profile_binding(self):

        run = run_manifest()
        profile = {
            "target_contract": "signal.label-scores/v1",
            "classifier": {
                "label_mapping": run["label_mapping"],
            },
        }
        base_model = {
            "repository": run["base_model"]["repo"],
            "revision": run["base_model"]["revision"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for manifest in (
                dataset_manifest(),
                run,
                artifact_manifest(),
                evaluation_manifest(),
            ):
                (directory / f"{manifest['kind']}.manifest.yaml").write_text(
                    yaml.safe_dump(manifest)
                )
            with patch(
                "src.training.model_eval.provenance.manifest.load_manifest",
                wraps=load_manifest,
            ) as load:
                validate_classifier_provenance(directory, profile, base_model)
                self.assertEqual(load.call_count, 4)
            changed = copy.deepcopy(profile)
            changed["classifier"]["label_mapping"] = {"jailbreak": 0, "benign": 1}
            with self.assertRaisesRegex(ManifestError, "label_mapping differ"):
                validate_classifier_provenance(directory, changed, base_model)
            changed_model = {**base_model, "revision": "e" * 40}
            with self.assertRaisesRegex(ManifestError, "base_model differ"):
                validate_classifier_provenance(directory, profile, changed_model)
            run["base_model"]["revision"] = "main"
            (directory / "run.manifest.yaml").write_text(yaml.safe_dump(run))
            with self.assertRaises(ManifestError):
                validate_classifier_provenance(directory, profile, base_model)


if __name__ == "__main__":
    unittest.main()
