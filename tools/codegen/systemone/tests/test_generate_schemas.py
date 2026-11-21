"""Tests of the System One schema generator: freshness, collisions and drift."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_schemas as gen


def published_from(standard: dict) -> dict:
    """An OpenAPI document with the structure our description records."""

    def rewrite(schema):
        if isinstance(schema, list):
            return [rewrite(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        result = {}
        for key, value in schema.items():
            if key == "summary":
                result["description"] = value
            elif key == "$ref":
                result[key] = value.replace("#/schemas/", "#/components/schemas/")
            elif key == "discriminator":
                result[key] = {
                    "propertyName": value["propertyName"],
                    "mapping": {
                        name: ref.replace("#/schemas/", "#/components/schemas/")
                        for name, ref in value["mapping"].items()
                    },
                }
            else:
                result[key] = rewrite(value)
        return result

    paths: dict = {}
    for endpoint in standard["endpoints"]:
        operation: dict = {"responses": {}}
        for status, schema in endpoint["responses"].items():
            operation["responses"][status] = {
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{schema}"}
                    }
                }
            }
        if endpoint.get("request"):
            content = {
                kind: {"schema": {"type": "object"}} for kind in endpoint["body"]
            }
            content["application/json"] = {
                "schema": {"$ref": f"#/components/schemas/{endpoint['request']}"}
            }
            operation["requestBody"] = {"content": content}
        operation["parameters"] = [
            {
                "name": name,
                "in": "header",
                "schema": {"type": "string", "enum": spec["enum"]},
            }
            for name, spec in (endpoint.get("headers") or {}).items()
        ]
        paths.setdefault(endpoint["path"], {})[endpoint["method"].lower()] = operation
    return {
        "openapi": "3.1.0",
        "info": {"title": "Published", "version": standard["source_version"]},
        "paths": paths,
        "components": {
            "schemas": {
                name: rewrite(schema) for name, schema in standard["schemas"].items()
            }
        },
    }


class GeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = gen.load_specs(gen.REPO_ROOT)

    def test_outputs_are_fresh(self) -> None:
        self.assertEqual(gen.main(["--check"]), 0)

    def test_no_collisions(self) -> None:
        self.assertEqual(gen.collisions(self.specs), [])

    def test_a_field_both_specs_define_differently_collides(self) -> None:
        specs = copy.deepcopy(self.specs)
        specs.extension["fields"].append(
            {
                "name": "model",
                "level": "request",
                "since": "5.0",
                "families": "all",
                "standard_target": "strip",
                "schema": {"type": "integer"},
                "summary": "A model number.",
            }
        )
        problems = gen.collisions(specs)
        self.assertEqual(len(problems), 1)
        self.assertIn("request.model", problems[0])

    def test_a_field_both_specs_define_alike_does_not_collide(self) -> None:
        specs = copy.deepcopy(self.specs)
        specs.extension["fields"].append(
            {
                "name": "model",
                "level": "request",
                "since": "5.0",
                "families": "all",
                "standard_target": "strip",
                "schema": {"type": "string"},
                "summary": "The model.",
            }
        )
        self.assertEqual(gen.collisions(specs), [])

    def test_removing_a_standard_field_collides(self) -> None:
        specs = copy.deepcopy(self.specs)
        specs.extension["removed"].append(
            {"name": "state", "level": "request", "replaced_by": "nothing"}
        )
        self.assertTrue(
            any("request.state" in problem for problem in gen.collisions(specs))
        )

    def test_an_unsupported_keyword_is_refused(self) -> None:
        with self.assertRaises(gen.SpecError):
            gen.check_keywords({"type": "string", "pattern": "^a$"}, "here")

    def test_standard_targets(self) -> None:
        targets = {
            (entry.level, entry.name): entry.standard_target
            for entry in gen.extension_fields(self.specs)
        }
        self.assertEqual(targets[("request", "roles")], "strip")
        self.assertEqual(targets[("request", "options")], "strip")
        self.assertEqual(targets[("question", "preset")], "strip")
        self.assertEqual(targets[("question", "head")], "refuse")
        self.assertEqual(targets[("question", "require_full_input")], "refuse")
        self.assertEqual(targets[("question_type", "span")], "refuse")
        self.assertEqual(targets[("asset_type", "video")], "refuse")
        self.assertEqual(targets[("answer", "input_coverage")], "absent")

    def test_standard_fields(self) -> None:
        fields = {
            (entry.level, entry.name): entry
            for entry in gen.standard_fields(self.specs.standard)
        }
        self.assertEqual(
            sorted(name for level, name in fields if level == "request"),
            ["assets", "model", "questions", "state"],
        )
        self.assertEqual(
            sorted(name for level, name in fields if level == "question_type"),
            ["box", "choice", "noul", "score"],
        )
        self.assertEqual(
            fields[("question", "criteria")].applies_to, ["choice", "noul", "score"]
        )

    def test_drift_is_empty_for_an_identical_document(self) -> None:
        self.assertEqual(
            gen.drift(self.specs.standard, published_from(self.specs.standard)), []
        )

    def test_drift_finds_a_changed_type(self) -> None:
        published = published_from(self.specs.standard)
        published["components"]["schemas"]["Usage"]["properties"]["input_tokens"][
            "type"
        ] = "number"
        problems = gen.drift(self.specs.standard, published)
        self.assertEqual(len(problems), 1)
        self.assertIn("schemas.Usage", problems[0])

    def test_drift_finds_a_new_field_and_endpoint(self) -> None:
        published = published_from(self.specs.standard)
        published["components"]["schemas"]["SystemOneRequest"]["properties"][
            "options"
        ] = {"type": "object"}
        published["paths"]["/v1/systemone/batch"] = {"post": {"responses": {}}}
        problems = gen.drift(self.specs.standard, published)
        self.assertTrue(any("SystemOneRequest" in problem for problem in problems))
        self.assertTrue(any("/v1/systemone/batch" in problem for problem in problems))

    @unittest.skipUnless(
        os.environ.get("SYSTEMONE_PUBLISHED_SPEC"),
        "set SYSTEMONE_PUBLISHED_SPEC to the published document",
    )
    def test_drift_against_the_published_document(self) -> None:
        published = json.loads(
            Path(os.environ["SYSTEMONE_PUBLISHED_SPEC"]).read_text(encoding="utf-8")
        )
        self.assertEqual(gen.drift(self.specs.standard, published), [])

    def test_golden_cases_are_python_outputs(self) -> None:
        golden = json.loads(gen.render_golden())
        for case in golden["cases"]:
            self.assertEqual(
                case["canonical"],
                json.dumps(json.loads(case["input"]), ensure_ascii=False),
            )


if __name__ == "__main__":
    unittest.main()
