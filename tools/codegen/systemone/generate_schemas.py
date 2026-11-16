#!/usr/bin/env python3
"""Generate the System One schema tables, field tables and reference pages.

The sources are in config/schemas/systemone: our description of the standard
(systemone-0.2.schema.yaml) and the vllm-sr extension spec
(vllm-sr-5.0.extensions.yaml). The outputs are the Go field registry and
specification table, the model runtime's field tables, the canonical JSON
golden file and the generated Extension reference and Compatibility matrix
pages.

--check fails when an output is stale or when both specs define a field with
different definitions. --drift compares the published standard with our
description and fails on any structural difference; it reads the published
document from --spec-file, or downloads it into a cache outside the
repository.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = Path("config/schemas/systemone")
STANDARD_FILE = SPEC_DIR / "systemone-0.2.schema.yaml"
EXTENSION_FILE = SPEC_DIR / "vllm-sr-5.0.extensions.yaml"
LEGACY_FILE = SPEC_DIR / "vllm-sr-4.0.openapi.yaml"
GO_FIELDS = Path("src/semantic-router/pkg/systemoneprotocol/zz_generated_fields.go")
GO_SPECS = Path("src/semantic-router/pkg/systemonecodec/zz_generated_specs.go")
PY_FIELDS = Path("src/model-runtime/vllm_srun/api/systemone_fields.py")
GOLDEN = Path(
    "src/semantic-router/pkg/systemoneprotocol/testdata/canonical_golden.json"
)
INTERNAL_CORPUS = Path(
    "src/semantic-router/pkg/systemonecodec/testdata/internal_corpus.json"
)
FAMILIES_DIR = Path("src/model-runtime/vllm_srun/families")
DOC_REFERENCE = Path("website/docs/system-one/reference/extension-reference.md")
DOC_MATRIX = Path("website/docs/system-one/reference/compatibility-matrix.md")
GENERATOR = "tools/codegen/systemone/generate_schemas.py"
DRIFT_CACHE = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "vllm-sr"
    / "systemone"
)

# JSON Schema keywords the specs may use; anything else is refused so a
# validator never silently ignores a constraint.
KEYWORDS = {
    "$ref",
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "anyOf",
    "oneOf",
    "const",
    "enum",
    "minProperties",
    "maxProperties",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "discriminator",
}
# How deep $refs may nest before a spec counts as a cycle.
MAX_REF_DEPTH = 64
# Annotations: kept in a spec, dropped before comparing two definitions.
ANNOTATIONS = {
    "summary",
    "title",
    "description",
    "examples",
    "example",
    "x-accepts",
    "default",
}
REQUEST_LEVELS = ("request", "question", "question_type", "asset_type")
RESPONSE_LEVELS = ("answer", "response", "error", "listing_entry")
LEVELS = REQUEST_LEVELS + RESPONSE_LEVELS + ("endpoint",)
TARGETS = {"strip", "refuse", "absent"}
GO_LEVELS = {
    "request": "LevelRequest",
    "question": "LevelQuestion",
    "question_type": "LevelQuestionType",
    "asset": "LevelAsset",
    "asset_type": "LevelAssetType",
    "answer": "LevelAnswer",
    "response": "LevelResponse",
    "error": "LevelError",
    "listing_entry": "LevelListingEntry",
    "endpoint": "LevelEndpoint",
}
GO_TARGETS = {
    "strip": "TargetStrip",
    "refuse": "TargetRefuse",
    "absent": "TargetAbsent",
    "": '""',
}


class SpecError(ValueError):
    """A specification that does not validate."""


@dataclass
class Field:
    level: str
    name: str
    origin: str
    since: str = ""
    families: list[str] | None = None
    applies_to: list[str] | None = None
    endpoints: list[str] | None = None
    standard_target: str = ""
    schema: Any = None
    summary: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Specs:
    standard: dict[str, Any]
    standard_sha256: str
    extension: dict[str, Any]
    extension_sha256: str
    legacy_sha256: str = ""


# ---------------------------------------------------------------- loading


def load_specs(root: Path) -> Specs:
    standard_bytes = (root / STANDARD_FILE).read_bytes()
    extension_bytes = (root / EXTENSION_FILE).read_bytes()
    standard = yaml.safe_load(standard_bytes)
    extension = yaml.safe_load(extension_bytes)
    validate_standard(standard)
    validate_extension(extension, standard)
    return Specs(
        standard,
        hashlib.sha256(standard_bytes).hexdigest(),
        extension,
        hashlib.sha256(extension_bytes).hexdigest(),
        hashlib.sha256((root / LEGACY_FILE).read_bytes()).hexdigest(),
    )


def check_keywords(schema: Any, where: str) -> None:
    if isinstance(schema, list):
        for index, item in enumerate(schema):
            check_keywords(item, f"{where}[{index}]")
        return
    if not isinstance(schema, dict):
        raise SpecError(f"{where}: a schema must be an object")
    for key, value in schema.items():
        if key in ANNOTATIONS:
            continue
        if key not in KEYWORDS:
            raise SpecError(f"{where}: unsupported keyword {key!r}")
        if key == "properties":
            for name, child in value.items():
                check_keywords(child, f"{where}.properties.{name}")
        elif (
            key in ("items", "additionalProperties") and isinstance(value, dict)
        ) or key in ("anyOf", "oneOf"):
            check_keywords(value, f"{where}.{key}")


def validate_standard(standard: Any) -> None:
    if not isinstance(standard, dict):
        raise SpecError(f"{STANDARD_FILE}: must be a mapping")
    for key in (
        "id",
        "source",
        "source_version",
        "source_sha256",
        "endpoints",
        "schemas",
    ):
        if key not in standard:
            raise SpecError(f"{STANDARD_FILE}: {key} is required")
    if not re.fullmatch(r"[0-9a-f]{64}", str(standard["source_sha256"])):
        raise SpecError(f"{STANDARD_FILE}: source_sha256 must be a sha256")
    for name, schema in standard["schemas"].items():
        check_keywords(schema, f"{STANDARD_FILE}: schemas.{name}")
    names = set(standard["schemas"])
    for ref in refs(standard["schemas"]):
        if (
            not ref.startswith("#/schemas/")
            or ref.removeprefix("#/schemas/") not in names
        ):
            raise SpecError(f"{STANDARD_FILE}: unresolved $ref {ref!r}")
    for endpoint in standard["endpoints"]:
        for schema in [
            endpoint.get("request"),
            *endpoint.get("responses", {}).values(),
        ]:
            if schema is not None and schema not in names:
                raise SpecError(
                    f"{STANDARD_FILE}: endpoint {endpoint['path']} names unknown schema {schema!r}"
                )


def validate_extension(extension: Any, standard: dict[str, Any]) -> None:
    where = str(EXTENSION_FILE)
    if not isinstance(extension, dict):
        raise SpecError(f"{where}: must be a mapping")
    for key in (
        "id",
        "extends",
        "extension_version",
        "families",
        "levels",
        "fields",
        "removed",
    ):
        if key not in extension:
            raise SpecError(f"{where}: {key} is required")
    if extension["extends"] != standard["id"]:
        raise SpecError(
            f"{where}: extends {extension['extends']!r}, not {standard['id']!r}"
        )
    if not re.fullmatch(r"[0-9]+\.[0-9]+", str(extension["extension_version"])):
        raise SpecError(f"{where}: extension_version must be <major>.<minor>")
    families = set(extension["families"])
    definitions = extension.get("definitions") or {}
    for name, schema in definitions.items():
        check_keywords(schema, f"{where}: definitions.{name}")
    standard_types = set(question_type_mapping(standard))
    extension_types = {
        entry["name"]
        for entry in extension["fields"]
        if entry.get("level") == "question_type"
    }
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(extension["fields"]):
        label = f"{where}: fields[{index}] ({entry.get('name')})"
        for key in ("name", "level", "since", "families", "standard_target", "summary"):
            if key not in entry:
                raise SpecError(f"{label}: {key} is required")
        if entry["level"] not in LEVELS:
            raise SpecError(f"{label}: level must be one of {', '.join(LEVELS)}")
        if (entry["level"], entry["name"]) in seen:
            raise SpecError(f"{label}: defined twice at level {entry['level']}")
        seen.add((entry["level"], entry["name"]))
        if not re.fullmatch(r"[0-9]+\.[0-9]+", str(entry["since"])):
            raise SpecError(f"{label}: since must be <major>.<minor>")
        if entry["families"] != "all" and not set(entry["families"]) <= families:
            raise SpecError(
                f"{label}: families must be all or a subset of {sorted(families)}"
            )
        target = entry["standard_target"]
        if target not in TARGETS:
            raise SpecError(f"{label}: standard_target must be strip, refuse or absent")
        if entry["level"] in REQUEST_LEVELS and target == "absent":
            raise SpecError(
                f"{label}: a request field is stripped or refused on a standard target"
            )
        if entry["level"] in RESPONSE_LEVELS and target != "absent":
            raise SpecError(
                f"{label}: a response field is absent from a standard target"
            )
        applies = entry.get("applies_to")
        if (
            applies is not None
            and applies != "all"
            and not set(applies) <= standard_types | extension_types
        ):
            raise SpecError(f"{label}: applies_to names unknown question types")
        schemas = (
            [entry.get("schema")]
            if entry["level"] != "endpoint"
            else [entry.get("request"), entry.get("response")]
        )
        if not any(schema is not None for schema in schemas):
            raise SpecError(f"{label}: a schema is required")
        for schema in schemas:
            if schema is not None:
                check_keywords(schema, label)
                resolve(schema, extension, standard)
    for index, entry in enumerate(extension["removed"]):
        for key in ("name", "level", "replaced_by"):
            if key not in entry:
                raise SpecError(f"{where}: removed[{index}]: {key} is required")


def refs(schema: Any) -> list[str]:
    found: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "$ref":
                found.append(value)
            elif key == "discriminator":
                continue
            else:
                found += refs(value)
    elif isinstance(schema, list):
        for item in schema:
            found += refs(item)
    return found


def pointer(document: Any, path: str) -> Any:
    node = document
    for part in path.strip("/").split("/"):
        if not isinstance(node, dict) or part not in node:
            raise SpecError(f"unresolved $ref path {path!r}")
        node = node[part]
    return node


def resolve(
    schema: Any, extension: dict[str, Any], standard: dict[str, Any], depth: int = 0
) -> Any:
    """The schema with every $ref inlined: extension definitions and parts of the standard."""
    if depth > MAX_REF_DEPTH:
        raise SpecError("a $ref cycle")
    if isinstance(schema, list):
        return [resolve(item, extension, standard, depth + 1) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/definitions/"):
            target = pointer(extension, ref.removeprefix("#"))
            return resolve(target, extension, standard, depth + 1)
        if ref.startswith("standard#/"):
            return resolve_standard(
                pointer(standard, ref.removeprefix("standard#")), standard, depth + 1
            )
        raise SpecError(f"unresolved $ref {ref!r}")
    return {
        key: resolve(value, extension, standard, depth + 1)
        for key, value in schema.items()
    }


def resolve_standard(schema: Any, standard: dict[str, Any], depth: int = 0) -> Any:
    if depth > MAX_REF_DEPTH:
        raise SpecError("a $ref cycle")
    if isinstance(schema, list):
        return [resolve_standard(item, standard, depth + 1) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        return resolve_standard(
            pointer(standard, schema["$ref"].removeprefix("#")), standard, depth + 1
        )
    return {
        key: resolve_standard(value, standard, depth + 1)
        for key, value in schema.items()
        if key != "discriminator"
    }


def normalize(schema: Any) -> Any:
    """A schema without annotations, with required sorted, for comparing definitions."""
    if isinstance(schema, list):
        return [normalize(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ANNOTATIONS:
            continue
        if key == "required":
            normalized[key] = sorted(value)
        elif key == "properties":
            normalized[key] = {
                name: normalize(child) for name, child in sorted(value.items())
            }
        else:
            normalized[key] = normalize(value)
    return dict(sorted(normalized.items()))


# ------------------------------------------------------------ the registry


def question_type_mapping(standard: dict[str, Any]) -> dict[str, str]:
    mapping = standard["schemas"]["Question"]["discriminator"]["mapping"]
    return {name: ref.removeprefix("#/schemas/") for name, ref in mapping.items()}


def answer_type_mapping(standard: dict[str, Any]) -> dict[str, str]:
    mapping = standard["schemas"]["Answer"]["discriminator"]["mapping"]
    return {name: ref.removeprefix("#/schemas/") for name, ref in mapping.items()}


def standard_fields(standard: dict[str, Any]) -> list[Field]:
    """The standard's fields, by level, in the description's order."""
    schemas = standard["schemas"]
    sid = standard["id"]
    fields: list[Field] = []

    def add(
        level: str,
        name: str,
        schema: Any,
        summary: str = "",
        applies: list[str] | None = None,
    ) -> None:
        for existing in fields:
            if existing.level == level and existing.name == name:
                if applies is not None and existing.applies_to is not None:
                    existing.applies_to = sorted(
                        set(existing.applies_to) | set(applies)
                    )
                return
        fields.append(
            Field(
                level,
                name,
                sid,
                schema=resolve_standard(schema, standard),
                summary=summary,
                applies_to=applies,
            )
        )

    def properties_of(
        component: str, level: str, applies: list[str] | None = None
    ) -> None:
        for name, schema in schemas[component].get("properties", {}).items():
            add(level, name, schema, applies=applies)

    properties_of("SystemOneRequest", "request")
    for question_type, component in question_type_mapping(standard).items():
        properties_of(component, "question", [question_type])
        add(
            "question_type",
            question_type,
            schemas[component],
            schemas[component].get("summary", ""),
        )
    properties_of("Asset", "asset")
    asset_type = schemas["Asset"]["properties"]["type"].get("const")
    add("asset_type", asset_type, schemas["Asset"], schemas["Asset"].get("summary", ""))
    for answer_type, component in answer_type_mapping(standard).items():
        properties_of(component, "answer", [answer_type])
    properties_of("SystemOneResponse", "response")
    properties_of("HTTPBadRequestError", "error")
    properties_of("HTTPValidationError", "error")
    properties_of("ModelMetadata", "listing_entry")
    for endpoint in standard["endpoints"]:
        name = f"{endpoint['method']} {endpoint['path']}"
        request = {"$ref": endpoint["request"]} if endpoint.get("request") else {}
        fields.append(
            Field(
                "endpoint",
                name,
                sid,
                schema=request,
                summary=endpoint.get("summary", ""),
            )
        )
    return fields


def extension_fields(specs: Specs) -> list[Field]:
    extension, standard = specs.extension, specs.standard
    fields: list[Field] = []
    for entry in extension["fields"]:
        families = None if entry["families"] == "all" else list(entry["families"])
        applies = entry.get("applies_to")
        applies = None if applies in (None, "all") else list(applies)
        if entry["level"] == "endpoint":
            schema = {
                key: resolve(entry[key], extension, standard)
                for key in ("request", "response")
                if key in entry
            }
        else:
            schema = resolve(entry["schema"], extension, standard)
        fields.append(
            Field(
                entry["level"],
                entry["name"],
                extension["id"],
                since=str(entry["since"]),
                families=families,
                applies_to=applies,
                endpoints=entry.get("endpoints"),
                standard_target=entry["standard_target"],
                schema=schema,
                summary=" ".join(str(entry["summary"]).split()),
            )
        )
    return fields


def collisions(specs: Specs) -> list[str]:
    """Fields both specs define with different definitions, and removed forms the standard defines."""
    problems: list[str] = []
    standard = standard_fields(specs.standard)
    for entry in extension_fields(specs):
        for existing in standard:
            if existing.level != entry.level or existing.name != entry.name:
                continue
            if entry.level == "endpoint" or normalize(existing.schema) != normalize(
                entry.schema
            ):
                problems.append(
                    f"{entry.level}.{entry.name} is defined by {existing.origin} and by {entry.origin} with "
                    "different definitions; the standard comes first, so drop or rename the extension field"
                )
            else:
                print(
                    f"note: {entry.level}.{entry.name} is now defined by {existing.origin}; drop it from {entry.origin}",
                    file=sys.stderr,
                )
    for removed in specs.extension["removed"]:
        if any(
            existing.level == removed["level"] and existing.name == removed["name"]
            for existing in standard
        ):
            problems.append(
                f"removed {removed['level']}.{removed['name']} is a field of {specs.standard['id']}"
            )
    return problems


# ------------------------------------------------------------------- drift


def official_structure(document: dict[str, Any]) -> dict[str, Any]:
    """The published OpenAPI document reduced to our description's structure."""

    def rewrite(schema: Any) -> Any:
        if isinstance(schema, list):
            return [rewrite(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        result: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "$ref":
                result[key] = value.replace("#/components/schemas/", "#/schemas/")
            elif key == "discriminator":
                result[key] = {
                    "propertyName": value.get("propertyName"),
                    "mapping": {
                        name: ref.replace("#/components/schemas/", "#/schemas/")
                        for name, ref in value.get("mapping", {}).items()
                    },
                }
            else:
                result[key] = rewrite(value)
        return result

    schemas = {
        name: rewrite(schema)
        for name, schema in document["components"]["schemas"].items()
    }
    endpoints = []
    for path, operations in document["paths"].items():
        for method, operation in operations.items():
            body = (operation.get("requestBody") or {}).get("content") or {}
            request = (body.get("application/json") or {}).get("schema", {}).get("$ref")
            responses = {}
            for status, response in operation.get("responses", {}).items():
                ref = (
                    ((response.get("content") or {}).get("application/json") or {})
                    .get("schema", {})
                    .get("$ref")
                )
                responses[str(status)] = (
                    ref.replace("#/components/schemas/", "") if ref else None
                )
            headers = {
                parameter["name"]: (parameter.get("schema") or {}).get("enum")
                for parameter in operation.get("parameters", [])
                if parameter.get("in") == "header"
            }
            endpoints.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "request": (
                        request.replace("#/components/schemas/", "")
                        if request
                        else None
                    ),
                    "body": sorted(body),
                    "headers": headers,
                    "responses": responses,
                }
            )
    return {
        "version": document.get("info", {}).get("version"),
        "schemas": schemas,
        "endpoints": endpoints,
    }


def described_structure(standard: dict[str, Any]) -> dict[str, Any]:
    endpoints = []
    for endpoint in standard["endpoints"]:
        endpoints.append(
            {
                "method": endpoint["method"],
                "path": endpoint["path"],
                "request": endpoint.get("request"),
                "body": sorted(endpoint.get("body") or []),
                "headers": {
                    name: spec.get("enum")
                    for name, spec in (endpoint.get("headers") or {}).items()
                },
                "responses": {
                    str(status): schema
                    for status, schema in endpoint["responses"].items()
                },
            }
        )
    return {
        "version": standard["source_version"],
        "schemas": standard["schemas"],
        "endpoints": endpoints,
    }


def drift(standard: dict[str, Any], document: dict[str, Any]) -> list[str]:
    """Every structural difference between the published document and our description."""
    official, described = official_structure(document), described_structure(standard)
    problems: list[str] = []
    if official["version"] != described["version"]:
        problems.append(
            f"version: published {official['version']!r}, described {described['version']!r}"
        )
    for name in sorted(set(official["schemas"]) | set(described["schemas"])):
        if name not in described["schemas"]:
            problems.append(f"schemas.{name}: published, not described")
        elif name not in official["schemas"]:
            problems.append(f"schemas.{name}: described, not published")
        elif normalize(official["schemas"][name]) != normalize(
            described["schemas"][name]
        ):
            problems.append(
                f"schemas.{name}: differs: published {json.dumps(normalize(official['schemas'][name]), sort_keys=True)}"
            )
    key = lambda endpoint: (endpoint["method"], endpoint["path"])  # noqa: E731
    published = {key(endpoint): endpoint for endpoint in official["endpoints"]}
    ours = {key(endpoint): endpoint for endpoint in described["endpoints"]}
    for endpoint in sorted(set(published) | set(ours)):
        if endpoint not in ours:
            problems.append(f"endpoint {' '.join(endpoint)}: published, not described")
        elif endpoint not in published:
            problems.append(f"endpoint {' '.join(endpoint)}: described, not published")
        else:
            for part in ("request", "body", "headers", "responses"):
                if published[endpoint][part] != ours[endpoint][part]:
                    problems.append(
                        f"endpoint {' '.join(endpoint)} {part}: published {published[endpoint][part]!r}, described {ours[endpoint][part]!r}"
                    )
    return problems


def fetch_published(url: str, cache: Path) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data: bytes = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"drift: {url} is not reachable ({exc}); skipped", file=sys.stderr)
        return None
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "openapi.json").write_bytes(data)
    return data


def run_drift(root: Path, spec_file: str | None, url: str | None) -> int:
    standard = yaml.safe_load((root / STANDARD_FILE).read_bytes())
    if spec_file:
        data = Path(spec_file).read_bytes()
    else:
        fetched = fetch_published(url or standard["source"], DRIFT_CACHE)
        if fetched is None:
            return 0
        data = fetched
    digest = hashlib.sha256(data).hexdigest()
    problems = drift(standard, json.loads(data))
    if digest != standard["source_sha256"]:
        print(
            f"drift: the published document's sha256 is {digest}, not the verified {standard['source_sha256']}",
            file=sys.stderr,
        )
    if problems:
        print(
            f"drift: {STANDARD_FILE} no longer describes the published {standard['id']}:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"drift: {STANDARD_FILE} matches the published document (sha256 {digest})")
    return 0


# ------------------------------------------------------------------ outputs


def go_string(text: str) -> str:
    return json.dumps(text)


def go_strings(values: list[str] | None) -> str:
    if values is None:
        return "nil"
    return "[]string{" + ", ".join(go_string(value) for value in values) + "}"


def compact(schema: Any) -> str:
    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)


def render_go_fields(specs: Specs) -> str:
    standard_id, extension_id = specs.standard["id"], specs.extension["id"]
    lines = [
        f"// Code generated by {GENERATOR} from {SPEC_DIR}; DO NOT EDIT.",
        "",
        "package systemoneprotocol",
        "",
        "// StandardSchemaID is the standard's schema ID.",
        f"const StandardSchemaID = {go_string(standard_id)}",
        "",
        "// ExtensionSchemaID is the vllm-sr extension set's schema ID.",
        f"const ExtensionSchemaID = {go_string(extension_id)}",
        "",
        "// ExtensionSetVersion is the vllm-sr extension set's version.",
        f"const ExtensionSetVersion = {go_string(str(specs.extension['extension_version']))}",
        "",
        "var generatedFamilies = " + go_strings(list(specs.extension["families"])),
        "",
        "var generatedFields = []Field{",
    ]
    for entry in standard_fields(specs.standard) + extension_fields(specs):
        origin = (
            "StandardSchemaID" if entry.origin == standard_id else "ExtensionSchemaID"
        )
        lines.append(
            "\t{"
            f"Level: {GO_LEVELS[entry.level]}, Name: {go_string(entry.name)}, Origin: {origin}, "
            f"Since: {go_string(entry.since)}, Families: {go_strings(entry.families)}, "
            f"AppliesTo: {go_strings(entry.applies_to)}, Endpoints: {go_strings(entry.endpoints)}, "
            f"StandardTarget: {GO_TARGETS[entry.standard_target]}, Schema: {go_string(compact(entry.schema))}, "
            f"Summary: {go_string(entry.summary)}"
            "},"
        )
    lines += ["}", "", "var generatedRemoved = []RemovedField{"]
    for removed in specs.extension["removed"]:
        lines.append(
            "\t{"
            f"Level: {go_string(removed['level'])}, Name: {go_string(removed['name'])}, "
            f"ReplacedBy: {go_string(removed['replaced_by'])}, Note: {go_string(removed.get('note', ''))}"
            "},"
        )
    lines += [
        "}",
        "",
        "// StandardDescription is the standard's description as JSON.",
        f"const StandardDescription = {go_string(compact(specs.standard))}",
        "",
    ]
    return "\n".join(lines)


def render_go_specs(specs: Specs) -> str:
    standard, extension = specs.standard, specs.extension
    entries = [
        (
            standard["id"],
            str(STANDARD_FILE),
            specs.standard_sha256,
            standard["source"],
            standard["source_version"],
        ),
        (
            extension["id"],
            str(EXTENSION_FILE),
            specs.extension_sha256,
            "",
            str(extension["extension_version"]),
        ),
        (
            "vllm-sr/4.0",
            str(LEGACY_FILE),
            specs.legacy_sha256,
            "the model runtime contract 4.0.0",
            "4.0.0",
        ),
    ]
    lines = [
        f"// Code generated by {GENERATOR} from {SPEC_DIR}; DO NOT EDIT.",
        "",
        "package systemonecodec",
        "",
        "var generatedSpecs = []SpecRef{",
    ]
    for sid, path, digest, source, version in entries:
        lines.append(
            "\t{"
            f"ID: {go_string(sid)}, Path: {go_string(path)}, SHA256: {go_string(digest)}, "
            f"Source: {go_string(source)}, Version: {go_string(version)}"
            "},"
        )
    lines += ["}", ""]
    return "\n".join(lines)


def py_tuple(values: list[str], indent: str = "") -> str:
    if not values:
        return "()"
    inner = "".join(f"{indent}    {json.dumps(value)},\n" for value in values)
    return f"(\n{inner}{indent})"


def render_python(specs: Specs) -> str:
    standard = standard_fields(specs.standard)
    extension = extension_fields(specs)
    every = standard + extension

    def names(level: str, fields: list[Field]) -> list[str]:
        return sorted({entry.name for entry in fields if entry.level == level})

    question_types = names("question_type", every)
    question_fields: dict[str, list[str]] = {}
    for question_type in question_types:
        members = (
            {"type", "instructions", "criteria"}
            if question_type not in question_type_mapping(specs.standard)
            else set()
        )
        for entry in every:
            if entry.level == "question" and (
                entry.applies_to is None or question_type in entry.applies_to
            ):
                members.add(entry.name)
        question_fields[question_type] = sorted(members)
    option_fields = sorted(
        (
            next(entry for entry in extension if entry.name == "options").schema or {}
        ).get("properties", {})
    )
    lines = [
        f'"""The System One field tables, generated by {GENERATOR}.',
        "",
        f"Do not edit: change {SPEC_DIR} and run ``make systemone-schemas-generate``.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"STANDARD_SCHEMA = {json.dumps(specs.standard['id'])}",
        f"EXTENSION_SCHEMA = {json.dumps(specs.extension['id'])}",
        f"EXTENSION_VERSION = {json.dumps(str(specs.extension['extension_version']))}",
        "",
        "# The members a request may carry: the standard's and the extension set's.",
        f"REQUEST_FIELDS = {py_tuple(names('request', every))}",
        "",
        "# The members of request options.",
        f"OPTION_FIELDS = {py_tuple(option_fields)}",
        "",
        "# The question types, and the members a question of each type may carry.",
        f"QUESTION_TYPES = {py_tuple(question_types)}",
        "QUESTION_FIELDS = {",
    ]
    for question_type in question_types:
        lines.append(
            f"    {json.dumps(question_type)}: {py_tuple(question_fields[question_type], '    ')},"
        )
    lines += [
        "}",
        "",
        "# The asset types, and the members an asset may carry.",
        f"ASSET_TYPES = {py_tuple(names('asset_type', every))}",
        f"ASSET_FIELDS = {py_tuple(names('asset', every))}",
        "",
        "# The members a response, an answer and an error may carry.",
        f"RESPONSE_FIELDS = {py_tuple(names('response', every))}",
        f"ANSWER_FIELDS = {py_tuple(names('answer', every))}",
        f"ERROR_FIELDS = {py_tuple(names('error', every))}",
        f"LISTING_FIELDS = {py_tuple(names('listing_entry', every))}",
        "",
        "# The 4.0 forms 5.0 no longer accepts, by level and name, and their replacement.",
        "REMOVED = {",
    ]
    for removed in specs.extension["removed"]:
        lines.append(
            f"    ({json.dumps(removed['level'])}, {json.dumps(removed['name'])}): {json.dumps(removed['replaced_by'])},"
        )
    lines += ["}", ""]
    return "\n".join(lines)


def markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def describe_type(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "any"
    if "const" in schema:
        return f"`{json.dumps(schema['const'])}`"
    if "enum" in schema:
        return " or ".join(f"`{value}`" for value in schema["enum"])
    if "anyOf" in schema:
        return " or ".join(describe_type(branch) for branch in schema["anyOf"])
    kind = schema.get("type")
    if isinstance(kind, list):
        return " or ".join(kind)
    if (
        kind == "object"
        and isinstance(schema.get("additionalProperties"), dict)
        and "properties" not in schema
    ):
        return f"object of {describe_type(schema['additionalProperties'])}"
    if kind == "array" and isinstance(schema.get("items"), dict) and schema["items"]:
        return f"array of {describe_type(schema['items'])}"
    return kind or "any"


def families_text(entry: Field) -> str:
    if entry.families is None:
        return "all"
    if not entry.families:
        return "the Router"
    return ", ".join(entry.families)


TARGET_TEXT = {
    "strip": "stripped",
    "refuse": "refused at load",
    "absent": "never sent",
}


def render_reference(specs: Specs) -> str:
    extension = extension_fields(specs)
    level_titles = {
        name: spec["title"] for name, spec in specs.extension["levels"].items()
    }
    lines = [
        "---",
        "title: Extension reference",
        "description: Every field the vllm-sr/5.0 extension set adds to the systemone/0.2 standard, generated from config/schemas/systemone.",
        "---",
        "",
        f"<!-- Generated by {GENERATOR} from {SPEC_DIR}; do not edit. -->",
        "",
        "# Extension reference",
        "",
        f"`{specs.extension['id']}` is `{specs.standard['id']}`, the published System One standard, plus the "
        f"fields below, extension set {specs.extension['extension_version']}. Every vllm-sr message is a "
        "complete standard message: an extension field is an additive hint in its natural place, and a vllm-sr "
        "server answers an unknown field with 400 naming it.",
        "",
        "The last column says what the Router does with the field when the target deployment speaks "
        f"`{specs.standard['id']}`: it strips a field the standard part of the message already expresses, "
        "refuses at load a rule that needs one the standard cannot carry, and never expects a response field.",
        "",
    ]
    for level in LEVELS:
        entries = [entry for entry in extension if entry.level == level]
        if not entries:
            continue
        lines += [f"## {level_titles.get(level, level)}", ""]
        if level == "endpoint":
            lines += [
                "| Endpoint | Since | Families | Description |",
                "| --- | --- | --- | --- |",
            ]
            for entry in entries:
                lines.append(
                    f"| `{entry.name}` | {entry.since} | {families_text(entry)} | {markdown_cell(entry.summary)} |"
                )
        else:
            lines += [
                f"| Field | Type | Since | Families | On {specs.standard['id']} | Description |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for entry in entries:
                scope = ""
                if entry.applies_to:
                    scope = " (" + ", ".join(entry.applies_to) + ")"
                lines.append(
                    f"| `{entry.name}`{markdown_cell(scope)} | {markdown_cell(describe_type(entry.schema))} | {entry.since} | "
                    f"{families_text(entry)} | {TARGET_TEXT[entry.standard_target]} | {markdown_cell(entry.summary)} |"
                )
        lines.append("")
    lines += [
        "## Removed forms",
        "",
        "The model runtime contract 4.0.0 accepted these forms; the standard expresses them, so 5.0 answers them "
        "with 400 naming the replacement.",
        "",
        "| Form | Level | Replaced by | Note |",
        "| --- | --- | --- | --- |",
    ]
    for removed in specs.extension["removed"]:
        lines.append(
            f"| `{removed['name']}` | {removed['level']} | `{markdown_cell(removed['replaced_by'])}` | {markdown_cell(removed.get('note', ''))} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_matrix(specs: Specs) -> str:
    extension = extension_fields(specs)
    families = list(specs.extension["families"])
    standard_types = list(question_type_mapping(specs.standard))
    lines = [
        "---",
        "title: Compatibility matrix",
        "description: Which schema carries each System One feature, and which model families honour it, generated from config/schemas/systemone.",
        "---",
        "",
        f"<!-- Generated by {GENERATOR} from {SPEC_DIR}; do not edit. -->",
        "",
        "# Compatibility matrix",
        "",
        f"Each row is a feature: a question type of the standard, or a field of the extension set. `{specs.standard['id']}` "
        f"is the published standard and `{specs.extension['id']}` the standard plus the vllm-sr extension set. A family "
        "column says whether the model runtime's family honours the feature; a dash means it does not.",
        "",
        "| Feature | "
        + " | ".join([specs.standard["id"], specs.extension["id"], *families])
        + " |",
        "| --- | " + " | ".join(["---"] * (2 + len(families))) + " |",
    ]
    for question_type in standard_types:
        lines.append(
            f"| `{question_type}` question | yes | yes | "
            + " | ".join(["yes"] * len(families))
            + " |"
        )
    for entry in extension:
        if entry.level in ("answer", "response", "error", "listing_entry", "endpoint"):
            continue
        label = f"`{entry.name}` {'question' if entry.level == 'question_type' else entry.level.replace('_', ' ')}"
        cells = [TARGET_TEXT[entry.standard_target], "yes"]
        cells += [
            "yes" if entry.families is None or family in entry.families else "—"
            for family in families
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


GOLDEN_CASES = [
    ("text", '"Our export button fails with a 500 since this morning."'),
    ("chinese", '"导出按钮从今天早上开始一直报 500 错误。"'),
    ("emoji", '"Thanks 🙂👍🏽 — great!"'),
    ("escaped_emoji", '"\\ud83d\\ude00 and \\u00e9"'),
    ("newlines_tabs", '"line one\\nline two\\r\\n\\tindented"'),
    ("quotes_backslashes", '"she said \\"hi\\" and C:\\\\path\\\\file"'),
    ("html", '"<script>alert(1)</script> & co"'),
    ("line_separators", '"a\\u2028b\\u2029c"'),
    ("control_characters", '"\\u0000\\u0001\\u0007\\b\\f\\u000b\\u001f\\u007f"'),
    (
        "router_parts",
        '{"history": "How do I export my invoices?", "request": "Our export fails with a 500 again."}',
    ),
    (
        "previous_turns",
        '{"previous_turns": "user: Hi\\nassistant: Hello!", "request": "Same question again"}',
    ),
    ("key_order", '{"zeta": 1, "alpha": 2, "Mid": 3, "beta": [3, 2, 1]}'),
    ("duplicate_keys", '{"a": 1, "b": 2, "a": 3}'),
    (
        "nested",
        '{"ticket": {"subject": "Duplicate charge", "messages": [{"from": "customer", "text": "Charged twice"}, {"from": "support", "text": "Checking"}]}, "tags": ["billing", ["nested", {"deep": null}]], "open": true, "escalated": false}',
    ),
    ("empty", '{"object": {}, "array": [], "text": ""}'),
    ("integers", "[0, -0, 1, -1, 42, 12345678901234567890123456789]"),
    (
        "floats",
        "[0.0, -0.0, 1.0, 1.5, 0.1, 100.25, 2.5e-5, 1e2, 1E2, 1e15, 1e16, 1.5e16]",
    ),
    ("small_floats", "[0.001, 0.0001, 0.00001, 1e-7, 5e-324]"),
    (
        "large_floats",
        "[1.7976931348623157e308, 123456789012345678.0, 9007199254740993.0]",
    ),
    ("out_of_range", "[1e400, -1e400, 1e-400, -1e-400]"),
    (
        "array_state",
        '["Hi", "My customer number is TS1337.", "My card was charged twice."]',
    ),
]


def render_golden() -> str:
    cases = []
    for name, text in GOLDEN_CASES:
        value = json.loads(text)
        cases.append(
            {
                "name": name,
                "input": text,
                "canonical": json.dumps(value, ensure_ascii=False),
            }
        )
    return (
        json.dumps(
            {
                "generator": GENERATOR,
                "python": "json.dumps(json.loads(input), ensure_ascii=False)",
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


# Requests in the internal form, the IR's encoding on /v1/bundle and
# /v1/decisions, beside the families' golden requests: each 4.0 spelling,
# a batch with shared assets, the roles forms and number literals. The
# internal codec must write every one back unchanged.
INTERNAL_CASES = [
    {
        "model": "vllm-sr/vela3-fast",
        "state": {
            "request": "The export fails with a 500 again.",
            "history": "How do I export invoices?",
        },
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "choices": [
                    {"key": "billing", "description": "Payments"},
                    {"key": "eng", "description": "Bugs"},
                ],
            },
            "jailbreak": {"type": "choice", "preset": "jailbreak"},
            "pii": {
                "type": "span",
                "labels": [{"key": "EMAIL_ADDRESS", "description": "An email address"}],
                "threshold": 0.5,
            },
        },
        "images": [],
        "options": {"deadline_ms": 1500, "return_meta": True, "long_context": False},
    },
    {
        "states": {
            "1": {
                "state": "Our export fails with a 500 again.",
                "questions": {"q": {"type": "noul", "instructions": "Is it a bug?"}},
            },
            "2": {
                "state": {
                    "history": "How do I export my invoices?",
                    "request": "Again ⟦img⟧",
                },
                "questions": {
                    "s": {
                        "type": "score",
                        "levels": ["low", "high"],
                        "require_full_input": True,
                    }
                },
                "roles": {"history": "context", "request": "user"},
            },
        },
        "model": "d3",
        "options": {"deadline_ms": 949.873},
        "assets": {
            "⟦img⟧": {"type": "image", "format": "png", "data_base64": "iVBORw0KGgo="}
        },
    },
    {
        "state": {
            "context": "The refund policy is 30 days.",
            "answer": "You have 90 days.",
        },
        "questions": {"halu": {"preset": "halu", "overflow": "window"}},
        "roles": None,
    },
    {
        "questions": {
            "track": {"type": "track", "instructions": "Track the red car.", "rate": 2},
            "box": {"type": "box", "instructions": "The car", "at": 1.50},
        },
        "state": "Where is the car?",
        "videos": ["data:video/mp4;base64,AAAAIGZ0eXBpc29t"],
        "options": {"max_tokens": 4096},
    },
    {
        "model": "vllm-sr/vela2",
        "state": [
            "导出失败",
            {"code": 500, "retry": None, "ratio": 1e-7},
            "emoji 😀 \u2028 <&>",
        ],
        "questions": {
            "set": {
                "type": "set",
                "instructions": "Which apply?",
                "criteria": {"bug": None, "billing": "Payments"},
                "over": ["request"],
            },
            "adapter": {
                "type": "noul",
                "instructions": None,
                "criteria": {"true": "yes", "false": "no"},
                "decision_adapter": True,
                "head": "router",
            },
        },
    },
]


class _Literal(ast.NodeTransformer):
    """Reads a function's parameter as a placeholder text, so its returned literal evaluates."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.names:
            return ast.copy_location(ast.Constant(f"<{node.id}>"), node)
        return node


def shared_golden(root: Path) -> dict[str, Any] | None:
    """vllm_srun/systemone.py's readiness request, GOLDEN_STATE and golden_questions(other), which the Decision families ask."""
    tree = ast.parse(
        (root / FAMILIES_DIR.parent / "systemone.py").read_text(encoding="utf-8")
    )
    state, questions = None, None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", "") == "GOLDEN_STATE"
        ):
            state = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "golden_questions":
            returned = next(item for item in node.body if isinstance(item, ast.Return))
            names = {argument.arg for argument in node.args.args}
            questions = ast.literal_eval(_Literal(names).visit(returned.value))
    if state is None or questions is None:
        return None
    return {"name": "systemone", "body": {"state": state, "questions": questions}}


def family_goldens(root: Path) -> list[dict[str, Any]]:
    """The families' golden requests: the shared readiness request and each family.py's GOLDEN_STATE and GOLDEN_QUESTIONS literals."""
    shared = shared_golden(root)
    cases = [shared] if shared else []
    for path in sorted((root / FAMILIES_DIR).glob("*/family.py")):
        literals: dict[str, Any] = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name = node.targets[0].id
                if name in ("GOLDEN_STATE", "GOLDEN_QUESTIONS"):
                    with contextlib.suppress(ValueError):
                        literals[name] = ast.literal_eval(node.value)
        if "GOLDEN_STATE" in literals and "GOLDEN_QUESTIONS" in literals:
            cases.append(
                {
                    "name": path.parent.name,
                    "body": {
                        "state": literals["GOLDEN_STATE"],
                        "questions": literals["GOLDEN_QUESTIONS"],
                    },
                }
            )
    return cases


def render_internal_corpus(root: Path) -> str:
    cases = family_goldens(root) + [
        {"name": f"form_{index}", "body": body}
        for index, body in enumerate(INTERNAL_CASES, 1)
    ]
    corpus = [
        {
            "name": case["name"],
            "body": json.dumps(case["body"], ensure_ascii=False, separators=(",", ":")),
        }
        for case in cases
    ]
    return (
        json.dumps(
            {"generator": GENERATOR, "cases": corpus}, ensure_ascii=False, indent=2
        )
        + "\n"
    )


def outputs(specs: Specs, root: Path = REPO_ROOT) -> dict[Path, str]:
    return {
        GO_FIELDS: render_go_fields(specs),
        GO_SPECS: render_go_specs(specs),
        PY_FIELDS: render_python(specs),
        GOLDEN: render_golden(),
        INTERNAL_CORPUS: render_internal_corpus(root),
        DOC_REFERENCE: render_reference(specs),
        DOC_MATRIX: render_matrix(specs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail on a stale output or a field both specs define differently",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="compare the published standard with our description",
    )
    parser.add_argument(
        "--spec-file",
        help="the published standard to compare, instead of downloading it",
    )
    parser.add_argument("--url", help="where the published standard is downloaded from")
    args = parser.parse_args(argv)
    root = Path(args.repository_root)
    if args.drift:
        return run_drift(root, args.spec_file, args.url)
    try:
        specs = load_specs(root)
    except SpecError as exc:
        print(f"systemone schemas: {exc}", file=sys.stderr)
        return 1
    problems = collisions(specs)
    if problems:
        print(
            "systemone schemas: the standard and the extension set collide:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    stale = []
    for path, content in outputs(specs, root).items():
        target = root / path
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                stale.append(str(path))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    if stale:
        print(
            "systemone schemas: stale outputs; run make systemone-schemas-generate:",
            file=sys.stderr,
        )
        for path in stale:
            print(f"  - {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
