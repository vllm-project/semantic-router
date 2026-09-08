#!/usr/bin/env python3
"""Compile the authored model catalog into every committed product projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

CATALOG_TOOL_ROOT = Path(__file__).resolve().parent
if str(CATALOG_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(CATALOG_TOOL_ROOT))

from catalog_common import (  # noqa: E402
    MODEL_ID,
    PROTOCOL_ID,
    SLUG,
    CatalogBuildError,
)
from catalog_common import mapping as _mapping  # noqa: E402
from catalog_common import nonempty_string as _nonempty_string  # noqa: E402
from catalog_common import reject_unknown as _reject_unknown  # noqa: E402
from catalog_common import sequence as _sequence  # noqa: E402
from catalog_common import validate_https_url as _validate_https_url  # noqa: E402
from catalog_evaluations import (  # noqa: E402
    evaluation_coverage as _evaluation_coverage,
)
from catalog_evaluations import index_results as _index_results  # noqa: E402
from catalog_evaluations import metric_catalog as _metric_catalog  # noqa: E402
from catalog_evaluations import (  # noqa: E402, F401 - tested compatibility seam
    normalize_component as _normalize_component,
)
from catalog_evaluations import validate_indices as _validate_indices  # noqa: E402
from catalog_inventory import (  # noqa: E402
    validate_evaluation_resource_layout as _validate_evaluation_resource_layout_impl,
)
from catalog_inventory import (  # noqa: E402
    validate_inventory_policy as _validate_inventory_policy,
)
from catalog_inventory import (  # noqa: E402
    validate_model_resource_layout as _validate_model_resource_layout_impl,
)
from catalog_io import load_json as _load_json  # noqa: E402
from catalog_io import load_yaml as _load_yaml  # noqa: E402
from catalog_io import validate_schema as _validate_schema  # noqa: E402
from catalog_provider_definitions import (  # noqa: E402
    validate_provider_presentation as _validate_provider_presentation,
)
from catalog_provider_definitions import (  # noqa: E402
    validate_providers as _validate_providers,
)
from catalog_validation import (  # noqa: E402
    validate_evaluations as _validate_evaluations,
)
from catalog_validation import (  # noqa: E402
    validate_provider_bindings as _validate_provider_bindings,
)
from catalog_validation import validate_security as _validate_security  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "config" / "catalog"
SOURCE_MANIFEST = SOURCE_ROOT / "catalog.yaml"
SOURCE_SCHEMA_PATH = SOURCE_ROOT / "schemas" / "catalog-source-v1.schema.json"
RESOURCE_SCHEMA_PATH = SOURCE_ROOT / "schemas" / "catalog-resources-v1.schema.json"
SNAPSHOT_SCHEMA_PATH = SOURCE_ROOT / "schemas" / "catalog-snapshot-v2.schema.json"
RECIPE_MANIFEST = (
    REPO_ROOT / "config" / "recipes" / "built-in" / "latest" / "catalog.yaml"
)
CLI_MANIFEST = (
    REPO_ROOT / "src" / "vllm-sr" / "cli" / "model_assets" / "latest" / "catalog.yaml"
)
GO_OUTPUT = (
    REPO_ROOT
    / "src"
    / "semantic-router"
    / "pkg"
    / "catalog"
    / "zz_generated_catalog.go"
)
DASHBOARD_OUTPUT = (
    REPO_ROOT / "dashboard" / "frontend" / "src" / "generated" / "modelCatalog.json"
)
WEBSITE_OUTPUT = REPO_ROOT / "website" / "static" / "model-catalog" / "catalog.json"

CLI_ROOT = REPO_ROOT / "src" / "vllm-sr"
if str(CLI_ROOT) not in sys.path:
    sys.path.insert(0, str(CLI_ROOT))

from cli.model_bundle import model_bundle_digest  # noqa: E402

SOURCE_SCHEMA = "vllm-sr/catalog-source/v1"
OUTPUT_SCHEMA = "vllm-sr/model-catalog/v2"
RESOURCE_KEYS = (
    "protocols",
    "providers",
    "reasoning_families",
    "models",
    "benchmarks",
    "evaluations",
    "indices",
)


def _resource_documents(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    resources = _mapping(manifest.get("resources"), "resources")
    _reject_unknown(resources, RESOURCE_KEYS, "resources")
    result: dict[str, list[dict[str, Any]]] = {}
    for kind in RESOURCE_KEYS:
        relative = _nonempty_string(resources.get(kind), f"resources.{kind}")
        target = (SOURCE_ROOT / relative).resolve()
        if SOURCE_ROOT.resolve() not in target.parents:
            raise CatalogBuildError(f"resources.{kind} escapes config/catalog")
        paths = sorted(target.rglob("*.yaml")) if target.is_dir() else [target]
        if not paths or any(not path.is_file() for path in paths):
            raise CatalogBuildError(
                f"resources.{kind} does not resolve to YAML resources"
            )
        items: list[dict[str, Any]] = []
        for path in paths:
            raw = _load_yaml(path)
            documents = raw if isinstance(raw, list) else [raw]
            for index, item in enumerate(documents):
                if not isinstance(item, dict):
                    location = f"{path.relative_to(REPO_ROOT)}[{index}]"
                    raise CatalogBuildError(f"{location} must be a mapping")
                items.append(item)
        result[kind] = items
    return result


def _validate_unique_ids(resources: dict[str, list[dict[str, Any]]]) -> None:
    for kind, items in resources.items():
        seen: set[str] = set()
        for index, item in enumerate(items):
            identity = _nonempty_string(item.get("id"), f"{kind}[{index}].id")
            if identity in seen:
                raise CatalogBuildError(f"duplicate {kind} id: {identity}")
            seen.add(identity)


def _validate_protocols(items: list[dict[str, Any]]) -> None:
    wire_formats: set[str] = set()
    for index, item in enumerate(items):
        path = f"protocols[{index}]"
        _reject_unknown(
            item,
            {
                "id",
                "display_name",
                "wire_format",
                "default_base_path",
                "operations",
                "capabilities",
            },
            path,
        )
        identity = _nonempty_string(item.get("id"), f"{path}.id")
        if not PROTOCOL_ID.fullmatch(identity):
            raise CatalogBuildError(
                f"{path}.id must be a namespaced major-version identity"
            )
        wire = _nonempty_string(item.get("wire_format"), f"{path}.wire_format")
        if wire in wire_formats:
            raise CatalogBuildError(f"duplicate protocol wire_format: {wire}")
        wire_formats.add(wire)
        default_base_path = _nonempty_string(
            item.get("default_base_path"), f"{path}.default_base_path"
        ).rstrip("/")
        if not default_base_path.startswith("/"):
            raise CatalogBuildError(f"{path}.default_base_path must be absolute")
        operations = _sequence(item.get("operations"), f"{path}.operations")
        if not operations:
            raise CatalogBuildError(f"{path}.operations cannot be empty")
        operation_ids: set[str] = set()
        for operation_index, raw_operation in enumerate(operations):
            operation = _mapping(raw_operation, f"{path}.operations[{operation_index}]")
            _reject_unknown(
                operation,
                {"id", "method", "path"},
                f"{path}.operations[{operation_index}]",
            )
            operation_id = _nonempty_string(
                operation.get("id"), f"{path}.operations[{operation_index}].id"
            )
            if operation_id in operation_ids:
                raise CatalogBuildError(f"{path} duplicates operation {operation_id}")
            operation_ids.add(operation_id)
            if operation.get("method") not in {"GET", "POST", "DELETE"}:
                raise CatalogBuildError(
                    f"{path}.operations[{operation_index}].method is unsupported"
                )
            operation_path = _nonempty_string(
                operation.get("path"), f"{path}.operations[{operation_index}].path"
            )
            if not operation_path.startswith("/"):
                raise CatalogBuildError(
                    f"{path}.operations[{operation_index}].path must be absolute"
                )
            if default_base_path and not (
                operation_path == default_base_path
                or operation_path.startswith(default_base_path + "/")
            ):
                raise CatalogBuildError(
                    f"{path}.operations[{operation_index}].path must start with "
                    f"default_base_path {default_base_path}"
                )


def _validate_reasoning_levels(item: dict[str, Any], path: str) -> list[str]:
    raw_levels = item.get("levels")
    values = _sequence(raw_levels, f"{path}.levels") if raw_levels is not None else []
    levels = [
        _nonempty_string(value, f"{path}.levels[{index}]")
        for index, value in enumerate(values)
    ]
    if len(levels) != len(set(levels)):
        raise CatalogBuildError(f"{path}.levels contains duplicates")
    if not levels and item.get("default") is not None:
        raise CatalogBuildError(f"{path}.levels is required with default")
    if (
        item.get("type")
        in {
            "reasoning_effort",
            "top_level_reasoning_effort",
        }
        and not levels
    ):
        raise CatalogBuildError(f"{path}.levels is required for effort families")
    if item.get("default") is not None and item.get("default") not in levels:
        raise CatalogBuildError(f"{path}.default must be listed in levels")
    return levels


def _validate_reasoning_modes(item: dict[str, Any], path: str) -> None:
    modes = _sequence(item.get("modes"), f"{path}.modes")
    default_mode = item.get("default_mode")
    if (
        not modes
        or len(modes) != len(set(modes))
        or any(mode not in {"enabled", "disabled", "adaptive"} for mode in modes)
        or default_mode not in modes
    ):
        raise CatalogBuildError(f"{path}.modes/default_mode is invalid")
    if item.get("disabled") is not None and "disabled" not in modes:
        raise CatalogBuildError(
            f"{path}.disabled requires disabled to be listed in modes"
        )


def _validate_reasoning_activation(item: dict[str, Any], path: str) -> str | None:
    raw_parameter = item.get("activation_parameter")
    if raw_parameter is None:
        return None
    parameter = _nonempty_string(raw_parameter, f"{path}.activation_parameter")
    if parameter == item.get("parameter"):
        raise CatalogBuildError(
            f"{path}.activation_parameter must differ from parameter"
        )
    if item.get("type") != "reasoning_effort":
        raise CatalogBuildError(
            f"{path}.activation_parameter requires reasoning_effort type"
        )
    return parameter


def _validate_reasoning_effort_flags(
    item: dict[str, Any],
    path: str,
    levels: list[str],
    activation_parameter: str | None,
) -> None:
    raw_flags = item.get("effort_flags")
    if raw_flags is None:
        return
    effort_flags = _mapping(raw_flags, f"{path}.effort_flags")
    if item.get("type") != "reasoning_effort":
        raise CatalogBuildError(f"{path}.effort_flags requires reasoning_effort type")
    if not activation_parameter:
        raise CatalogBuildError(f"{path}.effort_flags requires activation_parameter")
    seen_parameters: set[str] = set()
    for effort, raw_parameter in effort_flags.items():
        if effort not in levels:
            raise CatalogBuildError(
                f"{path}.effort_flags key {effort!r} must be listed in levels"
            )
        parameter = _nonempty_string(raw_parameter, f"{path}.effort_flags.{effort}")
        if parameter in {item.get("parameter"), activation_parameter}:
            raise CatalogBuildError(
                f"{path}.effort_flags.{effort} must differ from parameter "
                "and activation_parameter"
            )
        if parameter in seen_parameters:
            raise CatalogBuildError(
                f"{path}.effort_flags contains duplicate parameter {parameter!r}"
            )
        seen_parameters.add(parameter)
    active_levels = [level for level in levels if level != item.get("disabled")]
    if len(active_levels) - len(effort_flags) > 1:
        raise CatalogBuildError(
            f"{path}.effort_flags leaves multiple levels indistinguishable by omission"
        )


def _validate_reasoning(items: list[dict[str, Any]]) -> None:
    allowed_fields = {
        "id",
        "type",
        "parameter",
        "activation_parameter",
        "effort_flags",
        "levels",
        "default",
        "modes",
        "default_mode",
        "disabled",
    }
    supported_types = {
        "chat_template_kwargs",
        "reasoning_effort",
        "reasoning_mode",
        "top_level_reasoning_effort",
    }
    for index, item in enumerate(items):
        path = f"reasoning_families[{index}]"
        _reject_unknown(item, allowed_fields, path)
        if not SLUG.fullmatch(_nonempty_string(item.get("id"), f"{path}.id")):
            raise CatalogBuildError(f"{path}.id must be a lowercase slug")
        if item.get("type") not in supported_types:
            raise CatalogBuildError(f"{path}.type is unsupported")
        _nonempty_string(item.get("parameter"), f"{path}.parameter")
        levels = _validate_reasoning_levels(item, path)
        _validate_reasoning_modes(item, path)
        activation_parameter = _validate_reasoning_activation(item, path)
        _validate_reasoning_effort_flags(item, path, levels, activation_parameter)


def _validate_models(
    items: list[dict[str, Any]],
    asset_ids: set[str],
    reasoning_ids: set[str],
) -> None:
    for index, item in enumerate(items):
        path = f"models[{index}]"
        kind = _validate_model_identity(item, path, reasoning_ids)
        _validate_model_verification(item, path)
        if kind == "virtual":
            _validate_virtual_model(item, path, asset_ids)
        else:
            _validate_physical_model(item, path)


_MODEL_FIELDS = {
    "id",
    "display_name",
    "description",
    "kind",
    "publisher",
    "presentation",
    "distribution",
    "family",
    "generation",
    "parameter_size",
    "policy_version",
    "revision",
    "released_at",
    "knowledge_cutoff",
    "lifecycle",
    "limits",
    "capabilities",
    "modalities",
    "reasoning_family",
    "asset",
    "entrypoint",
    "recipe",
    "traits",
    "roles",
    "verification",
    "compatibility",
    "tags",
}


def _validate_model_identity(
    item: dict[str, Any],
    path: str,
    reasoning_ids: set[str],
) -> str:
    _reject_unknown(item, _MODEL_FIELDS, path)
    identity = _nonempty_string(item.get("id"), f"{path}.id")
    if not MODEL_ID.fullmatch(identity):
        raise CatalogBuildError(f"{path}.id must be a namespaced model identity")
    kind = item.get("kind")
    if kind not in {"physical", "virtual"}:
        raise CatalogBuildError(f"{path}.kind is unsupported")
    _nonempty_string(item.get("publisher"), f"{path}.publisher")
    _validate_provider_presentation(item, path)
    distribution = _mapping(item.get("distribution"), f"{path}.distribution")
    _reject_unknown(distribution, {"type", "source", "license"}, f"{path}.distribution")
    _validate_https_url(distribution.get("source"), f"{path}.distribution.source")
    if distribution.get("type") not in {
        "proprietary_api",
        "open_weights",
        "router_recipe",
    }:
        raise CatalogBuildError(f"{path}.distribution.type is unsupported")
    if distribution.get("type") == "open_weights" and not distribution.get("license"):
        raise CatalogBuildError(f"{path}.distribution.license is required")
    family = item.get("reasoning_family")
    if family and family not in reasoning_ids:
        raise CatalogBuildError(f"{path}.reasoning_family references an unknown family")
    return str(kind)


def _validate_model_verification(item: dict[str, Any], path: str) -> None:
    verification = _mapping(item.get("verification"), f"{path}.verification")
    _reject_unknown(
        verification,
        {"authority", "status", "verified_at", "source"},
        f"{path}.verification",
    )
    if verification.get("status") not in {"claimed", "imported", "reproduced"}:
        raise CatalogBuildError(f"{path}.verification.status is unsupported")
    if verification.get("source") is not None:
        _validate_https_url(verification["source"], f"{path}.verification.source")


def _validate_virtual_model(
    item: dict[str, Any],
    path: str,
    asset_ids: set[str],
) -> None:
    if item.get("asset") not in asset_ids:
        raise CatalogBuildError(f"{path}.asset references an unknown asset")
    for required in ("entrypoint", "recipe", "roles", "generation", "policy_version"):
        if item.get(required) in (None, "", []):
            raise CatalogBuildError(f"{path}.{required} is required for virtual models")
    if item["distribution"]["type"] != "router_recipe":
        raise CatalogBuildError(f"{path}.distribution.type must be router_recipe")
    for role_index, raw_role in enumerate(
        _sequence(item.get("roles"), f"{path}.roles")
    ):
        _validate_virtual_model_role(raw_role, f"{path}.roles[{role_index}]")


def _validate_virtual_model_role(raw_role: Any, path: str) -> None:
    role = _mapping(raw_role, path)
    _reject_unknown(
        role,
        {"name", "required", "minimum_candidates", "traits", "recommended_pool"},
        path,
    )
    pool = _sequence(role.get("recommended_pool"), f"{path}.recommended_pool")
    minimum = role.get("minimum_candidates")
    if not isinstance(minimum, int) or minimum < 1 or minimum > len(pool):
        raise CatalogBuildError(f"{path}.minimum_candidates is invalid")


def _validate_physical_model(item: dict[str, Any], path: str) -> None:
    virtual_fields = ("asset", "entrypoint", "recipe", "roles")
    if any(item.get(field_name) is not None for field_name in virtual_fields):
        raise CatalogBuildError(f"{path} physical model contains virtual-only fields")
    if item["distribution"]["type"] == "router_recipe":
        raise CatalogBuildError(f"{path}.distribution.type is virtual-only")
    if not item["verification"].get("source"):
        raise CatalogBuildError(f"{path}.verification.source is required")
    if "chat" in item.get("capabilities", []):
        limits = _mapping(item.get("limits"), f"{path}.limits")
        context_window = limits.get("context_window_size")
        if not isinstance(context_window, int) or context_window < 1:
            raise CatalogBuildError(
                f"{path}.limits.context_window_size is required for chat models"
            )
    if item["distribution"]["type"] == "open_weights" and not item.get(
        "parameter_size"
    ):
        raise CatalogBuildError(
            f"{path}.parameter_size is required for open-weight models"
        )


def load_and_validate() -> (
    tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, str]]]
):
    manifest = _mapping(_load_yaml(SOURCE_MANIFEST), "catalog")
    _validate_schema(manifest, _load_json(SOURCE_SCHEMA_PATH), "catalog")
    _reject_unknown(
        manifest,
        {
            "schema_version",
            "catalog_version",
            "channel",
            "release",
            "compatibility",
            "defaults",
            "assets",
            "inventory",
            "resources",
        },
        "catalog",
    )
    if manifest.get("schema_version") != SOURCE_SCHEMA:
        raise CatalogBuildError(f"schema_version must be {SOURCE_SCHEMA}")
    _validate_model_resource_layout_impl(manifest, SOURCE_ROOT, REPO_ROOT)
    _validate_evaluation_resource_layout_impl(manifest, SOURCE_ROOT, REPO_ROOT)
    resources = _resource_documents(manifest)
    resource_schema = _load_json(RESOURCE_SCHEMA_PATH)
    for kind, items in resources.items():
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": resource_schema["$defs"],
            "$ref": f"#/$defs/{kind}",
        }
        _validate_schema(items, schema, f"resources.{kind}")
    _validate_security({"manifest": manifest, "resources": resources})
    _validate_unique_ids(resources)

    assets: list[dict[str, str]] = []
    asset_ids: set[str] = set()
    for index, raw_asset in enumerate(_sequence(manifest.get("assets"), "assets")):
        asset = _mapping(raw_asset, f"assets[{index}]")
        _reject_unknown(asset, {"id", "bundle"}, f"assets[{index}]")
        identity = _nonempty_string(asset.get("id"), f"assets[{index}].id")
        if identity in asset_ids or not SLUG.fullmatch(identity):
            raise CatalogBuildError(f"assets[{index}].id is invalid or duplicated")
        bundle_path = (
            SOURCE_ROOT
            / _nonempty_string(asset.get("bundle"), f"assets[{index}].bundle")
        ).resolve()
        expected_root = (
            REPO_ROOT / "config" / "recipes" / "built-in" / "latest"
        ).resolve()
        if bundle_path.parent != expected_root or not bundle_path.is_dir():
            raise CatalogBuildError(
                f"assets[{index}].bundle must name a latest built-in recipe bundle"
            )
        assets.append(
            {
                "id": identity,
                "bundle": bundle_path.name,
                "sha256": model_bundle_digest(bundle_path),
            }
        )
        asset_ids.add(identity)

    _validate_protocols(resources["protocols"])
    protocol_ids = {item["id"] for item in resources["protocols"]}
    _validate_providers(resources["providers"], resources["protocols"])
    _validate_reasoning(resources["reasoning_families"])
    reasoning_ids = {item["id"] for item in resources["reasoning_families"]}
    _validate_models(resources["models"], asset_ids, reasoning_ids)
    _validate_inventory_policy(manifest, resources["models"])
    model_ids = {item["id"] for item in resources["models"]}
    providers = {item["id"]: item for item in resources["providers"]}
    models = {item["id"]: item for item in resources["models"]}
    reasoning = {item["id"]: item for item in resources["reasoning_families"]}
    _validate_provider_bindings(providers, models, protocol_ids, reasoning)
    metrics = _metric_catalog(resources["benchmarks"])
    _validate_indices(resources["indices"], metrics)
    _validate_evaluations(resources["evaluations"], models, reasoning, metrics)

    defaults = _mapping(manifest.get("defaults"), "defaults")
    _reject_unknown(defaults, {"model", "enabled", "intelligence_index"}, "defaults")
    if defaults.get("model") not in model_ids:
        raise CatalogBuildError("defaults.model references an unknown model")
    enabled = _sequence(defaults.get("enabled"), "defaults.enabled")
    if defaults["model"] not in enabled or any(
        model not in model_ids for model in enabled
    ):
        raise CatalogBuildError(
            "defaults.enabled references an unknown model or omits defaults.model"
        )
    if defaults.get("intelligence_index") not in {
        item["id"] for item in resources["indices"]
    }:
        raise CatalogBuildError(
            "defaults.intelligence_index references an unknown index"
        )
    return manifest, resources, assets


def _generated_models(
    resources: dict[str, list[dict[str, Any]]], assets: list[dict[str, str]]
) -> list[dict[str, Any]]:
    digest_by_asset = {asset["id"]: asset["sha256"] for asset in assets}
    generated: list[dict[str, Any]] = []
    for source in resources["models"]:
        model = json.loads(json.dumps(source))
        if model.get("kind") == "virtual":
            model["verification"]["asset_sha256"] = digest_by_asset[model["asset"]]
        generated.append(model)
    return generated


def render_outputs() -> dict[Path, bytes]:
    manifest, resources, assets = load_and_validate()
    models = _generated_models(resources, assets)
    results = _index_results(resources)
    coverage = _evaluation_coverage(
        resources, manifest["defaults"]["intelligence_index"], results
    )
    generated_manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "catalog_version": manifest["catalog_version"],
        "channel": manifest["channel"],
        "release": manifest["release"],
        "compatibility": manifest["compatibility"],
        "defaults": manifest["defaults"],
        "assets": assets,
        "protocols": resources["protocols"],
        "providers": resources["providers"],
        "reasoning_families": resources["reasoning_families"],
        "models": models,
        "benchmarks": resources["benchmarks"],
        "evaluations": resources["evaluations"],
        "evaluation_coverage": coverage,
        "indices": resources["indices"],
        "index_results": results,
    }
    public = {
        "schema_version": OUTPUT_SCHEMA,
        "catalogs": [
            {
                "catalog_version": manifest["catalog_version"],
                "channel": manifest["channel"],
                "default_model": manifest["defaults"]["model"],
                "enabled_models": manifest["defaults"]["enabled"],
                "default_intelligence_index": manifest["defaults"][
                    "intelligence_index"
                ],
            }
        ],
        "protocols": resources["protocols"],
        "providers": resources["providers"],
        "reasoning_families": resources["reasoning_families"],
        "models": models,
        "benchmarks": resources["benchmarks"],
        "evaluations": resources["evaluations"],
        "evaluation_coverage": coverage,
        "indices": resources["indices"],
        "index_results": results,
    }
    _validate_schema(public, _load_json(SNAPSHOT_SCHEMA_PATH), "generated snapshot")
    public_json = (
        json.dumps(public, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    digest = "sha256:" + hashlib.sha256(public_json.encode("utf-8")).hexdigest()
    go_source = (
        "// Code generated by tools/catalog/generate_model_catalog.py; DO NOT EDIT.\n\n"
        "package catalog\n\n"
        f'const builtInCatalogDigest = "{digest}"\n\n'
        "const builtInCatalogJSON = `" + public_json.rstrip() + "`\n"
    )
    manifest_bytes = yaml.safe_dump(
        generated_manifest, sort_keys=False, width=120, allow_unicode=True
    ).encode("utf-8")
    return {
        RECIPE_MANIFEST: manifest_bytes,
        CLI_MANIFEST: manifest_bytes,
        GO_OUTPUT: go_source.encode("utf-8"),
        DASHBOARD_OUTPUT: public_json.encode("utf-8"),
        WEBSITE_OUTPUT: public_json.encode("utf-8"),
    }


def _relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def check(outputs: dict[Path, bytes]) -> int:
    errors: list[str] = []
    for path, expected in outputs.items():
        if not path.is_file():
            errors.append(f"missing generated catalog artifact: {_relative(path)}")
        elif path.read_bytes() != expected:
            errors.append(f"stale generated catalog artifact: {_relative(path)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


def write(outputs: dict[Path, bytes]) -> int:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as staged:
            staged.write(content)
            staged_path = Path(staged.name)
        staged_path.replace(path)
    return check(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="reject stale generated projections"
    )
    args = parser.parse_args()
    try:
        outputs = render_outputs()
    except CatalogBuildError as error:
        print(f"model catalog invalid: {error}", file=sys.stderr)
        return 1
    return check(outputs) if args.check else write(outputs)


if __name__ == "__main__":
    raise SystemExit(main())
