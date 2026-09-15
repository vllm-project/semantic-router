"""Repository-owned curation and layout rules for the built-in model inventory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from catalog_common import CatalogBuildError
from catalog_common import mapping as _mapping
from catalog_common import nonempty_string as _nonempty_string
from catalog_common import reject_unknown as _reject_unknown
from catalog_common import sequence as _sequence
from catalog_io import load_yaml as _load_yaml


def _resource_directory(
    manifest: dict[str, Any], source_root: Path, resource: str
) -> Path:
    resources = _mapping(manifest.get("resources"), "resources")
    relative = _nonempty_string(resources.get(resource), f"resources.{resource}")
    resolved_source_root = source_root.resolve()
    resource_root = (source_root / relative).resolve()
    if resolved_source_root not in resource_root.parents:
        raise CatalogBuildError(f"resources.{resource} escapes config/catalog")
    return resource_root


def validate_model_resource_layout(
    manifest: dict[str, Any], source_root: Path, repo_root: Path
) -> None:
    model_root = _resource_directory(manifest, source_root, "models")
    if not model_root.is_dir():
        raise CatalogBuildError(
            "resources.models must be a directory partitioned into single and virtual"
        )
    expected_kinds = {"single": "physical", "virtual": "virtual"}
    physical_creator_files: dict[str, Path] = {}
    for partition, expected_kind in expected_kinds.items():
        partition_root = model_root / partition
        if not partition_root.is_dir():
            raise CatalogBuildError(f"resources.models/{partition} is required")
        paths = sorted(partition_root.glob("*.yaml"))
        if not paths:
            raise CatalogBuildError(
                f"resources.models/{partition} must contain model resources"
            )
        for path in paths:
            raw = _load_yaml(path)
            documents = raw if isinstance(raw, list) else [raw]
            if any(
                not isinstance(item, dict) or item.get("kind") != expected_kind
                for item in documents
            ):
                raise CatalogBuildError(
                    f"{path.relative_to(repo_root)} may contain only {expected_kind} models"
                )
            if expected_kind == "physical":
                publishers = {str(item.get("publisher") or "") for item in documents}
                if len(publishers) != 1 or "" in publishers:
                    raise CatalogBuildError(
                        f"{path.relative_to(repo_root)} must contain one model creator"
                    )
                publisher = publishers.pop()
                previous = physical_creator_files.get(publisher)
                if previous is not None:
                    raise CatalogBuildError(
                        f"physical model creator {publisher!r} must use one file: "
                        f"{previous.relative_to(repo_root)} and "
                        f"{path.relative_to(repo_root)}"
                    )
                physical_creator_files[publisher] = path

    misplaced = [
        path
        for path in model_root.rglob("*.yaml")
        if path.parent.name not in expected_kinds or path.parent.parent != model_root
    ]
    if misplaced:
        raise CatalogBuildError(
            "model resources must live directly under models/single or models/virtual: "
            + ", ".join(str(path.relative_to(repo_root)) for path in sorted(misplaced))
        )


def _model_resource_locations(
    manifest: dict[str, Any], source_root: Path
) -> dict[str, tuple[str, str]]:
    model_root = _resource_directory(manifest, source_root, "models")
    locations: dict[str, tuple[str, str]] = {}
    for partition in ("single", "virtual"):
        for path in sorted((model_root / partition).glob("*.yaml")):
            raw = _load_yaml(path)
            documents = raw if isinstance(raw, list) else [raw]
            for item in documents:
                if isinstance(item, dict) and item.get("id"):
                    locations[str(item["id"])] = (partition, path.name)
    return locations


def validate_evaluation_resource_layout(
    manifest: dict[str, Any], source_root: Path, repo_root: Path
) -> None:
    """Keep physical and virtual measurements beside the matching model group."""

    evaluation_root = _resource_directory(manifest, source_root, "evaluations")
    if not evaluation_root.is_dir():
        raise CatalogBuildError(
            "resources.evaluations must be a directory partitioned into single and virtual"
        )
    model_locations = _model_resource_locations(manifest, source_root)
    partitions = {"single", "virtual"}
    for partition in sorted(partitions):
        partition_root = evaluation_root / partition
        if not partition_root.is_dir():
            raise CatalogBuildError(f"resources.evaluations/{partition} is required")
        for path in sorted(partition_root.glob("*.yaml")):
            raw = _load_yaml(path)
            documents = raw if isinstance(raw, list) else [raw]
            for index, item in enumerate(documents):
                if not isinstance(item, dict):
                    raise CatalogBuildError(
                        f"{path.relative_to(repo_root)}[{index}] must be an evaluation mapping"
                    )
                model_id = str(item.get("model") or "")
                model_location = model_locations.get(model_id)
                if model_location is None:
                    continue
                expected_partition, expected_filename = model_location
                if partition != expected_partition or path.name != expected_filename:
                    raise CatalogBuildError(
                        f"evaluation for {model_id!r} must live in "
                        f"evaluations/{expected_partition}/{expected_filename}"
                    )

    misplaced = [
        path
        for path in evaluation_root.rglob("*.yaml")
        if path.parent.name not in partitions or path.parent.parent != evaluation_root
    ]
    if misplaced:
        raise CatalogBuildError(
            "evaluation resources must live directly under evaluations/single or "
            "evaluations/virtual: "
            + ", ".join(str(path.relative_to(repo_root)) for path in sorted(misplaced))
        )


def _positive_integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CatalogBuildError(f"{path} must be a positive integer")
    return value


def _creator_policies(physical: dict[str, Any]) -> dict[str, tuple[int, list[str]]]:
    default_minimum = _positive_integer(
        physical.get("default_min_representatives"),
        "inventory.physical.default_min_representatives",
    )
    policies: dict[str, tuple[int, list[str]]] = {}
    for index, raw_creator in enumerate(
        _sequence(physical.get("creators"), "inventory.physical.creators")
    ):
        path = f"inventory.physical.creators[{index}]"
        creator = _mapping(raw_creator, path)
        _reject_unknown(
            creator,
            {"publisher", "min_representatives", "representative_models"},
            path,
        )
        publisher = _nonempty_string(creator.get("publisher"), f"{path}.publisher")
        if publisher in policies:
            raise CatalogBuildError(
                f"{path}.publisher duplicates curated creator {publisher!r}"
            )
        minimum = _positive_integer(
            creator.get("min_representatives", default_minimum),
            f"{path}.min_representatives",
        )
        representatives = [
            _nonempty_string(model, f"{path}.representative_models[{model_index}]")
            for model_index, model in enumerate(
                _sequence(
                    creator.get("representative_models"),
                    f"{path}.representative_models",
                )
            )
        ]
        if len(representatives) != len(set(representatives)):
            raise CatalogBuildError(f"{path}.representative_models must be unique")
        if len(representatives) < minimum:
            raise CatalogBuildError(
                f"{path}.representative_models has {len(representatives)} models "
                f"(minimum {minimum})"
            )
        policies[publisher] = (minimum, representatives)
    if not policies:
        raise CatalogBuildError("inventory.physical.creators cannot be empty")
    return policies


def _validate_creator_membership(
    policies: dict[str, tuple[int, list[str]]],
    published_physical: list[dict[str, Any]],
) -> None:
    expected_publishers = set(policies)
    actual_publishers = {
        str(model.get("publisher") or "") for model in published_physical
    }
    differences = {
        "missing curated creators": sorted(expected_publishers - actual_publishers),
        "unlisted physical creators": sorted(actual_publishers - expected_publishers),
    }
    details = [
        f"{label}: {', '.join(values)}"
        for label, values in differences.items()
        if values
    ]
    if details:
        raise CatalogBuildError(
            "inventory.physical does not match models: " + "; ".join(details)
        )


def _validate_representative_models(
    policies: dict[str, tuple[int, list[str]]],
    published_physical: list[dict[str, Any]],
) -> None:
    models_by_id = {str(model.get("id") or ""): model for model in published_physical}
    for publisher, (_, representatives) in sorted(policies.items()):
        for model_id in representatives:
            model = models_by_id.get(model_id)
            if model is None:
                raise CatalogBuildError(
                    f"inventory creator {publisher!r} representative model "
                    f"{model_id!r} is missing or removed"
                )
            if model.get("publisher") != publisher:
                raise CatalogBuildError(
                    f"inventory creator {publisher!r} cannot claim representative "
                    f"model {model_id!r} from {model.get('publisher')!r}"
                )
            if model.get("lifecycle") not in {"active", "experimental"}:
                raise CatalogBuildError(
                    f"inventory representative model {model_id!r} must be current"
                )


def validate_inventory_policy(
    manifest: dict[str, Any], items: list[dict[str, Any]]
) -> None:
    """Enforce the internal creator curation boundary for physical cards."""

    inventory = _mapping(manifest.get("inventory"), "inventory")
    _reject_unknown(inventory, {"physical"}, "inventory")
    physical = _mapping(inventory.get("physical"), "inventory.physical")
    _reject_unknown(
        physical,
        {"strategy", "default_min_representatives", "creators"},
        "inventory.physical",
    )
    if physical.get("strategy") != "curated_creator_companies":
        raise CatalogBuildError("inventory.physical.strategy is unsupported")
    policies = _creator_policies(physical)
    published_physical = [
        model
        for model in items
        if model.get("kind") == "physical" and model.get("lifecycle") != "removed"
    ]
    _validate_creator_membership(policies, published_physical)
    _validate_representative_models(policies, published_physical)
