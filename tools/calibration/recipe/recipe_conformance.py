#!/usr/bin/env python3
"""Discover, validate, shard, and execute maintained recipe conformance."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from recipe_conformance_coverage import (
    collect_coverage,
    configured_signal_names,
    evaluate_live_tag_policy,
    summarize_catalog_coverage,
)
from recipe_conformance_report import (
    build_consolidated_report,
    render_consolidated_markdown,
    render_coverage_markdown,
    render_tag_acceptance_markdown,
)
from recipe_conformance_runtime import (
    bind_runtime_entrypoints,
    management_auth_bindings,
    prepare_builtin_runtime,
    verify_composed_policy,
)
from recipe_conformance_sources import (
    discover_recipe_sources,
    discover_source_inventories,
    matrix_payload,
    render_source_rows,
    source_matrix_payload,
)
from recipe_metadata_schema import (
    load_recipe_metadata_document,
    validate_recipe_metadata_schema,
)
from router_calibration_evaluation import EVALUATION_SCOPES
from router_calibration_manifest import Probe, load_probe_manifest
from router_calibration_report import render_markdown_summary
from router_calibration_signal_values import signal_value_reference
from router_calibration_support import evaluate_probes, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECIPE_ROOT = REPO_ROOT / "config" / "recipes"
DEFAULT_REPORT_ROOT = REPO_ROOT / ".agent-harness" / "recipe-conformance"
REQUIRED_RECIPE_FILES = frozenset(
    {"README.md", "config.yaml", "metadata.yaml", "probes.yaml", "recipe.dsl"}
)
REQUIRED_MODEL_CARD_HEADINGS = (
    "## Overview",
    "## Model details",
    "## Intended use",
    "## Routing behavior",
    "## Requirements",
    "## Data handling and safety",
    "## Quick start",
    "## Evaluation",
    "## Limitations",
    "## References",
)
RUNTIME_RECIPE_DIRECTORIES = frozenset({".vllm-sr"})
RECIPE_CATALOG_DIRECTORIES = frozenset({"built-in"})
DEFAULT_AUTO_ENTRYPOINTS = ("vllm-sr/auto", "auto")
DEFAULT_AUTO_MODEL_NAME = "MoM"


@dataclass(frozen=True)
class RecipeIdentity:
    schema_version: str
    id: str
    name: str
    version: str
    description: str
    authors: tuple[dict[str, str], ...]
    license: str
    tags: tuple[str, ...]
    links: dict[str, str]


@dataclass(frozen=True)
class RecipeInventory:
    name: str
    directory: str
    metadata: str
    identity: RecipeIdentity
    config: str
    dsl: str
    probes: str
    decisions: tuple[str, ...]
    entrypoints: tuple[str, ...]
    auto_entrypoints: tuple[str, ...]
    variants: int
    request_shapes: tuple[str, ...]
    languages: tuple[str, ...]
    signal_families: tuple[str, ...]
    algorithms: tuple[str, ...]
    plugins: tuple[str, ...]
    fallback_decisions: tuple[str, ...]
    coverage: dict[str, Any]
    required_devices: tuple[str, ...] = ()


@dataclass
class ProbeReferenceContext:
    path: Path
    profiles: dict[str, dict[str, Any]]
    entrypoints: dict[str, str]
    decisions: dict[tuple[str, str], dict[str, Any]]
    covered_decisions: set[tuple[str, str]]
    covered_entrypoints: set[str]


def discover_recipe_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"recipe catalog does not exist: {root}")
    recipes = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name not in RECIPE_CATALOG_DIRECTORIES
    )
    if not recipes:
        raise ValueError(f"recipe catalog contains no recipe directories: {root}")
    return recipes


def validate_recipe_directory(path: Path) -> None:
    children = list(path.iterdir())
    files = {entry.name for entry in children if entry.is_file()}
    catalog_entries = {
        entry.name
        for entry in children
        if not (entry.is_dir() and entry.name in RUNTIME_RECIPE_DIRECTORIES)
    }
    missing = sorted(REQUIRED_RECIPE_FILES - files)
    extra = sorted(catalog_entries - REQUIRED_RECIPE_FILES)
    if missing or extra:
        raise ValueError(
            f"{path}: recipe files differ from the five-file contract; "
            f"missing={missing}, extra={extra}"
        )


def validate_recipe_model_card(path: Path) -> None:
    readme_path = path / "README.md"
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    title = lines[0].strip() if lines else ""
    if not title.startswith("# ") or not title.endswith(" Model Card"):
        raise ValueError(
            f"{readme_path}: recipe README title must end with 'Model Card'"
        )

    headings = [line.strip() for line in lines if line.startswith("## ")]
    missing = [
        heading for heading in REQUIRED_MODEL_CARD_HEADINGS if heading not in headings
    ]
    if missing:
        raise ValueError(
            f"{readme_path}: Model Card is missing required sections: {missing}"
        )
    positions = [headings.index(heading) for heading in REQUIRED_MODEL_CARD_HEADINGS]
    if positions != sorted(positions):
        raise ValueError(
            f"{readme_path}: Model Card sections must follow the documented order"
        )


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def load_recipe_identity(path: Path, directory_name: str) -> RecipeIdentity:
    metadata = load_recipe_metadata_document(path.read_text(encoding="utf-8"))
    validate_recipe_metadata_schema(metadata, path)
    recipe_id = str(metadata["id"])
    if recipe_id != directory_name:
        raise ValueError(
            f"{path}: metadata id must match directory {directory_name!r}, "
            f"got {recipe_id!r}"
        )
    return RecipeIdentity(
        schema_version=str(metadata["schema_version"]),
        id=recipe_id,
        name=str(metadata["name"]),
        version=str(metadata["version"]),
        description=str(metadata["description"]),
        authors=tuple(dict(author) for author in metadata["authors"]),
        license=str(metadata["license"]),
        tags=tuple(str(tag) for tag in metadata["tags"]),
        links={str(key): str(value) for key, value in metadata["links"].items()},
    )


def recipe_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {"default": _mapping(config.get("routing"))}
    for raw_recipe in _sequence(config.get("recipes")):
        recipe = _mapping(raw_recipe)
        name = str(recipe.get("name") or "").strip()
        if not name:
            continue
        profiles[name] = _mapping(recipe.get("routing"))
    return profiles


def config_auto_entrypoints(config: dict[str, Any]) -> tuple[str, ...]:
    global_config = _mapping(config.get("global"))
    if not global_config:
        return ()
    if config.get("recipes") and not any(
        key in config for key in ("providers", "routing", "entrypoints")
    ):
        # Portable bundles may carry global defaults without declaring a
        # runtime. Their entrypoints are supplied when recipes are composed.
        return ()
    router = _mapping(global_config.get("router"))
    if "auto_model_names" in router:
        raw_names = _sequence(router.get("auto_model_names"))
    else:
        configured = str(
            router.get("auto_model_name") or DEFAULT_AUTO_MODEL_NAME
        ).strip()
        raw_names = [*DEFAULT_AUTO_ENTRYPOINTS, configured]
    return tuple(
        dict.fromkeys(
            normalized
            for raw_name in raw_names
            if (normalized := str(raw_name).strip())
        )
    )


def config_named_entrypoints(config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_entrypoint in _sequence(config.get("entrypoints")):
        entrypoint = _mapping(raw_entrypoint)
        recipe = str(entrypoint.get("recipe") or "").strip()
        for model_name in _sequence(entrypoint.get("model_names")):
            normalized = str(model_name).strip()
            if normalized:
                result[normalized] = recipe
    return result


def config_entrypoints(config: dict[str, Any]) -> dict[str, str]:
    result = dict.fromkeys(config_auto_entrypoints(config), "default")
    for model_name, recipe in config_named_entrypoints(config).items():
        existing = result.get(model_name)
        if existing is not None and existing != recipe:
            raise ValueError(
                f"entrypoint {model_name!r} resolves to both "
                f"{existing!r} and {recipe!r}"
            )
        result[model_name] = recipe
    return result


def bind_default_entrypoints(
    config: dict[str, Any], probes: list[Probe]
) -> list[Probe]:
    auto_entrypoints = config_auto_entrypoints(config)
    if not auto_entrypoints:
        return probes
    bound: list[Probe] = []
    default_probe_index = 0
    for probe in probes:
        recipe = probe.expected_recipe or "default"
        if probe.model is not None or recipe != "default":
            bound.append(probe)
            continue
        model = auto_entrypoints[default_probe_index % len(auto_entrypoints)]
        bound.append(replace(probe, model=model))
        default_probe_index += 1
    return bound


def profile_decisions(
    profiles: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for recipe, profile in profiles.items():
        for raw_decision in _sequence(profile.get("decisions")):
            decision = _mapping(raw_decision)
            name = str(decision.get("name") or "").strip()
            if not name:
                continue
            key = (recipe, name)
            if key in result:
                raise ValueError(f"duplicate decision {recipe}:{name}")
            result[key] = decision
    return result


def resolve_probe_recipe(
    *,
    expected_recipe: str | None,
    request_model: str | None,
    entrypoints: dict[str, str],
    profiles: dict[str, dict[str, Any]],
) -> str:
    if expected_recipe:
        return expected_recipe
    if request_model and request_model in entrypoints:
        return entrypoints[request_model]
    if "default" in profiles:
        return "default"
    raise ValueError("cannot resolve probe recipe")


def validate_probe_references(
    path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    probes: list[Probe],
) -> None:
    profiles = recipe_profiles(config)
    context = ProbeReferenceContext(
        path=path,
        profiles=profiles,
        entrypoints=config_entrypoints(config),
        decisions=profile_decisions(profiles),
        covered_decisions=set(),
        covered_entrypoints=set(),
    )
    for probe in probes:
        _validate_probe_reference(context, probe)
    _assert_probe_coverage(context)
    if not isinstance(manifest.get("acceptance"), dict):
        raise ValueError(f"{path}: acceptance policy is required")


def _validate_probe_reference(context: ProbeReferenceContext, probe: Probe) -> None:
    recipe = resolve_probe_recipe(
        expected_recipe=probe.expected_recipe,
        request_model=probe.model,
        entrypoints=context.entrypoints,
        profiles=context.profiles,
    )
    if recipe not in context.profiles:
        raise ValueError(
            f"{context.path}: probe {probe.probe_id} references unknown "
            f"recipe {recipe!r}"
        )
    key = (recipe, probe.expected_decision)
    decision = context.decisions.get(key)
    if decision is None:
        raise ValueError(
            f"{context.path}: probe {probe.probe_id} references unknown "
            f"decision {recipe}:{probe.expected_decision}"
        )
    context.covered_decisions.add(key)
    _validate_probe_entrypoint(context, probe, recipe)
    _validate_probe_decision_contract(context.path, probe, decision)
    _validate_probe_signals(
        context.path,
        probe,
        configured_signal_names(context.profiles[recipe]),
    )


def _validate_probe_entrypoint(
    context: ProbeReferenceContext, probe: Probe, recipe: str
) -> None:
    if not probe.model:
        return
    resolved_recipe = context.entrypoints.get(probe.model)
    if resolved_recipe is None:
        if recipe != "default":
            raise ValueError(
                f"{context.path}: probe {probe.probe_id} model "
                f"{probe.model!r} is not a configured entrypoint"
            )
        return
    context.covered_entrypoints.add(probe.model)
    if resolved_recipe != recipe:
        raise ValueError(
            f"{context.path}: probe {probe.probe_id} model resolves to "
            f"{resolved_recipe!r}, not {recipe!r}"
        )


def _validate_probe_decision_contract(
    path: Path, probe: Probe, decision: dict[str, Any]
) -> None:
    model_refs = {
        str(_mapping(model_ref).get("model") or "").strip()
        for model_ref in _sequence(decision.get("modelRefs"))
    }
    if probe.expected_alias and probe.expected_alias not in model_refs:
        raise ValueError(
            f"{path}: probe {probe.probe_id} alias {probe.expected_alias!r} "
            f"is outside decision modelRefs {sorted(model_refs)}"
        )
    actual_algorithm = str(
        _mapping(decision.get("algorithm")).get("type") or "static"
    ).strip()
    if probe.expected_algorithm and probe.expected_algorithm != actual_algorithm:
        raise ValueError(
            f"{path}: probe {probe.probe_id} expects algorithm "
            f"{probe.expected_algorithm!r}, configured {actual_algorithm!r}"
        )
    configured_plugins = {
        str(_mapping(plugin).get("type") or "").strip()
        for plugin in _sequence(decision.get("plugins"))
    }
    missing_plugins = set(probe.expected_plugins) - configured_plugins
    if missing_plugins:
        raise ValueError(
            f"{path}: probe {probe.probe_id} expects missing plugins "
            f"{sorted(missing_plugins)}"
        )


def _validate_probe_signals(
    path: Path,
    probe: Probe,
    configured_signals: dict[str, set[str]],
) -> None:
    for key in probe.expected_signal_values:
        try:
            signal_value_reference(key, configured_signals)
        except ValueError as exc:
            raise ValueError(f"{path}: probe {probe.probe_id}: {exc}") from exc
    for signal_type, name in probe.expected_signals:
        configured_names = configured_signals.get(signal_type, set())
        base_name = name.split(":", 1)[0]
        if name not in configured_names and base_name not in configured_names:
            raise ValueError(
                f"{path}: probe {probe.probe_id} expects unknown signal "
                f"{signal_type}:{name}"
            )


def _assert_probe_coverage(context: ProbeReferenceContext) -> None:
    missing_decisions = sorted(set(context.decisions) - context.covered_decisions)
    if missing_decisions:
        raise ValueError(
            f"{context.path}: decisions without probes: "
            + ", ".join(f"{recipe}:{name}" for recipe, name in missing_decisions)
        )
    missing_entrypoints = sorted(set(context.entrypoints) - context.covered_entrypoints)
    if missing_entrypoints:
        raise ValueError(
            f"{context.path}: entrypoints without probes: "
            f"{', '.join(missing_entrypoints)}"
        )


def configured_accelerators(config: dict[str, Any]) -> tuple[str, ...]:
    """Keep explicit hardware requirements; never project a GPU recipe onto CPU."""
    catalog = _mapping(_mapping(config.get("global")).get("model_catalog"))
    deployments = _mapping(catalog.get("deployments"))
    devices = {
        str(_mapping(deployment).get("device") or "cpu").strip().lower()
        for deployment in deployments.values()
    }
    return tuple(sorted(devices - {"cpu"}))


def cpu_inventory(inventory: list[RecipeInventory]) -> list[RecipeInventory]:
    return [recipe for recipe in inventory if not recipe.required_devices]


def cpu_eligibility(inventory: list[RecipeInventory]) -> dict[str, Any]:
    return {
        "platform": "cpu",
        "eligible": [recipe.name for recipe in cpu_inventory(inventory)],
        "excluded": [
            {"recipe": recipe.name, "required_devices": recipe.required_devices}
            for recipe in inventory
            if recipe.required_devices
        ],
    }


def build_recipe_inventory(path: Path) -> RecipeInventory:
    validate_recipe_directory(path)
    validate_recipe_model_card(path)
    metadata_path = path / "metadata.yaml"
    config_path = path / "config.yaml"
    dsl_path = path / "recipe.dsl"
    probes_path = path / "probes.yaml"
    identity = load_recipe_identity(metadata_path, path.name)
    config = load_yaml_mapping(config_path)
    manifest, probes = load_probe_manifest(probes_path)
    probes = bind_default_entrypoints(config, probes)
    if str(manifest.get("name") or "").strip() != path.name:
        raise ValueError(
            f"{probes_path}: manifest name must match directory {path.name!r}"
        )
    assets = _mapping(manifest.get("routing_assets"))
    recipe_directory = path.relative_to(REPO_ROOT).as_posix()
    expected_yaml = f"{recipe_directory}/config.yaml"
    expected_dsl = f"{recipe_directory}/recipe.dsl"
    if assets.get("yaml") != expected_yaml or assets.get("dsl") != expected_dsl:
        raise ValueError(
            f"{probes_path}: routing_assets must be "
            f"yaml={expected_yaml!r}, dsl={expected_dsl!r}"
        )
    validate_probe_references(probes_path, config, manifest, probes)

    profiles = recipe_profiles(config)
    decisions = profile_decisions(profiles)
    entrypoints = config_entrypoints(config)
    auto_entrypoints = config_auto_entrypoints(config)
    algorithms: set[str] = set()
    plugins: set[str] = set()
    fallback_decisions: set[str] = set()
    for (recipe, name), decision in decisions.items():
        decision_plugins = {
            str(_mapping(plugin).get("type") or "").strip()
            for plugin in _sequence(decision.get("plugins"))
            if str(_mapping(plugin).get("type") or "").strip()
        }
        algorithm = str(_mapping(decision.get("algorithm")).get("type") or "").strip()
        if algorithm or "fast_response" not in decision_plugins:
            algorithms.add(algorithm or "static")
        plugins.update(decision_plugins)
        rules = _mapping(decision.get("rules"))
        if not _sequence(rules.get("conditions")):
            fallback_decisions.add(f"{recipe}:{name}")

    request_shapes = {"messages" if probe.messages else "text" for probe in probes}
    if any(probe.tools for probe in probes):
        request_shapes.add("tools")
    languages = {
        tag.split(":", 1)[1] if tag.startswith("language:") else tag
        for probe in probes
        for tag in probe.tags
        if tag.startswith("language:") or tag in {"de", "en", "es", "fr", "ja", "zh"}
    }
    signal_families = {
        signal_type
        for profile in profiles.values()
        for signal_type in configured_signal_names(profile)
    }
    coverage = collect_coverage(
        profiles=profiles,
        decisions=decisions,
        entrypoints=entrypoints,
        probes=probes,
        policy=_mapping(manifest.get("coverage")),
    )
    if not coverage["passed"]:
        raise ValueError(
            f"{probes_path}: coverage policy failed: " + "; ".join(coverage["errors"])
        )
    return RecipeInventory(
        name=path.name,
        directory=str(path.relative_to(REPO_ROOT)),
        metadata=str(metadata_path.relative_to(REPO_ROOT)),
        identity=identity,
        config=str(config_path.relative_to(REPO_ROOT)),
        dsl=str(dsl_path.relative_to(REPO_ROOT)),
        probes=str(probes_path.relative_to(REPO_ROOT)),
        decisions=tuple(sorted(f"{recipe}:{name}" for recipe, name in decisions)),
        entrypoints=tuple(sorted(entrypoints)),
        auto_entrypoints=tuple(sorted(auto_entrypoints)),
        variants=len(probes),
        request_shapes=tuple(sorted(request_shapes)),
        languages=tuple(sorted(languages)),
        signal_families=tuple(sorted(signal_families)),
        algorithms=tuple(sorted(algorithms)),
        plugins=tuple(sorted(plugins)),
        fallback_decisions=tuple(sorted(fallback_decisions)),
        coverage=coverage,
        required_devices=configured_accelerators(config),
    )


def discover_inventory(root: Path) -> list[RecipeInventory]:
    return [build_recipe_inventory(path) for path in discover_recipe_directories(root)]


def validate_catalog_readme(root: Path, inventory: list[RecipeInventory]) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8")
    missing = [
        recipe.name
        for recipe in inventory
        if f"({recipe.name}/README.md)" not in readme
    ]
    if missing:
        raise ValueError(
            f"{root / 'README.md'} does not catalog recipes: {', '.join(missing)}"
        )


def coverage_payload(inventory: list[RecipeInventory]) -> dict[str, Any]:
    entrypoint_count = sum(len(recipe.entrypoints) for recipe in inventory)
    auto_entrypoint_count = sum(len(recipe.auto_entrypoints) for recipe in inventory)
    return {
        "schema_version": "v1",
        "receipt": {
            "domain": "recipe-conformance",
            "blocking_tiers": ["T0", "T1", "T2", "T3"],
            "reporting_tiers": ["T4"],
        },
        "recipes": [asdict(recipe) for recipe in inventory],
        "summary": {
            "recipes": len(inventory),
            "decisions": sum(len(recipe.decisions) for recipe in inventory),
            "entrypoints": entrypoint_count,
            "auto_entrypoints": auto_entrypoint_count,
            "named_entrypoints": entrypoint_count - auto_entrypoint_count,
            "variants": sum(recipe.variants for recipe in inventory),
            "signal_families": sorted(
                {signal for recipe in inventory for signal in recipe.signal_families}
            ),
            "algorithms": sorted(
                {algorithm for recipe in inventory for algorithm in recipe.algorithms}
            ),
            "plugins": sorted(
                {plugin for recipe in inventory for plugin in recipe.plugins}
            ),
            "coverage": summarize_catalog_coverage(
                [recipe.coverage for recipe in inventory]
            ),
        },
    }


def write_github_output(path: Path, name: str, value: str) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"{name}={value}\n")


def command_static(args: argparse.Namespace) -> int:
    inventory = discover_inventory(args.recipes_root)
    if not args.skip_catalog_readme:
        validate_catalog_readme(args.recipes_root, inventory)
    payload = coverage_payload(inventory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "inventory.json", payload)
    (args.output_dir / "coverage.md").write_text(
        render_coverage_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


def command_list(args: argparse.Namespace) -> int:
    inventory = discover_inventory(args.recipes_root)
    if args.platform == "cpu":
        inventory = cpu_inventory(inventory)
    names = [recipe.name for recipe in inventory]
    print(",".join(names) if args.format == "csv" else json.dumps(names))
    return 0


def command_plan(args: argparse.Namespace) -> int:
    inventory = discover_inventory(args.recipes_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "cpu-eligibility.json", cpu_eligibility(inventory))
    matrix = matrix_payload(cpu_inventory(inventory), args.shards)
    encoded = json.dumps(matrix, separators=(",", ":"))
    if args.github_output:
        write_github_output(args.github_output, "matrix", encoded)
    else:
        print(json.dumps(matrix, indent=2))
    return 0


def command_check_cpu(args: argparse.Namespace) -> int:
    names = [name.strip() for name in args.recipes.split(",") if name.strip()]
    inventory = {
        recipe.name: recipe for recipe in discover_inventory(args.recipes_root)
    }
    if not names:
        raise ValueError("at least one recipe is required")
    for name in names:
        if name not in inventory:
            raise ValueError(f"unknown recipe: {name}")
        devices = inventory[name].required_devices
        if devices:
            raise ValueError(
                f"{name} requires explicit devices {', '.join(devices)}; "
                "the CPU runner does not override deployment hardware"
            )
    return 0


def command_static_all(args: argparse.Namespace) -> int:
    for source in discover_recipe_sources(args.recipes_root):
        source_args = argparse.Namespace(**vars(args))
        source_args.recipes_root = source.recipes_root
        source_args.output_dir = args.output_dir / source.report_subdir
        source_args.skip_catalog_readme = not source.validate_readme
        command_static(source_args)
    return 0


def cpu_sources(args: argparse.Namespace) -> list:
    inventories = discover_source_inventories(args.recipes_root, discover_inventory)
    for source_inventory in inventories:
        output_dir = args.output_dir / source_inventory.source.report_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "cpu-eligibility.json",
            cpu_eligibility(list(source_inventory.recipes)),
        )
    return [
        replace(item, recipes=tuple(cpu_inventory(list(item.recipes))))
        for item in inventories
    ]


def command_plan_all(args: argparse.Namespace) -> int:
    matrix = source_matrix_payload(cpu_sources(args), args.shards, REPO_ROOT)
    if not matrix["include"]:
        raise ValueError("no CPU recipes selected")
    if args.github_output:
        write_github_output(
            args.github_output, "matrix", json.dumps(matrix, separators=(",", ":"))
        )
    else:
        print(json.dumps(matrix, indent=2))
    return 0


def command_sources(args: argparse.Namespace) -> int:
    print(render_source_rows(cpu_sources(args), REPO_ROOT, args.format))
    return 0


def command_prepare_runtime(args: argparse.Namespace) -> int:
    recipe_path = args.recipes_root / args.recipe
    build_recipe_inventory(recipe_path)
    authored = load_yaml_mapping(recipe_path / "config.yaml")
    if not any(key in authored for key in ("providers", "routing", "entrypoints")):
        composed = prepare_builtin_runtime(recipe_path, args.config, REPO_ROOT)
        verify_composed_policy(authored, composed)
    else:
        args.config.parent.mkdir(parents=True, exist_ok=True)
        with args.config.open("x", encoding="utf-8") as output:
            output.write((recipe_path / "config.yaml").read_text(encoding="utf-8"))
    return 0


def command_runtime_auth(args: argparse.Namespace) -> int:
    for name, can_read_ready in management_auth_bindings(
        load_yaml_mapping(args.config)
    ):
        print(f"{name}|{'ready' if can_read_ready else 'token'}")
    return 0


def command_eval(args: argparse.Namespace) -> int:
    recipe_path = args.recipes_root / args.recipe
    inventory = build_recipe_inventory(recipe_path)
    config = load_yaml_mapping(recipe_path / "config.yaml")
    manifest, probes = load_probe_manifest(recipe_path / "probes.yaml")
    probes = bind_default_entrypoints(config, probes)
    runtime_config = getattr(args, "runtime_config", None)
    if runtime_config:
        config = load_yaml_mapping(runtime_config)
        probes = bind_runtime_entrypoints(config, probes)
        validate_probe_references(recipe_path / "probes.yaml", config, manifest, probes)
    evaluation = evaluate_probes(
        args.router_url, probes, manifest, scope=getattr(args, "scope", "deployment")
    )
    tag_acceptance = evaluate_live_tag_policy(
        evaluation, _mapping(manifest.get("coverage"))
    )
    evaluation["coverage_acceptance"] = tag_acceptance
    evaluation["passed"] = bool(evaluation["passed"]) and tag_acceptance["passed"]
    expected_ids = [probe.probe_id for probe in probes]
    executed_ids = [result["id"] for result in evaluation["results"]]
    execution_complete = bool(expected_ids) and sorted(executed_ids) == sorted(
        expected_ids
    )
    expected_entrypoints = sorted({probe.model for probe in probes if probe.model})
    executed_entrypoints = sorted(
        {result["model"] for result in evaluation["results"] if result.get("model")}
    )
    evaluation["execution"] = {
        "expected_probes": len(expected_ids),
        "executed_probes": len(executed_ids),
        "expected_entrypoints": expected_entrypoints,
        "executed_entrypoints": executed_entrypoints,
        "complete": execution_complete and expected_entrypoints == executed_entrypoints,
    }
    evaluation["passed"] = evaluation["passed"] and evaluation["execution"]["complete"]
    for scope, summary in evaluation.get("scopes", {}).items():
        scoped_results = [
            {**result, "matched": bool(result.get(f"{scope}_matched"))}
            for result in evaluation["results"]
        ]
        scoped_tags = evaluate_live_tag_policy(
            {"results": scoped_results}, _mapping(manifest.get("coverage"))
        )
        summary["coverage_acceptance"] = scoped_tags
        summary["passed"] = bool(summary["passed"]) and scoped_tags["passed"]
    report = {"inventory": asdict(inventory), "evaluation": evaluation}
    output_dir = args.output_dir / args.recipe
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "eval-report.json", report)
    summary = render_markdown_summary(
        manifest,
        evaluation,
        evaluation,
        None,
        None,
    )
    summary = summary.rstrip() + "\n\n" + render_tag_acceptance_markdown(tag_acceptance)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            {
                "recipe": args.recipe,
                "evaluation_scope": evaluation.get("evaluation_scope", "deployment"),
                "scopes": evaluation.get("scopes", {}),
                "matched": evaluation["matched"],
                "total": evaluation["total"],
                "passed": evaluation["passed"],
                "report": str(output_dir),
            },
            indent=2,
        )
    )
    return 0 if evaluation["passed"] else 1


def command_report(args: argparse.Namespace) -> int:
    payload = build_consolidated_report(args.output_dir)
    write_json(args.output_dir / "conformance-report.json", payload)
    (args.output_dir / "conformance-report.md").write_text(
        render_consolidated_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0 if payload["summary"]["cpu_compatible_passed"] else 1


def command_report_all(args: argparse.Namespace) -> int:
    result = 0
    for source in discover_recipe_sources(args.recipes_root):
        source_args = argparse.Namespace(**vars(args))
        source_args.output_dir = args.output_dir / source.report_subdir
        result = max(result, command_report(source_args))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipes-root",
        type=Path,
        default=DEFAULT_RECIPE_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
    )
    parser.add_argument(
        "--skip-catalog-readme",
        action="store_true",
        help="validate a versioned built-in root whose catalog.yaml owns discovery",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--format", choices=("csv", "json"), default="json")
    list_parser.add_argument("--platform", choices=("all", "cpu"), default="all")
    list_parser.set_defaults(func=command_list)

    static = subparsers.add_parser("static")
    static.set_defaults(func=command_static)
    subparsers.add_parser("static-all").set_defaults(func=command_static_all)

    plan = subparsers.add_parser("plan", help="plan compatible live CPU shards")
    plan.add_argument("--shards", type=int, default=3)
    plan.add_argument("--github-output", type=Path)
    plan.set_defaults(func=command_plan)
    plan_all = subparsers.add_parser("plan-all")
    plan_all.add_argument("--shards", type=int, default=3)
    plan_all.add_argument("--github-output", type=Path)
    plan_all.set_defaults(func=command_plan_all)
    sources = subparsers.add_parser("sources")
    sources.add_argument("--format", choices=("json", "pipe"), default="json")
    sources.set_defaults(func=command_sources)
    prepare = subparsers.add_parser("prepare-runtime")
    prepare.add_argument("--recipe", required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.set_defaults(func=command_prepare_runtime)
    runtime_auth = subparsers.add_parser("runtime-auth")
    runtime_auth.add_argument("--config", type=Path, required=True)
    runtime_auth.set_defaults(func=command_runtime_auth)

    check_cpu = subparsers.add_parser("check-cpu")
    check_cpu.add_argument("--recipes", required=True)
    check_cpu.set_defaults(func=command_check_cpu)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--recipe", required=True)
    evaluate.add_argument("--router-url", required=True)
    evaluate.add_argument("--runtime-config", type=Path)
    evaluate.add_argument(
        "--scope",
        choices=EVALUATION_SCOPES,
        default="deployment",
        help="Policy checks real routing evidence; deployment also requires the expected live model selection.",
    )
    evaluate.set_defaults(func=command_eval)

    report = subparsers.add_parser("report")
    report.set_defaults(func=command_report)
    subparsers.add_parser("report-all").set_defaults(func=command_report_all)
    return parser


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def main() -> int:
    args = build_parser().parse_args()
    args.recipes_root = args.recipes_root.resolve()
    args.output_dir = args.output_dir.resolve()
    try:
        return args.func(args)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"recipe conformance failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
