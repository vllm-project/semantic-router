"""Runtime bindings needed by live Recipe conformance."""

from __future__ import annotations

import copy
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

RUNTIME_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_READY_ROLES = frozenset({"viewer", "operator", "admin"})
BACKEND_FIXTURE_COUNT = 2


def bind_runtime_entrypoints(config: dict[str, Any], probes: list) -> list:
    """Supply deployment entrypoints to portable, otherwise unmodified probes."""
    names = {
        entrypoint["recipe"]: entrypoint["model_names"][0]
        for entrypoint in config.get("entrypoints", [])
    }
    return [
        (
            replace(probe, model=names[probe.expected_recipe])
            if not probe.model and probe.expected_recipe in names
            else probe
        )
        for probe in probes
    ]


def prepare_builtin_runtime(recipe_path: Path, output: Path, repo_root: Path) -> dict:
    """Compose the source bundle through the public CLI binding implementation.

    Backend fixtures provide selector metadata only. Preview never sends these
    providers a request; all routing signals still use the real router models.
    """
    sys.path.insert(0, str(repo_root / "src" / "vllm-sr"))
    # Only live composition requires the installed source CLI.
    from cli.builtin_recipes import (  # noqa: PLC0415
        initialize_builtin_recipe,
        list_builtin_recipes,
    )
    from cli.model_bundle import model_bundle_digest  # noqa: PLC0415
    from cli.model_catalog import _load_catalog_document  # noqa: PLC0415

    catalog_version, catalog = _load_catalog_document("latest")
    bundle = next(
        item
        for item in list_builtin_recipes(catalog_version)["bundles"]
        if item["name"] == recipe_path.name
    )
    if model_bundle_digest(recipe_path) != bundle["sha256"]:
        raise ValueError("CLI package and selected source bundle differ")
    entrypoints = {
        model["recipe"]: model["entrypoint"]
        for model in catalog["models"]
        if model.get("asset") == recipe_path.name and model.get("kind") == "virtual"
    }
    if set(entrypoints) != {recipe["name"] for recipe in bundle["recipes"]}:
        raise ValueError("every built-in recipe must have a catalog entrypoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = output.parent / "providers.yaml"
    source.write_text(yaml.safe_dump(builtin_provider_fixture(), sort_keys=False))
    for index, recipe in enumerate(bundle["recipes"]):
        bindings = {
            decision["name"]: [
                {"model": f"conformance/model-{candidate}"}
                for candidate in range(decision["minimum_candidates"])
            ]
            for decision in recipe["decisions"]
        }
        if any(len(refs) > BACKEND_FIXTURE_COUNT for refs in bindings.values()):
            raise ValueError("built-in fixture needs more than two backend candidates")
        bindings_path = output.parent / f"bindings-{recipe['name']}.yaml"
        bindings_path.write_text(yaml.safe_dump(bindings, sort_keys=False))
        destination = (
            output
            if index == len(bundle["recipes"]) - 1
            else output.parent / f"composed-{index}.yaml"
        )
        initialize_builtin_recipe(
            recipe["name"],
            bundle=recipe_path.name,
            config_path=source,
            bindings_path=bindings_path,
            model_name=entrypoints[recipe["name"]],
            output=destination,
            version=catalog_version,
        )
        source = destination
    return yaml.safe_load(output.read_text())


def builtin_provider_fixture() -> dict[str, Any]:
    """Explicit deployment data, with no copied recipe policy or signal mocks."""
    models = [f"conformance/model-{index}" for index in range(BACKEND_FIXTURE_COUNT)]
    benchmarks = (
        ("tiger-ai-lab/mmlu-pro@1.0.0", "accuracy", "published-standard"),
        ("idavidrein/gpqa-diamond@1.0.0", "accuracy", "published-standard"),
        ("cais/humanitys-last-exam@1.0.0", "accuracy", "text-only"),
        ("harbor/terminal-bench@2.1.0", "resolved", "published-agent"),
    )
    return {
        "version": "v0.3",
        "listeners": [{"name": "http", "address": "0.0.0.0", "port": 8899}],
        "providers": {
            "defaults": {"model": models[0]},
            "models": [
                {
                    "name": model,
                    "backend_refs": [
                        {
                            "name": "preview-only",
                            "provider": "vllm",
                            "endpoint": "127.0.0.1:9",
                            "protocol": "http",
                        }
                    ],
                    "pricing": {
                        "currency": "USD",
                        "prompt_per_1m": index + 1,
                        "completion_per_1m": index + 1,
                    },
                }
                for index, model in enumerate(models)
            ],
        },
        "routing": {
            "modelCards": [
                {
                    "name": model,
                    "context_window_size": 2_000_000,
                    "max_output_tokens": 128_000,
                    "capabilities": [
                        "chat",
                        "reasoning",
                        "tools",
                        "vision",
                        "structured_output",
                    ],
                }
                for model in models
            ],
        },
        "evaluation": {
            "records": [
                {
                    "model": model,
                    "benchmark": benchmark,
                    "benchmark_profile": profile,
                    "metrics": {metric: 0.8 + index * 0.1},
                    "source": "synthetic-conformance-selector-fixture",
                }
                for index, model in enumerate(models)
                for benchmark, metric, profile in benchmarks
            ]
        },
        "global": {
            "router": {"auto_model_names": []},
            "services": {
                "management_api": {
                    "bind_address": "0.0.0.0",
                    "auth": {
                        "mode": "bearer",
                        "tokens": [
                            {"env": "VLLM_SR_CONFORMANCE_TOKEN", "role": "admin"}
                        ],
                    },
                }
            },
        },
    }


def verify_composed_policy(authored: dict, composed: dict) -> None:
    """Preserve authored policy except for the fixture's explicit deployment data."""
    actual = copy.deepcopy(composed.get("recipes", []))
    for recipe in actual:
        for decision in recipe["routing"]["decisions"]:
            decision.pop("modelRefs", None)
    if actual != authored.get("recipes", []):
        raise ValueError("runtime composition changed authored routing policy")

    expected_global = copy.deepcopy(authored.get("global", {}))
    fixture_global = builtin_provider_fixture()["global"]
    # Keep the allowlist at owned leaves: sharing the management_api mapping
    # does not permit dropping unrelated authored settings such as its port.
    for path in (
        ("router", "auto_model_names"),
        ("services", "management_api", "bind_address"),
        ("services", "management_api", "auth"),
    ):
        expected, override = expected_global, fixture_global
        for key in path[:-1]:
            expected = expected.setdefault(key, {})
            override = override[key]
        expected[path[-1]] = override[path[-1]]
    if composed.get("global") != expected_global:
        raise ValueError("runtime composition changed authored global policy")


def management_auth_bindings(config: dict[str, Any]) -> list[tuple[str, bool]]:
    """Return configured bearer env names and identify a readiness credential."""

    management = _mapping(
        _mapping(_mapping(config.get("global")).get("services")).get("management_api")
    )
    auth = _mapping(management.get("auth"))
    mode = str(auth.get("mode") or "disabled").strip()
    if mode == "disabled":
        return []
    if mode != "bearer":
        raise ValueError("management API auth mode must be disabled or bearer")

    roles = _mapping(auth.get("roles"))
    bindings: list[tuple[str, bool]] = []
    for raw_token in _sequence(auth.get("tokens")):
        token = _mapping(raw_token)
        env_name = str(token.get("env") or "").strip()
        role = str(token.get("role") or "").strip()
        if not RUNTIME_ENV_NAME.fullmatch(env_name):
            raise ValueError("management API auth token env name is invalid")
        if not role:
            raise ValueError("management API auth token role is invalid")
        permissions = roles.get(role, [])
        if roles and not isinstance(permissions, list):
            raise ValueError("management API auth role permissions must be a list")
        can_read_ready = (
            role in DEFAULT_READY_ROLES
            if not roles
            else any(
                isinstance(permission, str) and permission in {"ready.read", "*"}
                for permission in permissions
            )
        )
        bindings.append((env_name, can_read_ready))

    if not any(can_read_ready for _env_name, can_read_ready in bindings):
        raise ValueError(
            "management API bearer auth requires a token with ready.read permission"
        )
    return bindings


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
