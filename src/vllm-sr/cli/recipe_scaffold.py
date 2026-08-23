"""Scaffold maintained five-file recipe directories for vLLM-SR."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from cli.model_catalog import DEFAULT_CHANNEL, ModelCatalogError, find_catalog_model
from cli.parser import parse_user_config
from cli.validator import validate_user_config

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parent / "templates" / "config.template.yaml"
)
RECIPES_ROOT = REPO_ROOT / "config" / "recipes"
BUILTIN_ROOT = RECIPES_ROOT / "built-in" / "latest"
REQUIRED_RECIPE_FILES = (
    "config.yaml",
    "metadata.yaml",
    "recipe.dsl",
    "probes.yaml",
    "README.md",
)
PLACEHOLDER_ENDPOINT = "host.docker.internal:8000"
_RECIPE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class RecipeScaffoldError(Exception):
    """Raised when recipe scaffolding cannot complete safely."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecipeScaffoldError(f"expected mapping in {path}")
    return payload


def _dump_yaml(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False)


def _validate_recipe_id(name: str) -> str:
    slug = name.strip().lower().replace("_", "-")
    if not _RECIPE_ID.fullmatch(slug):
        raise RecipeScaffoldError(
            "recipe name must use lowercase letters, digits, and hyphens only"
        )
    return slug


def _redact_config(document: dict[str, Any]) -> dict[str, Any]:
    redacted = yaml.safe_load(yaml.safe_dump(document))
    for model in redacted.get("providers", {}).get("models", []):
        for backend in model.get("backend_refs", []):
            if "endpoint" in backend:
                backend["endpoint"] = PLACEHOLDER_ENDPOINT
            if "base_url" in backend:
                backend["base_url"] = "https://example.invalid/v1"
            for secret_key in ("api_key", "api_key_env"):
                if secret_key in backend:
                    backend.pop(secret_key, None)
    return redacted


def _default_config(recipe_id: str, *, multi_profile: bool) -> dict[str, Any]:
    template_path = DEFAULT_TEMPLATE
    document = _load_yaml(template_path)
    model_name = f"{recipe_id}-model"
    document["providers"]["defaults"]["default_model"] = model_name
    document["providers"]["models"] = [
        {
            "name": model_name,
            "backend_refs": [
                {
                    "name": "primary",
                    "endpoint": PLACEHOLDER_ENDPOINT,
                    "protocol": "http",
                    "weight": 100,
                }
            ],
        }
    ]
    document["routing"]["modelCards"] = [{"name": model_name}]
    document["routing"]["decisions"][0]["modelRefs"] = [
        {"model": model_name, "use_reasoning": False}
    ]
    document.setdefault("global", {})
    document["global"].setdefault("router", {})
    if multi_profile:
        routing_block = document.pop("routing")
        model_cards = routing_block.pop("modelCards", [])
        recipe_routing = {"decisions": routing_block.pop("decisions")}
        document["routing"] = {"modelCards": model_cards}
        document["entrypoints"] = [
            {"model_names": [f"vllm-sr/{recipe_id}-blend"], "recipe": "default"}
        ]
        document["recipes"] = [{"name": "default", "routing": recipe_routing}]
    else:
        document["global"]["router"]["auto_model_names"] = ["vllm-sr/auto"]
    return document


def _catalog_recipe_dir(model_id: str) -> Path:
    _catalog, model = find_catalog_model(model_id, catalog_version=DEFAULT_CHANNEL)
    candidate = BUILTIN_ROOT / model.asset
    if candidate.is_dir():
        return candidate
    raise RecipeScaffoldError(
        f"no built-in recipe directory found for catalog model {model_id} (asset={model.asset})"
    )


def _source_recipe_dir(from_recipe: str | None, from_model: str | None) -> Path | None:
    if from_recipe:
        candidate = RECIPES_ROOT / from_recipe
        if not candidate.is_dir():
            raise RecipeScaffoldError(f"recipe not found: {from_recipe}")
        return candidate
    if from_model:
        return _catalog_recipe_dir(from_model)
    return None


def _metadata_for(recipe_id: str, title: str) -> dict[str, Any]:
    return {
        "schema_version": "vllm-sr/recipe-metadata/v1",
        "id": recipe_id,
        "name": title,
        "version": "0.1.0",
        "description": f"Scaffolded recipe for {title}. Replace placeholders before production use.",
        "authors": [
            {
                "name": "vLLM Semantic Router Contributors",
                "url": "https://github.com/vllm-project/semantic-router/graphs/contributors",
            }
        ],
        "license": "Apache-2.0",
        "tags": ["scaffold", "starter"],
        "links": {
            "source": f"https://github.com/vllm-project/semantic-router/tree/main/config/recipes/{recipe_id}",
            "documentation": f"https://github.com/vllm-project/semantic-router/blob/main/config/recipes/{recipe_id}/README.md",
        },
    }


def _probes_for(recipe_id: str, decision_name: str) -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "name": recipe_id,
        "description": f"Starter probes for the {recipe_id} recipe. Expand coverage before promotion.",
        "routing_assets": {
            "yaml": f"config/recipes/{recipe_id}/config.yaml",
            "dsl": f"config/recipes/{recipe_id}/recipe.dsl",
        },
        "router_eval_endpoint": "/api/v1/eval",
        "evaluation": {"request_timeout_seconds": 300, "concurrency": 1},
        "acceptance": {"min_probe_pass_rate": 100.0, "min_decision_pass_rate": 100.0},
        "coverage": {
            "min_signal_assertion_percent": 0.0,
            "min_projection_assertion_percent": 0.0,
            "min_algorithm_assertion_percent": 100.0,
            "min_plugin_assertion_percent": 0.0,
            "required_request_shapes": ["text"],
            "min_tag_counts": {},
            "min_tag_pass_rate": {},
        },
        "decisions": [
            {
                "id": decision_name,
                "expected_decision": decision_name,
                "expected_algorithm": "static",
                "objective": "Default catch-all route for the scaffolded recipe.",
                "variants": [
                    {
                        "id": "hello",
                        "query": "Say hello and confirm the router is reachable.",
                        "tags": ["baseline"],
                    }
                ],
            }
        ],
    }


def _dsl_for(decision_name: str, model_name: str) -> str:
    return "\n".join(
        [
            "# Scaffolded routing DSL.",
            "# Extend signals and decisions using config/fragments/ as reference.",
            "",
            f'DECISION "{decision_name}" {{',
            "  priority: 100",
            "  rules: []",
            f'  modelRefs: [{{ model: "{model_name}", use_reasoning: false }}]',
            "  algorithm: static",
            "}",
            "",
        ]
    )


def _readme_for(recipe_id: str, title: str) -> str:
    return "\n".join(
        [
            f"# {title} Recipe Model Card",
            "",
            "## Overview",
            "",
            f"{title} is a scaffolded routing recipe generated by `vllm-sr recipe scaffold`.",
            "Replace placeholder backends and expand probes before production use.",
            "",
            "## Model details",
            "",
            "| Role | Placeholder |",
            "| --- | --- |",
            f"| Default route | `{recipe_id}-model` |",
            "",
            "## Intended use",
            "",
            "Use this recipe as a starting point for a custom maintained routing profile.",
            "",
            "## Routing behavior",
            "",
            "All requests currently route to the default catch-all decision.",
            "",
            "## Requirements",
            "",
            f"- Reachable OpenAI-compatible endpoint at `{PLACEHOLDER_ENDPOINT}`.",
            "- Replace placeholder endpoints before activation.",
            "- Pass secrets with `vllm-sr serve --recipe-env VAR` when env-backed credentials are configured.",
            "",
            "## Data handling and safety",
            "",
            "Review data retention, replay, and plugin behavior before production use.",
            "",
            "## Quick start",
            "",
            "```bash",
            f"vllm-sr validate --config config/recipes/{recipe_id}/config.yaml",
            f"vllm-sr serve --config config/recipes/{recipe_id}/config.yaml",
            "```",
            "",
            "## Evaluation",
            "",
            f"Starter probes live in [`probes.yaml`](probes.yaml). See [`../CONFORMANCE.md`](../CONFORMANCE.md).",
            "",
            "## Limitations",
            "",
            "- Placeholder backends are not production-ready.",
            "- Probe coverage is minimal until expanded.",
            "",
            "## References",
            "",
            "- [Recipe metadata](metadata.yaml)",
            "- [Runtime configuration](config.yaml)",
            "- [Routing DSL](recipe.dsl)",
            "- [Evaluation probes](probes.yaml)",
            "- [Config fragments](https://github.com/vllm-project/semantic-router/tree/main/config/fragments)",
            "",
        ]
    )


def _validate_document(document: dict[str, Any], config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_dump_yaml(document), encoding="utf-8")
    user_config = parse_user_config(str(config_path), log_summary=False)
    errors = validate_user_config(user_config, log_summary=False)
    if errors:
        raise RecipeScaffoldError(
            "generated config failed validation:\n" + "\n".join(errors)
        )


def scaffold_recipe(
    name: str,
    *,
    output: Path | None = None,
    from_recipe: str | None = None,
    from_model: str | None = None,
    multi_profile: bool = False,
) -> dict[str, Any]:
    recipe_id = _validate_recipe_id(name)
    title = recipe_id.replace("-", " ").title()
    destination = output or (RECIPES_ROOT / recipe_id)
    destination = destination.resolve()
    if destination.exists():
        raise RecipeScaffoldError(
            f"refusing to overwrite existing directory: {destination}"
        )

    source_dir = _source_recipe_dir(from_recipe, from_model)
    if source_dir is not None:
        shutil.copytree(source_dir, destination)
        config_path = destination / "config.yaml"
        document = _redact_config(_load_yaml(config_path))
        metadata = _load_yaml(destination / "metadata.yaml")
        metadata["id"] = recipe_id
        metadata["name"] = title
        metadata["version"] = "0.1.0"
        metadata.setdefault("tags", [])
        if "scaffold" not in metadata["tags"]:
            metadata["tags"] = list(metadata["tags"]) + ["scaffold"]
        probes = _load_yaml(destination / "probes.yaml")
        probes["name"] = recipe_id
        probes["routing_assets"] = {
            "yaml": f"config/recipes/{recipe_id}/config.yaml",
            "dsl": f"config/recipes/{recipe_id}/recipe.dsl",
        }
        readme = _readme_for(recipe_id, title)
        _validate_document(document, config_path)
        (destination / "metadata.yaml").write_text(
            _dump_yaml(metadata), encoding="utf-8"
        )
        (destination / "probes.yaml").write_text(_dump_yaml(probes), encoding="utf-8")
        (destination / "README.md").write_text(readme, encoding="utf-8")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        document = _default_config(recipe_id, multi_profile=multi_profile)
        if multi_profile:
            decision_name = document["recipes"][0]["routing"]["decisions"][0]["name"]
            model_name = document["providers"]["models"][0]["name"]
        else:
            decision_name = document["routing"]["decisions"][0]["name"]
            model_name = document["providers"]["models"][0]["name"]
        config_path = destination / "config.yaml"
        _validate_document(document, config_path)
        (destination / "metadata.yaml").write_text(
            _dump_yaml(_metadata_for(recipe_id, title)), encoding="utf-8"
        )
        (destination / "probes.yaml").write_text(
            _dump_yaml(_probes_for(recipe_id, decision_name)), encoding="utf-8"
        )
        (destination / "recipe.dsl").write_text(
            _dsl_for(decision_name, model_name), encoding="utf-8"
        )
        (destination / "README.md").write_text(
            _readme_for(recipe_id, title), encoding="utf-8"
        )

    written = sorted(path.name for path in destination.iterdir() if path.is_file())
    missing = [name for name in REQUIRED_RECIPE_FILES if name not in written]
    if missing:
        raise RecipeScaffoldError(f"scaffold incomplete; missing files: {missing}")

    return {
        "recipe_id": recipe_id,
        "output": str(destination),
        "files": written,
        "source": str(source_dir) if source_dir else "template",
        "multi_profile": multi_profile,
        "verification": "validated",
    }
