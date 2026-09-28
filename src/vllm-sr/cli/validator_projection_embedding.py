"""Projection graph and embedding modality validators."""

from __future__ import annotations

from cli.config_contract import CONDITION_TYPE_PROJECTION, iter_routing_profiles
from cli.models import UserConfig
from cli.validation_error import ValidationError


def _projection_dfs_visit(
    name: str,
    adj: dict[str, list[str]],
    state: dict[str, int],
    path: list[str],
    errors: list[ValidationError],
    unvisited: int,
    visiting: int,
    visited: int,
) -> None:
    if state.get(name) == visited:
        return
    if state.get(name) == visiting:
        cycle = [*list(path), name]
        start = cycle.index(name)
        cycle_str = " -> ".join(cycle[start:])
        errors.append(
            ValidationError(
                field="routing.projections.scores",
                message=f"dependency cycle detected: {cycle_str}",
            )
        )
        return
    state[name] = visiting
    path.append(name)
    for dep in adj.get(name, []):
        _projection_dfs_visit(
            dep, adj, state, path, errors, unvisited, visiting, visited
        )
    path.pop()
    state[name] = visited


def _projection_output_to_source_from_mappings(projections) -> dict[str, str]:
    output_to_source: dict[str, str] = {}
    for mapping in getattr(projections, "mappings", None) or []:
        source_score = getattr(mapping, "source", None)
        if not source_score:
            continue
        for output in getattr(mapping, "outputs", None) or []:
            output_name = getattr(output, "name", None)
            if output_name:
                output_to_source[output_name] = source_score
    return output_to_source


def _projection_deps_from_inputs(
    score_name: str,
    inputs,
    output_to_source: dict[str, str],
    score_names: set[str],
    errors: list[ValidationError],
) -> list[str]:
    deps: list[str] = []
    for inp in inputs or []:
        if (getattr(inp, "type", "") or "").lower() != CONDITION_TYPE_PROJECTION:
            continue
        dep_name = getattr(inp, "name", None)
        if not dep_name:
            continue
        vs = (getattr(inp, "value_source", "") or "").strip().lower()
        if vs == "confidence":
            src = output_to_source.get(dep_name)
            if not src:
                errors.append(
                    ValidationError(
                        field=f"routing.projections.scores[{score_name}]",
                        message=(
                            f'projection input references undefined mapping output "{dep_name}"'
                        ),
                    )
                )
                continue
            deps.append(src)
        elif dep_name not in score_names:
            errors.append(
                ValidationError(
                    field=f"routing.projections.scores[{score_name}]",
                    message=f'projection input references undefined score "{dep_name}"',
                )
            )
        else:
            deps.append(dep_name)
    return deps


def _register_projection_names(
    kind: str,
    collection: list,
    declared_names: dict[str, set[str]],
    output_names: set[str],
    errors: list[ValidationError],
    profile_name: str,
) -> None:
    for projection in collection:
        name = getattr(projection, "name", "")
        if name and name in declared_names[kind]:
            errors.append(
                ValidationError(
                    field=f"recipes.{profile_name}.routing.projections.{kind}",
                    message=(
                        f'duplicate projection name "{name}" in recipe "{profile_name}"'
                    ),
                )
            )
        if name:
            declared_names[kind].add(name)
        if kind != "mappings":
            continue
        for output in getattr(projection, "outputs", None) or []:
            output_name = getattr(output, "name", "")
            if output_name and output_name in output_names:
                errors.append(
                    ValidationError(
                        field=(f"recipes.{profile_name}.routing.projections.mappings"),
                        message=(
                            f'duplicate projection output "{output_name}" in recipe '
                            f'"{profile_name}"'
                        ),
                    )
                )
            if output_name:
                output_names.add(output_name)


def _validate_projection_profile(
    projections, profile_name: str
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    declared_names: dict[str, set[str]] = {
        "scores": set(),
        "mappings": set(),
        "partitions": set(),
    }
    output_names: set[str] = set()
    for kind, collection in (
        ("scores", getattr(projections, "scores", None) or []),
        ("mappings", getattr(projections, "mappings", None) or []),
        ("partitions", getattr(projections, "partitions", None) or []),
    ):
        _register_projection_names(
            kind,
            collection,
            declared_names,
            output_names,
            errors,
            profile_name,
        )

    scores = getattr(projections, "scores", None) or []
    score_names = {s.name for s in scores if s.name}
    output_to_source = _projection_output_to_source_from_mappings(projections)

    adj: dict[str, list[str]] = {}
    for score in scores:
        adj[score.name] = _projection_deps_from_inputs(
            score.name,
            getattr(score, "inputs", None),
            output_to_source,
            score_names,
            errors,
        )

    unvisited, visiting, visited = 0, 1, 2
    state: dict[str, int] = {s.name: unvisited for s in scores}
    path: list[str] = []

    for score in scores:
        if state.get(score.name) == unvisited:
            _projection_dfs_visit(
                score.name,
                adj,
                state,
                path,
                errors,
                unvisited,
                visiting,
                visited,
            )

    return errors


def validate_projection_score_dependencies(
    config: UserConfig,
) -> list[ValidationError]:
    """Validate each recipe's projection graph as an isolated namespace."""
    errors: list[ValidationError] = []
    for profile_name, routing in iter_routing_profiles(config):
        errors.extend(_validate_projection_profile(routing.projections, profile_name))
    return errors


def validate_embedding_modality_compatibility(
    config: UserConfig,
) -> list[ValidationError]:
    """Require a shared multimodal encoder or an explicit recipe binding.

    The Router validates the loaded binding's actual image/original-audio
    capabilities before publishing a runtime. CLI validation never loads models.
    """
    errors: list[ValidationError] = []
    catalog = (config.global_ or {}).get("model_catalog", {})
    if not isinstance(catalog, dict):
        catalog = {}
    semantic = catalog
    for key in ("embeddings", "semantic", "embedding_config"):
        semantic = semantic.get(key, {}) if isinstance(semantic, dict) else {}
    model_type = semantic.get("model_type", "") if isinstance(semantic, dict) else ""
    normalized = model_type.strip().lower() if isinstance(model_type, str) else ""
    global_bindings = catalog.get("bindings") or {}

    for profile_name, routing in iter_routing_profiles(config):
        bound = "embedding" in routing.model_bindings or "embedding" in global_bindings
        for rule in routing.signals.embeddings or []:
            raw = (rule.query_modality or "").strip().lower()
            if (
                (rule.image_candidates or rule.negative_image_candidates)
                and normalized != "multimodal"
                and not bound
            ):
                errors.append(
                    ValidationError(
                        f"Embedding rule '{rule.name}' image candidates require model_type=multimodal or an explicit embedding binding with image capability.",
                        field=f"{profile_name}.signals.embeddings.{rule.name}",
                    )
                )
            if raw in ("", "text"):
                continue
            if raw in ("image", "audio"):
                if normalized == "multimodal" or bound:
                    continue
                message = (
                    f"Embedding rule '{rule.name}' declares query_modality={raw}, "
                    "which requires global.model_catalog.embeddings.semantic."
                    "embedding_config.model_type=multimodal or an explicit embedding "
                    f"binding with that capability (current: '{model_type}')."
                )
            else:
                message = (
                    f"Embedding rule '{rule.name}' declares unknown "
                    f"query_modality='{rule.query_modality}' "
                    "(allowed values: text, image, audio)."
                )
            errors.append(
                ValidationError(
                    message, field=f"{profile_name}.signals.embeddings.{rule.name}"
                )
            )
    return errors
