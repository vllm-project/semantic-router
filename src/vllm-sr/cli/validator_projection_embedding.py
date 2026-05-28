"""Projection graph and embedding modality validators."""

from __future__ import annotations

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
        if (getattr(inp, "type", "") or "").lower() != "projection":
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


def validate_projection_score_dependencies(
    config: UserConfig,
) -> list[ValidationError]:
    """Validate that projection scores have no dependency cycles."""
    errors: list[ValidationError] = []
    projections = getattr(getattr(config, "routing", None), "projections", None)
    if not projections:
        return errors

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


def validate_embedding_modality_compatibility(
    config: UserConfig,
) -> list[ValidationError]:
    """
    Validate that image/audio query_modality embedding rules are paired with a
    multimodal embedding model. Mirrors the Go-side ``validateEmbeddingContracts``
    so the CLI catches the same misconfiguration the router would reject at load.

    Rules:
      - ``text`` (or unset): always allowed.
      - ``image``: requires
        ``global.model_catalog.embeddings.semantic.embedding_config.model_type=multimodal``.
      - ``audio``: rejected with a "planned" message — the audio FFI is not
        yet exposed by candle-binding.
      - any other value: rejected as unknown (defense-in-depth; Pydantic's
        ``Literal`` type enforces the same constraint at parse time).

    Args:
        config: User configuration

    Returns:
        list: List of validation errors
    """
    errors: list[ValidationError] = []
    embeddings = config.signals.embeddings or []
    if not embeddings:
        return errors

    # Resolve embedding model_type from the canonical v0.3 path
    # (global.model_catalog.embeddings.semantic.embedding_config.model_type).
    # Falls back to "" if the path is absent; the Go side defaults to "qwen3"
    # for an empty value, which keeps image/audio rules from being accepted
    # without an explicit multimodal opt-in.
    model_type = ""
    if isinstance(config.global_, dict):
        try:
            model_type = (
                config.global_.get("model_catalog", {})
                .get("embeddings", {})
                .get("semantic", {})
                .get("embedding_config", {})
                .get("model_type", "")
            ) or ""
        except (AttributeError, TypeError):
            model_type = ""

    normalized = model_type.strip().lower() if isinstance(model_type, str) else ""

    for rule in embeddings:
        raw = (rule.query_modality or "").strip().lower() if rule.query_modality else ""
        if raw in ("", "text"):
            continue
        if raw == "image":
            if normalized != "multimodal":
                errors.append(
                    ValidationError(
                        f"Embedding rule '{rule.name}' declares query_modality=image, "
                        f"which requires global.model_catalog.embeddings.semantic."
                        f"embedding_config.model_type=multimodal (current: "
                        f"'{model_type}'). Remove the rule, set query_modality to "
                        f"text, or change the embedding model_type to multimodal.",
                        field=f"signals.embeddings.{rule.name}",
                    )
                )
        elif raw == "audio":
            errors.append(
                ValidationError(
                    f"Embedding rule '{rule.name}' declares query_modality=audio, "
                    f"but the audio FFI (MultiModalEncodeAudioFromBase64) is not "
                    f"yet exposed by candle-binding. Audio query support is "
                    f"planned; remove the rule or set query_modality to "
                    f"text/image until the FFI lands.",
                    field=f"signals.embeddings.{rule.name}",
                )
            )
        else:
            errors.append(
                ValidationError(
                    f"Embedding rule '{rule.name}' declares unknown "
                    f"query_modality='{rule.query_modality}' "
                    f"(allowed values: text, image, audio).",
                    field=f"signals.embeddings.{rule.name}",
                )
            )

    return errors
