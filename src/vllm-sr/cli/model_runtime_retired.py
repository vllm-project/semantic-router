"""Model execution settings the router no longer accepts.

Every local model runs in the built-in model runtime (``provider:
model_runtime``), which detects a package's format and owns numerics and
kernels. The NLI explainer, and with it the response cache's NLI polarity
tier, is retired. The router's parser refuses the settings below; ``vllm-sr``
refuses the same paths when it parses a configuration, and ``vllm-sr config
migrate`` rewrites them.
"""

from __future__ import annotations

from typing import Any

REMOVED_PROVIDERS = frozenset({"candle", "ort", "openvino"})
REMOVED_EMBEDDING_BACKENDS = frozenset({"candle", "openvino"})
# EmbeddingGemma and MiniLM have no runtime family; Vela Embedding (mmbert)
# replaces them. Their path keys are refused even when empty.
RETIRED_EMBEDDING_TYPES = frozenset({"gemma", "bert"})
RETIRED_EMBEDDING_PATHS = ("gemma_model_path", "bert_model_path")
# A remote hallucination detector is a hallucination_detector binding; the
# `backend: endpoint` shorthand was desugared into one.
REMOVED_DETECTOR_BACKENDS = frozenset({"candle", "endpoint"})
REMOVED_DEPLOYMENT_FIELDS = ("precision", "custom_ops_profile", "compilation_cache_dir")
# Module paths under global.model_catalog.modules and their removed fields.
REMOVED_MODULE_FIELDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("prompt_guard",), ("variant", "model_type", "use_modernbert", "use_mmbert_32k")),
    (("classifier", "domain"), ("variant", "use_modernbert", "use_mmbert_32k")),
    (("classifier", "pii"), ("use_modernbert", "use_mmbert_32k")),
    (("hallucination_mitigation", "fact_check"), ("use_modernbert", "use_mmbert_32k")),
    (("feedback_detector",), ("use_modernbert", "use_mmbert_32k")),
    (
        ("hallucination_mitigation", "detector"),
        ("enable_nli_filtering", "nli_entailment_threshold"),
    ),
    (("hallucination_mitigation",), ("explainer", "nli_model")),
)


def retired_model_fields(data: dict[str, Any]) -> list[str]:
    """The retired model execution settings a raw configuration still sets."""
    removed: list[str] = []
    root = _mapping(data.get("global"))
    catalog = _mapping(root.get("model_catalog"))
    deployments = _mapping(catalog.get("deployments"))
    for name in sorted(deployments):
        deployment = _mapping(deployments[name])
        prefix = f"global.model_catalog.deployments.{name}"
        provider = deployment.get("provider")
        if isinstance(provider, str) and provider.strip().lower() in REMOVED_PROVIDERS:
            removed.append(f"{prefix}.provider: {provider}")
        removed.extend(_present(prefix, deployment, REMOVED_DEPLOYMENT_FIELDS))
    modules = _mapping(catalog.get("modules"))
    for path, fields in REMOVED_MODULE_FIELDS:
        block = modules
        for key in path:
            block = _mapping(block.get(key))
        removed.extend(
            _present("global.model_catalog.modules." + ".".join(path), block, fields)
        )
    detector = _mapping(
        _mapping(modules.get("hallucination_mitigation")).get("detector")
    )
    backend = detector.get("backend")
    if (
        isinstance(backend, str)
        and backend.strip().lower() in REMOVED_DETECTOR_BACKENDS
    ):
        removed.append(
            "global.model_catalog.modules.hallucination_mitigation.detector.backend: "
            + backend
        )
    removed.extend(
        _present(
            "global.model_catalog.modules.hallucination_mitigation.detector",
            detector,
            ("endpoint",),
        )
    )
    semantic = _mapping(_mapping(catalog.get("embeddings")).get("semantic"))
    embedding_config = _mapping(semantic.get("embedding_config"))
    backend = embedding_config.get("backend")
    if (
        isinstance(backend, str)
        and backend.strip().lower() in REMOVED_EMBEDDING_BACKENDS
    ):
        removed.append(
            f"global.model_catalog.embeddings.semantic.embedding_config.backend: {backend}"
        )
    removed.extend(_retired_embedding_models(root, semantic, embedding_config))
    removed.extend(
        _present(
            "global.model_catalog.system",
            _mapping(catalog.get("system")),
            ("hallucination_explainer",),
        )
    )
    removed.extend(
        _present(
            "global.stores.response_cache",
            _mapping(_mapping(root.get("stores")).get("response_cache")),
            ("polarity_guard",),
        )
    )
    removed.extend(_nli_routing_fields("routing", _mapping(data.get("routing"))))
    recipes = data.get("recipes")
    for index, recipe in enumerate(recipes if isinstance(recipes, list) else []):
        removed.extend(
            _nli_routing_fields(
                f"recipes[{index}].routing", _mapping(_mapping(recipe).get("routing"))
            )
        )
    return removed


def _retired_embedding_models(
    root: dict[str, Any], semantic: dict[str, Any], embedding_config: dict[str, Any]
) -> list[str]:
    """Settings that select the retired gemma or bert embedding models, and
    their path keys."""
    removed = _present(
        "global.model_catalog.embeddings.semantic", semantic, RETIRED_EMBEDDING_PATHS
    )
    selections = [
        (
            "global.model_catalog.embeddings.semantic.embedding_config.model_type",
            embedding_config.get("model_type"),
        )
    ]
    stores = _mapping(root.get("stores"))
    selections += [
        (
            f"global.stores.{name}.embedding_model",
            _mapping(stores[name]).get("embedding_model"),
        )
        for name in sorted(stores)
    ]
    ml = _mapping(
        _mapping(_mapping(root.get("router")).get("model_selection")).get("ml")
    )
    selections.append(
        ("global.router.model_selection.ml.model_type", ml.get("model_type"))
    )
    for field, value in selections:
        if isinstance(value, str) and value.strip().lower() in RETIRED_EMBEDDING_TYPES:
            removed.append(f"{field}: {value}")
    return removed


def _nli_routing_fields(prefix: str, routing: dict[str, Any]) -> list[str]:
    """``use_nli`` on hallucination rules and plugins, and the fusion
    grounding's NLI penalty (now ``contradiction_penalty``)."""
    removed: list[str] = []
    rules = _mapping(routing.get("signals")).get("hallucination")
    for index, rule in enumerate(rules if isinstance(rules, list) else []):
        removed.extend(
            _present(
                f"{prefix}.signals.hallucination[{index}]", _mapping(rule), ("use_nli",)
            )
        )
    decisions = routing.get("decisions")
    for index, decision in enumerate(decisions if isinstance(decisions, list) else []):
        algorithm = _mapping(_mapping(decision).get("algorithm"))
        removed.extend(
            _present(
                f"{prefix}.decisions[{index}].algorithm.fusion.grounding",
                _mapping(_mapping(algorithm.get("fusion")).get("grounding")),
                ("nli_contradiction_penalty",),
            )
        )
        plugins = _mapping(decision).get("plugins")
        for position, entry in enumerate(plugins if isinstance(plugins, list) else []):
            plugin = _mapping(entry)
            if plugin.get("type") == "hallucination":
                removed.extend(
                    _present(
                        f"{prefix}.decisions[{index}].plugins[{position}].configuration",
                        _mapping(plugin.get("configuration")),
                        ("use_nli",),
                    )
                )
    return removed


def _present(prefix: str, block: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [f"{prefix}.{field}" for field in fields if field in block]


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
