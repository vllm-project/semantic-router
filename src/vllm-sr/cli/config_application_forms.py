"""Forms of the learned signals from before each of them asked a preset of the
decision model: the signal types the Router no longer has (reasks,
topic_continuity, classifiers, preferences, embeddings, kb), the fields of the
application signals that became their labels, predicates, stages and inputs,
and the modules, system lines, admission gates and bindings only those signals
used.

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites the ones it can; the CLI refuses them with the same lines, in
the Router's order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from cli.config_recipe_scopes import recipe_scopes

MIGRATE = "run `vllm-sr config migrate`"
DECISION_TASK_CONTRACT = "decision.v1"
LABEL_DISTRIBUTION_CONTRACT = "label_distribution.v1"
REMOVED_MODULE = "drop it; the decision model answers the signal"

Form = tuple[str, str, str, str]

COMPLEXITY_REMOVED_FIELDS = (
    "hard",
    "easy",
    "threshold",
    "hard_above",
    "easy_below",
    "hard_below",
    "easy_above",
    "composer",
    "prototype_scoring",
)
JAILBREAK_REMOVED_FIELDS = (
    "threshold",
    "direction",
    "include_history",
    "method",
    "jailbreak_patterns",
    "benign_patterns",
)
PII_REMOVED_FIELDS = ("threshold", "pii_types_allowed", "include_history", "source")
LEGACY_FEEDBACK_LABELS = {
    "satisfied": "SAT",
    "need_clarification": "NEED_CLARIFICATION",
    "wrong_answer": "WRONG_ANSWER",
    "want_different": "WANT_DIFFERENT",
}
LEGACY_MODALITY_LABELS = {"AR": "text", "DIFFUSION": "image", "BOTH": "text_image"}
# (module, its threshold key, what the threshold becomes)
REMOVED_MODULES = (
    ("safety", "", ""),
    ("complexity", "", ""),
    ("feedback_detector", "threshold", "a user_feedbacks rule's predicate"),
    ("modality_detector", "confidence_threshold", "a modality rule's predicate"),
    ("classifier.domain", "threshold", "a domain rule's predicate"),
    ("classifier.mcp", "", ""),
    ("classifier.preference", "", ""),
    (
        "hallucination_mitigation.fact_check",
        "threshold",
        "a fact_check rule's predicate",
    ),
    ("prompt_guard", "threshold", "a jailbreak rule's predicate"),
    ("classifier.pii", "threshold", "a pii rule's predicate"),
    (
        "hallucination_mitigation.detector",
        "threshold",
        "a hallucination rule's predicate",
    ),
    ("hallucination_mitigation", "", ""),
)
REMOVED_SYSTEM_LINES = (
    "safety",
    "hazard",
    "domain_classifier",
    "fact_check_classifier",
    "feedback_detector",
    "prompt_guard",
    "pii_classifier",
    "hallucination_detector",
)
REMOVED_ADMISSION_KEYS = (
    "safety",
    "hazard",
    "domain",
    "fact_check",
    "user_feedback",
    "domain_classifier",
    "fact_check_classifier",
    "feedback_detector",
    "jailbreak",
    "pii",
    "hallucination",
    "prompt_guard",
    "pii_classifier",
    "hallucination_detector",
)
# Binding keys of removed signals and modules, with the signal type a
# decision.v1 binding of a module the decision model now answers serves.
REMOVED_BINDING_CONSUMERS = {
    "domain_classifier": "domain",
    "fact_check_classifier": "fact_check",
    "feedback_detector": "user_feedback",
    "modality_detector": "modality",
    "prompt_guard": "jailbreak",
    "pii_classifier": "pii",
    "hallucination_detector": "hallucination",
    "hazard": "",
    "classifier": "",
    "preference": "",
    "reask": "",
}


def removed_application_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed application forms in data."""

    forms = [
        *_removed_signal_type_forms(data),
        *_application_field_forms(data),
        *_application_module_forms(data),
    ]
    return [
        f"{path}: {code}: {what}; {MIGRATE} ({becomes})"
        for path, code, what, becomes in forms
    ]


def _removed_signal_type_forms(data: dict[str, Any]) -> Iterator[Form]:
    code = "removed_signal_type"
    yield from _conditions_of_type(
        data,
        "reask",
        code,
        "condition type reask is turn_relation",
        "type: turn_relation",
    )
    yield from _signal_collection(
        data,
        "reasks",
        code,
        "the reask signal is a turn_relation rule",
        "signals.turn_relation with labels [reask]; input.previous_turns is twice lookback_turns",
    )
    yield from _signal_collection(
        data,
        "topic_continuity",
        code,
        "the topic_continuity signal is a turn_relation rule",
        "signals.turn_relation with labels [continue, shift]; limits.max_prior_turns becomes input.previous_turns",
    )
    yield from _conditions_of_type(
        data,
        "classifier",
        code,
        "condition type classifier is systemone",
        "type: systemone",
    )
    yield from _signal_collection(
        data,
        "classifiers",
        code,
        "the classifier signal is a systemone question",
        "signals.systemone with a choice question over the classifier's labels",
    )
    yield from _conditions_of_type(
        data,
        "preference",
        code,
        "condition type preference reads a label of one systemone question",
        "type: systemone, name: preference, label: <preference>",
    )
    yield from _signal_collection(
        data,
        "preferences",
        code,
        "the preference signal is one systemone choice question",
        "signals.systemone[preference] choosing among the preferences",
    )
    for prefix, routing in recipe_scopes(data):
        if _embedding_partitions(routing):
            yield (
                prefix,
                code,
                "projection partitions over embedding rules are removed with the embedding signal",
                "drop them; a systemone choice question chooses one of a partition's members",
            )
    yield from _conditions_of_type(
        data,
        "embedding",
        code,
        "condition type embedding is systemone",
        "type: systemone",
    )
    for prefix, routing in recipe_scopes(data):
        rules = _signals(routing).get("embeddings")
        for path, rule in _named_items(f"{prefix}.signals.embeddings", rules):
            if _text_embedding_rule(rule):
                continue
            modality = rule.get("query_modality")
            if isinstance(modality, str) and modality.lower() == "audio":
                yield (
                    path,
                    code,
                    "the embedding signal is removed, and no decision model reads audio, so no question replaces a rule with an audio query",
                    "drop the rule: audio routing returns with a decision model that reads audio input",
                )
                continue
            yield (
                path,
                code,
                "the embedding signal is removed, and a rule with image anchors has no question that replaces it",
                "write a systemone question for it: input.image reads the request's images",
            )
    for prefix, routing in recipe_scopes(data):
        signals = _signals(routing)
        if "embeddings" in signals and all(
            _text_embedding_rule(rule) for rule in _mappings(signals["embeddings"])
        ):
            yield (
                f"{prefix}.signals.embeddings",
                code,
                "the embedding signal is a systemone question",
                "signals.systemone with a yes/no question that quotes the rule's examples",
            )
    yield from _signal_collection(
        data,
        "kb",
        code,
        "the kb signal is removed",
        "nothing: its rules never match; ask a systemone set question over the "
        "knowledge base's labels instead",
    )
    for path, node in _condition_nodes(data):
        if node.get("type") in ("kb", "kb_metric"):
            yield (
                path,
                code,
                "kb conditions and kb_metric projection inputs are removed",
                "nothing: a kb condition never matches, and a score drops a kb_metric "
                "input; read a systemone question's labels instead",
            )
    if "kbs" in _mapping_at(data, "global", "model_catalog"):
        yield (
            "global.model_catalog.kbs",
            "removed_section",
            "global.model_catalog.kbs is removed with the kb signal",
            "drop it; a systemone question names its labels",
        )


def _application_field_forms(data: dict[str, Any]) -> Iterator[Form]:
    code = "removed_field"
    for prefix, routing in recipe_scopes(data):
        for path, rule in _rules(prefix, routing, "domains"):
            if "mmlu_categories" in rule:
                yield (
                    f"{path}.mmlu_categories",
                    code,
                    "a domain rule's mmlu_categories are its labels",
                    "labels",
                )
    for prefix, routing in recipe_scopes(data):
        if not _mappings(_signals(routing).get("domains")) and _named_domains(routing):
            yield (
                prefix,
                code,
                "domain conditions no longer declare the domains they name",
                "signals.domains with one rule per domain the conditions name",
            )
    yield from _unlabeled_rules(
        data,
        "fact_check",
        lambda name: name == "no_fact_check_needed",
        lambda _: (
            "a fact_check rule's name no longer chooses its option, and no_fact_check_needed would match a request that needs a fact check",
            "labels: [NO_FACT_CHECK_NEEDED] and a predicate above one minus the fact-check threshold",
        ),
    )
    yield from _unlabeled_rules(
        data,
        "user_feedbacks",
        lambda name: name in LEGACY_FEEDBACK_LABELS,
        lambda name: (
            "a user_feedbacks rule's name no longer chooses its feedback",
            f"labels: [{LEGACY_FEEDBACK_LABELS[name]}]",
        ),
    )
    yield from _unlabeled_rules(
        data,
        "modality",
        lambda name: name in LEGACY_MODALITY_LABELS,
        lambda name: (
            "a modality rule's name no longer chooses its output (AR, DIFFUSION and BOTH are text, image and text_image)",
            f"labels: [{LEGACY_MODALITY_LABELS[name]}]",
        ),
    )
    yield from _rules_with_any_key(
        data,
        "complexity",
        COMPLEXITY_REMOVED_FIELDS,
        "a complexity rule no longer scores example banks against thresholds",
        "drop hard, easy, threshold, the boundaries, composer and prototype_scoring; the complexity preset answers easy, medium or hard",
    )
    for path, node in _decision_condition_nodes(data):
        name = node.get("name")
        if node.get("type") == "complexity" and isinstance(name, str) and ":" in name:
            yield (
                path,
                code,
                "a complexity condition names its level as a label",
                "name: <rule>, label: <level>",
            )
    yield from _rules_with_any_key(
        data,
        "jailbreak",
        JAILBREAK_REMOVED_FIELDS,
        "a jailbreak rule's threshold, direction and include_history are its predicate, stage and input.history, and method and patterns are removed",
        "predicate: {gte: <threshold>}, stage, input.history",
    )
    for prefix, routing in recipe_scopes(data):
        for path, rule in _rules(prefix, routing, "safety"):
            if "hazard" in rule:
                yield (
                    f"{path}.hazard",
                    code,
                    "a safety rule's hazard part is removed",
                    "labels: the harm categories of its categories, predicate: {gte: <hazard.threshold>}",
                )
    for prefix, routing in recipe_scopes(data):
        for path, rule in _rules(prefix, routing, "safety"):
            if "hazard" not in rule and any(
                key in rule for key in ("threshold", "unsafe_labels", "model")
            ):
                yield (
                    path,
                    code,
                    "a safety rule's threshold is its predicate, and its model labels, unsafe_labels and model are removed",
                    "predicate: {gte: <threshold>}; labels name harm categories",
                )
    yield from _rules_with_any_key(
        data,
        "pii",
        PII_REMOVED_FIELDS,
        "a pii rule's threshold, pii_types_allowed, include_history and source are its predicate, ignore_labels and input",
        "predicate: {gte: <threshold>}, ignore_labels, input.history, input.tool_results",
    )


def _application_module_forms(data: dict[str, Any]) -> Iterator[Form]:
    catalog = _mapping_at(data, "global", "model_catalog")
    modules = _mapping_at(catalog, "modules")
    for module, threshold_key, threshold in REMOVED_MODULES:
        parent = _mapping_at(modules, *module.split(".")[:-1])
        if module.split(".")[-1] not in parent:
            continue
        path = f"global.model_catalog.modules.{module}"
        becomes = (
            f"drop it; its {threshold_key} is {threshold}"
            if threshold
            else REMOVED_MODULE
        )
        yield (path, "removed_section", f"{path} is removed", becomes)
    code = "removed_field"
    for path, model in _named_items(
        "global.model_catalog.external", catalog.get("external")
    ):
        if "parser_type" in model or "threshold" in model:
            yield (
                path,
                code,
                "an external model's parser_type and threshold configured the removed chat jailbreak guard",
                "drop them; a jailbreak rule asks the decision model",
            )
    batch = _mapping_at(data, "global", "services", "api", "batch_classification")
    if "max_concurrency" in batch or "concurrency_threshold" in batch:
        yield (
            "global.services.api.batch_classification",
            code,
            "batch_classification.max_concurrency and concurrency_threshold gated the removed classifier modules",
            "drop them; global.model_catalog.admission gates a deployment",
        )
    for key in _mapping_at(catalog, "admission"):
        if key in REMOVED_ADMISSION_KEYS:
            yield (
                f"global.model_catalog.admission.{key}",
                code,
                "the admission gate of a removed module",
                "drop it; the admission of the deployment that answers the signal gates it",
            )
    for key in _mapping_at(catalog, "system"):
        if key in REMOVED_SYSTEM_LINES:
            yield (
                f"global.model_catalog.system.{key}",
                code,
                "the system line of a removed module",
                "drop it; the decision model answers the signal",
            )
    scopes = [
        (f"{prefix}.model_bindings", _mapping_at(routing, "model_bindings"))
        for prefix, routing in recipe_scopes(data)
    ]
    scopes.append(("global.model_catalog.bindings", _mapping_at(catalog, "bindings")))
    for prefix, bindings in scopes:
        for key, binding in bindings.items():
            form = _removed_binding(str(key), binding)
            if form is not None:
                yield (f"{prefix}.{key}", code, *form)


def _removed_binding(key: str, binding: Any) -> tuple[str, str] | None:
    contract = binding.get("contract") if isinstance(binding, dict) else None
    complexity = key == "complexity" or key.startswith("complexity.")
    if key in REMOVED_BINDING_CONSUMERS or key.startswith("classifier."):
        signal_type = REMOVED_BINDING_CONSUMERS.get(key, "")
        if signal_type and contract == DECISION_TASK_CONTRACT:
            return f"binding {key} is named for the signal type it serves", signal_type
    elif not (complexity and contract == LABEL_DISTRIBUTION_CONTRACT):
        return None
    return (
        f"binding {key} served a removed signal or a removed classifier model",
        REMOVED_MODULE,
    )


def _conditions_of_type(
    data: dict[str, Any], condition_type: str, code: str, what: str, becomes: str
) -> Iterator[Form]:
    for path, node in _condition_nodes(data):
        if node.get("type") == condition_type:
            yield (path, code, what, becomes)


def _signal_collection(
    data: dict[str, Any], collection: str, code: str, what: str, becomes: str
) -> Iterator[Form]:
    for prefix, routing in recipe_scopes(data):
        if collection in _signals(routing):
            yield (f"{prefix}.signals.{collection}", code, what, becomes)


def _unlabeled_rules(
    data: dict[str, Any],
    collection: str,
    legacy: Callable[[str], bool],
    describe: Callable[[str], tuple[str, str]],
) -> Iterator[Form]:
    for prefix, routing in recipe_scopes(data):
        for path, rule in _rules(prefix, routing, collection):
            name = rule.get("name")
            if isinstance(name, str) and legacy(name) and "labels" not in rule:
                yield (path, "invalid_value", *describe(name))


def _rules_with_any_key(
    data: dict[str, Any],
    collection: str,
    keys: tuple[str, ...],
    what: str,
    becomes: str,
) -> Iterator[Form]:
    for prefix, routing in recipe_scopes(data):
        for path, rule in _rules(prefix, routing, collection):
            if any(key in rule for key in keys):
                yield (path, "removed_field", what, becomes)


def _text_embedding_rule(rule: dict[str, Any]) -> bool:
    modality = rule.get("query_modality")
    return (
        not rule.get("image_candidates")
        and not rule.get("negative_image_candidates")
        and not (isinstance(modality, str) and modality.lower() == "audio")
    )


def _embedding_partitions(routing: dict[str, Any]) -> list[dict[str, Any]]:
    embeddings = {
        rule.get("name") for rule in _mappings(_signals(routing).get("embeddings"))
    }
    partitions = _mapping_at(routing, "projections").get("partitions")
    return [
        partition
        for partition in _mappings(partitions)
        if any(member in embeddings for member in _strings(partition.get("members")))
    ]


def _named_domains(node: Any) -> list[str]:
    """The domains the conditions anywhere in a routing scope name."""

    names: list[str] = []
    if isinstance(node, dict):
        name = node.get("name")
        if node.get("type") == "domain" and isinstance(name, str) and name:
            names.append(name)
        children = list(node.values())
    else:
        children = node if isinstance(node, list) else []
    for child in children:
        names.extend(name for name in _named_domains(child) if name not in names)
    return names


def _condition_nodes(data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Every place a routing scope names a signal: decision conditions at any
    depth, then projection score inputs."""

    yield from _decision_condition_nodes(data)
    for prefix, routing in recipe_scopes(data):
        scores = _mapping_at(routing, "projections").get("scores")
        for path, score in _named_items(f"{prefix}.projections.scores", scores):
            yield from _named_items(f"{path}.inputs", score.get("inputs"))


def _decision_condition_nodes(
    data: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    for prefix, routing in recipe_scopes(data):
        for path, decision in _named_items(
            f"{prefix}.decisions", routing.get("decisions")
        ):
            yield from _walk(f"{path}.rules", decision.get("rules"))


def _walk(path: str, node: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from _walk(f"{path}.{key}", value)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _walk(f"{path}[{_label(item, index)}]", item)


def _rules(
    prefix: str, routing: dict[str, Any], collection: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    yield from _named_items(
        f"{prefix}.signals.{collection}", _signals(routing).get(collection)
    )


def _named_items(path: str, items: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    for index, item in enumerate(items if isinstance(items, list) else []):
        if isinstance(item, dict):
            yield f"{path}[{_label(item, index)}]", item


def _label(item: Any, index: int) -> str:
    name = item.get("name") if isinstance(item, dict) else None
    return name if isinstance(name, str) and name else str(index)


def _signals(routing: dict[str, Any]) -> dict[str, Any]:
    return _mapping_at(routing, "signals")


def _mapping_at(node: Any, *keys: str) -> dict[str, Any]:
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}


def _mappings(items: Any) -> list[dict[str, Any]]:
    return (
        [item for item in items if isinstance(item, dict)]
        if isinstance(items, list)
        else []
    )


def _strings(items: Any) -> list[str]:
    return (
        [item for item in items if isinstance(item, str)]
        if isinstance(items, list)
        else []
    )
