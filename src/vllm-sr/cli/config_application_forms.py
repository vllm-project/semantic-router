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
# The domains the Router's domain classifier reports.
ROUTING_DOMAINS = (
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)
# What each form of a domain rule without labels or mmlu_categories is, and
# what it becomes (_domain_rule_form), in the Router's form order; "dropped"
# also drops the conditions on the rule.
DOMAIN_RULE_FORMS = {
    "cased": (
        "a domain rule named for a domain in another case matched that domain, since rule names were matched in any case",
        "the domain in lower case, in the rule and each condition, score input and partition that names it",
    ),
    "dropped": (
        "a domain rule without labels or mmlu_categories whose name is no domain never matched, since the domain classifier reports domains",
        "drop it; to route on it, set labels to the domains it means",
    ),
    "partition": (
        "a domain rule without labels or mmlu_categories whose name is no domain matched only as the default of a projection partition that names it",
        "labels: [other], so the partition keeps it",
    ),
}
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
    "hallucination_explainer",
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
    "hallucination_explainer": "",
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
    yield from _domain_rules(data, "cased")
    dropped = [
        (prefix, rule.get("name"))
        for prefix, routing in recipe_scopes(data)
        for _, rule in _rules(prefix, routing, "domains")
        if _domain_rule_form(routing, rule) == "dropped"
    ]
    for path, node in _condition_nodes(data):
        if node.get("type") == "domain" and any(
            path.startswith(f"{prefix}.") and node.get("name") == name
            for prefix, name in dropped
        ):
            yield (
                path,
                "invalid_value",
                "a domain condition reads a domain rule named for no domain, which never matched",
                "nothing: the condition never matches, so migrate drops a decision that needs it, its term of an OR and a score input on it",
            )
    yield from _domain_rules(data, "dropped")
    yield from _domain_rules(data, "partition")
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
        signal_type="user_feedback",
    )
    yield from _unlabeled_rules(
        data,
        "modality",
        lambda name: name in LEGACY_MODALITY_LABELS,
        lambda name: (
            "a modality rule's name no longer chooses its output (AR, DIFFUSION and BOTH are text, image and text_image)",
            f"labels: [{LEGACY_MODALITY_LABELS[name]}]",
        ),
        signal_type="modality",
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
    signal_type: str = "",
) -> Iterator[Form]:
    """Rules without labels whose name chose their option. With signal_type, a
    rule a condition or score input names with one of its options is a v0.5
    rule without labels, whatever its name, and is left alone."""

    for prefix, routing in recipe_scopes(data):
        for path, rule in _rules(prefix, routing, collection):
            name = rule.get("name")
            if (
                isinstance(name, str)
                and legacy(name)
                and "labels" not in rule
                and not (signal_type and _named_with_label(routing, signal_type, name))
            ):
                yield (path, "invalid_value", *describe(name))


def _named_with_label(routing: dict[str, Any], signal_type: str, name: str) -> bool:
    return any(
        node.get("type") == signal_type
        and (
            (node.get("name") == name and bool(node.get("label")))
            or str(node.get("name", "")).startswith(name + ":")
        )
        for node in _routing_conditions(routing)
    )


def _routing_conditions(routing: dict[str, Any]) -> list[dict[str, Any]]:
    """The places a routing scope names a signal: its decisions' conditions at
    any depth and its score inputs."""

    nodes = [
        node
        for decision in _mappings(routing.get("decisions"))
        for _, node in _walk("", decision.get("rules"))
    ]
    for score in _mappings(_mapping_at(routing, "projections").get("scores")):
        nodes.extend(_mappings(score.get("inputs")))
    return nodes


def _domain_rules(data: dict[str, Any], form: str) -> Iterator[Form]:
    for prefix, routing in recipe_scopes(data):
        for path, rule in _rules(prefix, routing, "domains"):
            if _domain_rule_form(routing, rule) == form:
                yield (path, "invalid_value", *DOMAIN_RULE_FORMS[form])


def _domain_rule_form(routing: dict[str, Any], rule: dict[str, Any]) -> str:
    """The form of a domain rule without labels or mmlu_categories, as the
    Router reads it: "cased" for a domain in another case, "partition" for a
    name of no domain a projection partition names, "dropped" for any other
    name of no domain; "" for a domain, or for a name the Router suggests a
    domain for that the routing scope reads, whose load error is the
    suggestion."""

    name = rule.get("name")
    name = name if isinstance(name, str) else ""
    if "labels" in rule or "mmlu_categories" in rule or _is_domain(name):
        return ""
    if _is_domain(name.lower()):
        return "cased"
    partitions = [
        partition
        for partition in _mappings(
            _mapping_at(routing, "projections").get("partitions")
        )
        if partition.get("default") == name
        or name in _strings(partition.get("members"))
    ]
    named = partitions or any(
        node.get("type") == "domain" and node.get("name") == name
        for node in _routing_conditions(routing)
    )
    if _domain_suggestion(name) and named:
        return ""
    return "partition" if partitions else "dropped"


def _is_domain(name: str) -> bool:
    return name.strip() in ROUTING_DOMAINS


def _domain_suggestion(name: str) -> str:
    """The domain the Router suggests for a name that is close to one
    (SuggestSupportedRoutingDomainName), or ""."""

    name = name.strip()
    if "_" in name and _is_domain(name.replace("_", " ")):
        return name.replace("_", " ")
    best, best_distance = "", len(name) + 1
    for domain in ROUTING_DOMAINS:
        distance = _edit_distance(name.lower(), domain)
        if distance < best_distance and distance <= len(domain) // 2 + 1:
            best, best_distance = domain, distance
    return best


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


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
    depth and projection score inputs, in the order the scope writes them, as
    the Router lists them."""

    for prefix, routing in recipe_scopes(data):
        for key in routing:
            if key == "decisions":
                for path, decision in _named_items(f"{prefix}.decisions", routing[key]):
                    yield from _walk(f"{path}.rules", decision.get("rules"))
            elif key == "projections":
                scores = _mapping_at(routing, key).get("scores")
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
