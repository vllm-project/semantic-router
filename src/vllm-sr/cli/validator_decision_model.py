"""Systemone conditions and selectors, and the model_runtime settings `serve` writes.

The Router validates the deployments that answer systemone questions and
selectors, and how each question resolves to one.
"""

import re

from cli.config_contract import iter_condition_leaves, iter_routing_profiles
from cli.models import UserConfig
from cli.models_systemone import OPTION_QUESTION_TYPES
from cli.validation_error import ValidationError

HUB_REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
MIN_SELECTOR_CANDIDATES = 2
MAX_SELECTOR_CANDIDATES = 255


def public_model_name(deployment: dict) -> str:
    artifact = deployment.get("artifact") or ""
    return deployment.get("public_name") or (
        artifact
        if not artifact.startswith(("models/", "./", "../"))
        and HUB_REPOSITORY_ID.fullmatch(artifact)
        else ""
    )


MODEL_RUNTIME_DTYPES = ("auto", "float32", "bfloat16", "float16")
# The runtime's bounds: a forward holds at least one row of 8 padded tokens,
# and the batching window is at most a second.
MIN_MAX_NUM_BATCHED_TOKENS = 8
MAX_BATCH_WINDOW_MS = 1000


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def engine_args_error(deployment: dict) -> str | None:
    """Why a deployment's engine arguments are malformed, or None (unset ones keep the runtime's defaults)."""
    seqs = deployment.get("max_num_seqs")
    if seqs is not None and (not _is_int(seqs) or seqs < 1):
        return "max_num_seqs must be an integer of at least 1"
    tokens = deployment.get("max_num_batched_tokens")
    if tokens is not None and (
        not _is_int(tokens) or tokens < MIN_MAX_NUM_BATCHED_TOKENS
    ):
        return "max_num_batched_tokens must be an integer of at least 8"
    window = deployment.get("batch_window_ms")
    if window is not None and (
        isinstance(window, bool)
        or not isinstance(window, (int, float))
        or not 0 <= window <= MAX_BATCH_WINDOW_MS
    ):
        return "batch_window_ms must be a number between 0 and 1000"
    dtype = deployment.get("dtype")
    if dtype is not None and dtype not in MODEL_RUNTIME_DTYPES:
        return "dtype must be one of auto, float32, bfloat16 or float16"
    shared = deployment.get("enable_shared_context")
    if shared is not None and not isinstance(shared, bool):
        return "enable_shared_context must be a boolean"
    limit = deployment.get("max_model_len")
    if limit is not None:
        if not _is_int(limit) or limit < 1:
            return "max_model_len must be an integer of at least 1"
        if deployment.get("endpoint") or deployment.get("replicas"):
            return (
                "max_model_len applies only to a managed worker, "
                "not to an attached endpoint or replicas"
            )
    return None


def validate_systemone_signals(config: UserConfig) -> list[ValidationError]:
    """Systemone conditions name declared answers; selectors choose among modelRefs."""
    errors = []
    for name, profile in iter_routing_profiles(config):
        prefix = f"recipes.{name}"
        rules = {rule.name: rule for rule in profile.signals.systemone or []}
        for decision in profile.decisions:
            errors.extend(_selector_errors(prefix, decision))
            errors.extend(_condition_errors(prefix, decision, rules))
    return errors


def _condition_errors(prefix, decision, rules) -> list[ValidationError]:
    errors = []
    field = f"{prefix}.decisions.{decision.name}.rules.conditions"
    for condition in iter_condition_leaves(decision.rules.conditions):
        if (condition.type or "").strip().lower() != "systemone":
            continue
        rule = rules.get(condition.name or "")
        if rule is None:
            continue
        if message := _condition_error(rule, condition.label):
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' systemone condition '{rule.name}' {message}",
                    field=field,
                )
            )
    return errors


def _condition_error(rule, label) -> str:
    """A condition names a rule that matches itself, or one of its counted options."""
    kind = rule.question.type
    if label is None:
        if rule.matches_itself():
            return ""
        return (
            "names a choice rule without labels; name the option as the "
            "condition's label, or give the rule labels"
        )
    if kind not in OPTION_QUESTION_TYPES:
        return f"is a {kind} question and takes no label"
    if label not in rule.option_keys():
        return f"names label '{label}', which is not an option of the question"
    if label in rule.ignore_labels:
        return f"names label '{label}', which the rule's ignore_labels never count"
    return ""


def _selector_errors(prefix, decision) -> list[ValidationError]:
    algorithm = decision.algorithm
    if algorithm is None or algorithm.systemone is None:
        return []
    field = f"{prefix}.decisions.{decision.name}.algorithm.systemone"
    models = [ref.model for ref in decision.modelRefs or []]
    message = None
    # A model-free built-in Recipe receives its candidates through assignments.
    if models and not MIN_SELECTOR_CANDIDATES <= len(models) <= MAX_SELECTOR_CANDIDATES:
        message = (
            f"needs {MIN_SELECTOR_CANDIDATES}..{MAX_SELECTOR_CANDIDATES} "
            "modelRefs to choose from"
        )
    elif len(set(models)) != len(models):
        message = "requires unique modelRefs"
    elif unknown := sorted(set(algorithm.systemone.candidates) - set(models)):
        message = (
            f"candidates names '{unknown[0]}', which is not one of the "
            "decision's modelRefs"
        )
    return [ValidationError(message, field=field)] if message else []
