"""Checks for model_runtime deployments and the decision signals and selectors that use them."""

import re
from pathlib import PurePosixPath
from urllib.parse import urlparse

from cli.config_contract import iter_condition_leaves, iter_routing_profiles
from cli.decision_model import (
    DECISION_MODEL_FIELD,
    configured_decision_model,
    decision_model_deployment,
)
from cli.models import UserConfig
from cli.models_decision import OPTION_QUESTION_TYPES
from cli.validation_error import ValidationError

MODEL_RUNTIME_PROVIDER = "model_runtime"
# The runtime owns profile and accelerator names, plugins' included: the
# validator checks their shape and leaves an unknown name to the runtime.
MODEL_RUNTIME_PROFILE = re.compile(r"^[a-z][a-z0-9_]*$")
MODEL_RUNTIME_DEVICE = re.compile(r"^[a-z][a-z0-9_]*(:[0-9]+)?$")
MODEL_RUNTIME_REVISION = re.compile(r"^[0-9a-f]{40}$")
HUB_REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
INPUT_OVERFLOWS = ("reject", "truncate", "window")
MIN_SELECTOR_CANDIDATES = 2
MAX_SELECTOR_CANDIDATES = 255
MAX_MODEL_REPLICAS = 64


def public_model_name(deployment: dict) -> str:
    artifact = deployment.get("artifact") or ""
    return deployment.get("public_name") or (
        artifact
        if not artifact.startswith(("models/", "./", "../"))
        and HUB_REPOSITORY_ID.fullmatch(artifact)
        else ""
    )


def model_runtime_deployment_error(deployment: dict) -> str | None:
    if "process" in deployment:
        return "process is retired; use replicas for independent model workers"
    public_name = deployment.get("public_name") or ""
    if public_name and (
        public_name.strip() != public_name
        or any(c in public_name for c in "\0\r\n\t ")
        or public_name.startswith("/")
        or "://" in public_name
    ):
        return "public_name must be a trimmed public model ID, not a path or endpoint"
    if deployment.get("external_model"):
        return "model_runtime deployments cannot set external_model"
    budget = deployment.get("input") or {}
    if (budget.get("max_tokens") or 0) < 0:
        return "input.max_tokens must not be negative"
    if (budget.get("overflow") or "reject") not in INPUT_OVERFLOWS:
        return "input.overflow must be reject, truncate or window"
    if not MODEL_RUNTIME_PROFILE.match(deployment.get("profile") or "exact"):
        return "profile must be a profile name, such as exact or batching"
    revision = deployment.get("revision") or ""
    if revision and not MODEL_RUNTIME_REVISION.match(revision):
        return "revision must be a 40-hex commit"
    replicas = deployment.get("replicas")
    if replicas is not None and not isinstance(replicas, list):
        return "replicas must be a list"
    if replicas:
        if len(replicas) > MAX_MODEL_REPLICAS:
            return f"replicas may contain at most {MAX_MODEL_REPLICAS} workers"
        if any(deployment.get(key) for key in ("device", "endpoint", "served_name")):
            return "replicas cannot be combined with top-level device, endpoint or served_name"
        attached = set()
        for index, replica in enumerate(replicas):
            if not isinstance(replica, dict) or set(replica) - {
                "device",
                "endpoint",
                "served_name",
            }:
                return f"replicas[{index}] may contain only device, endpoint and served_name"
            message = _placement_error(replica)
            if message:
                return f"replicas[{index}]: {message}"
            endpoint = replica.get("endpoint") or ""
            if endpoint:
                identity = (endpoint.rstrip("/"), replica.get("served_name") or "")
                if identity in attached:
                    return "duplicate attached replica endpoint and served_name"
                attached.add(identity)
    else:
        message = _placement_error(deployment)
        if message:
            return message
        if deployment.get("endpoint"):
            return None
    artifact = (deployment.get("artifact") or "").strip()
    if not artifact:
        return (
            "a managed model_runtime deployment requires artifact "
            "(a Hub repository or an absolute package path)"
        )
    absolute = PurePosixPath(artifact).is_absolute()
    if not absolute and not HUB_REPOSITORY_ID.match(artifact):
        return "artifact must be a Hub repository ID or an absolute package path"
    if absolute and revision:
        return "revision applies only to Hub repositories"
    return None


def _placement_error(placement: dict) -> str | None:
    device = placement.get("device") or "auto"
    if not isinstance(device, str) or not MODEL_RUNTIME_DEVICE.fullmatch(device):
        return "device must be an accelerator name with an optional index, such as cpu, cuda:0 or rocm:1"
    endpoint = placement.get("endpoint") or ""
    served_name = placement.get("served_name") or ""
    if endpoint:
        if placement.get("device"):
            return "attached endpoints cannot set device"
        if served_name and (
            served_name.strip() != served_name
            or any(c in served_name for c in "\0\r\n")
        ):
            return "served_name must be a trimmed model name"
        return _endpoint_error(endpoint)
    if served_name:
        return "served_name selects a model on an attached endpoint"
    return None


def _endpoint_error(endpoint: str) -> str | None:
    parsed = urlparse(endpoint)
    if parsed.scheme == "unix":
        if parsed.netloc or not PurePosixPath(parsed.path).is_absolute():
            return (
                "endpoint unix:// needs an absolute socket path "
                "(unix:///run/vllm-sr/runtime.sock)"
            )
        return None
    if parsed.scheme in {"http", "https"}:
        return None if parsed.netloc else "endpoint needs a host"
    return "endpoint must use unix://, http:// or https://"


def _reference_error(deployments: dict, name: str) -> str | None:
    deployment = deployments.get(name)
    if deployment is None:
        return (
            f"deployment '{name}' is not declared in global.model_catalog.deployments"
        )
    if deployment.get("provider") != MODEL_RUNTIME_PROVIDER:
        return f"deployment '{name}' must use provider {MODEL_RUNTIME_PROVIDER}"
    budget = deployment.get("input") or {}
    max_tokens = budget.get("max_tokens") or 0
    overflow = budget.get("overflow") or "reject"
    if not (
        (max_tokens == 0 and overflow == "reject")
        or (max_tokens > 0 and overflow == "window")
    ):
        return (
            f"deployment '{name}': decision models reject over-length input and "
            "never truncate; use input.overflow: window with max_tokens, or omit input"
        )
    return None


def _decision_model(config: UserConfig) -> tuple[str | None, list[ValidationError]]:
    """The configured decision model, canonical, or the error that names it."""

    try:
        document = {"global": config.global_ or {}}
        key = configured_decision_model(document)
        decision_model_deployment(document)
        return key, []
    except ValueError as error:
        return None, [ValidationError(str(error), field=DECISION_MODEL_FIELD)]


def validate_decision_model_references(
    config: UserConfig, deployments: dict
) -> list[ValidationError]:
    decision_model, errors = _decision_model(config)
    for name, profile in iter_routing_profiles(config):
        prefix = "routing" if name == "default" else f"recipes.{name}.routing"
        for rule in profile.signals.decision or []:
            if rule.deployment:
                message = _reference_error(deployments, rule.deployment)
            else:
                message = (
                    _reference_error(deployments, decision_model)
                    if decision_model
                    else None
                )
            if message:
                errors.append(
                    ValidationError(
                        message, field=f"{prefix}.signals.decision.{rule.name}"
                    )
                )
        rules = {rule.name: rule for rule in profile.signals.decision or []}
        errors.extend(
            _set_label_answer_errors(prefix, list(rules.values()), decision_model)
        )
        for decision in profile.decisions:
            errors.extend(
                _selector_errors(prefix, decision, deployments, decision_model)
            )
            errors.extend(_condition_errors(prefix, decision, rules))
    return errors


def _condition_errors(prefix, decision, rules) -> list[ValidationError]:
    errors = []
    field = f"{prefix}.decisions.{decision.name}.rules.conditions"
    for condition in iter_condition_leaves(decision.rules.conditions):
        if (condition.type or "").strip().lower() != "decision":
            continue
        rule = rules.get(condition.name or "")
        if rule is None:
            continue
        kind = rule.question.type
        if kind not in OPTION_QUESTION_TYPES and condition.label is not None:
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' condition '{rule.name}' is a "
                    f"{kind} question and takes no label",
                    field=field,
                )
            )
        elif (
            kind in OPTION_QUESTION_TYPES and condition.label not in rule.option_keys()
        ):
            option = "choice key" if kind == "choice" else "label"
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' {kind} condition '{rule.name}' "
                    f"requires a declared {option} as its label",
                    field=field,
                )
            )
    return errors


def _set_label_answer_errors(prefix, rules, decision_model) -> list[ValidationError]:
    """A rule named like a set label's answer key ("<rule>.<label>") on the same deployment."""
    names = {(rule.deployment or decision_model or "", rule.name) for rule in rules}
    errors = []
    for rule in rules:
        if rule.question.type != "set":
            continue
        for label in rule.question.labels:
            other = f"{rule.name}.{label.key}"
            if (rule.deployment or decision_model or "", other) in names:
                errors.append(
                    ValidationError(
                        f"the name collides with the answer key of set question "
                        f"'{rule.name}''s label '{label.key}' on deployment "
                        f"'{rule.deployment or 'of the decision model'}'; rename one of them",
                        field=f"{prefix}.signals.decision.{other}",
                    )
                )
    return errors


def _selector_errors(
    prefix, decision, deployments, decision_model
) -> list[ValidationError]:
    algorithm = decision.algorithm
    if algorithm is None or algorithm.decision is None:
        return []
    field = f"{prefix}.decisions.{decision.name}.algorithm.decision"
    if algorithm.decision.deployment:
        message = _reference_error(deployments, algorithm.decision.deployment)
    else:
        message = (
            _reference_error(deployments, decision_model) if decision_model else None
        )
    models = [ref.model for ref in decision.modelRefs or []]
    if message is None and not (
        MIN_SELECTOR_CANDIDATES <= len(models) <= MAX_SELECTOR_CANDIDATES
    ):
        message = (
            f"needs {MIN_SELECTOR_CANDIDATES}..{MAX_SELECTOR_CANDIDATES} "
            "modelRefs to choose from"
        )
    if message is None and len(set(models)) != len(models):
        message = "requires unique modelRefs"
    unknown = sorted(set(algorithm.decision.candidates) - set(models))
    if message is None and unknown:
        message = (
            f"candidates names '{unknown[0]}', which is not one of the "
            "decision's modelRefs"
        )
    return [ValidationError(message, field=field)] if message else []


def validate_systemone_listener_models(
    config: UserConfig, deployments: dict
) -> list[ValidationError]:
    """Check the separate native API scope without borrowing Chat permissions."""
    identities = {}
    for key, deployment in deployments.items():
        public_name = public_model_name(deployment)
        if public_name:
            identities.setdefault(public_name, []).append((key, deployment))
    errors = []
    native_names = {
        model.name
        for model in config.providers.models
        if model.api_format == "systemone"
    }
    native_names.update(
        name
        for entry in config.entrypoints
        if entry.api == "systemone"
        for name in entry.model_names
    )
    for listener in config.listeners:
        if listener.systemone is None:
            continue
        for public_name in listener.systemone.models:
            if public_name in native_names:
                continue
            candidates = identities.get(public_name, [])
            message = None
            if not candidates:
                message = f"public System One model {public_name!r} is not declared"
            elif len(candidates) != 1:
                message = f"public System One model {public_name!r} names multiple deployments; assign distinct public_name values"
            elif candidates[0][1].get("provider") != MODEL_RUNTIME_PROVIDER:
                message = f"public System One model {public_name!r} requires a model_runtime deployment"
            if message:
                errors.append(
                    ValidationError(
                        message, field=f"listeners.{listener.name}.systemone.models"
                    )
                )
    return errors
