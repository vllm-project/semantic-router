"""Checks for model_runtime deployments and the decision signals and selectors that use them."""

import re
from pathlib import PurePosixPath
from urllib.parse import urlparse

from cli.config_contract import iter_condition_leaves, iter_routing_profiles
from cli.models import UserConfig
from cli.validation_error import ValidationError

MODEL_RUNTIME_PROVIDER = "model_runtime"
MODEL_RUNTIME_PROFILES = ("exact", "shared_context", "batching", "max_speed")
MODEL_RUNTIME_DEVICE = re.compile(r"^(auto|cpu|mps|(cuda|rocm|xpu)(:[0-9]+)?)$")
MODEL_RUNTIME_REVISION = re.compile(r"^[0-9a-f]{40}$")
MODEL_RUNTIME_PROCESS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
HUB_REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
INPUT_OVERFLOWS = ("reject", "truncate", "window")
MIN_SELECTOR_CANDIDATES = 2
MAX_SELECTOR_CANDIDATES = 255


def model_runtime_deployment_error(deployment: dict) -> str | None:
    if deployment.get("external_model"):
        return "model_runtime deployments cannot set external_model"
    budget = deployment.get("input") or {}
    if (budget.get("max_tokens") or 0) < 0:
        return "input.max_tokens must not be negative"
    if (budget.get("overflow") or "reject") not in INPUT_OVERFLOWS:
        return "input.overflow must be reject, truncate or window"
    if not MODEL_RUNTIME_DEVICE.match(deployment.get("device") or "auto"):
        return "device must be auto, cpu, mps, cuda[:N], rocm[:N] or xpu[:N]"
    if (deployment.get("profile") or "exact") not in MODEL_RUNTIME_PROFILES:
        return "profile must be one of " + ", ".join(MODEL_RUNTIME_PROFILES)
    revision = deployment.get("revision") or ""
    if revision and not MODEL_RUNTIME_REVISION.match(revision):
        return "revision must be a 40-hex commit"
    endpoint = (deployment.get("endpoint") or "").strip()
    served_name = deployment.get("served_name") or ""
    process = deployment.get("process") or ""
    if endpoint:
        if process:
            return (
                "process groups apply only to managed deployments; "
                "an attached endpoint is one process"
            )
        if served_name and (
            served_name.strip() != served_name or any(c in served_name for c in "\0\n")
        ):
            return "served_name must be a trimmed model name"
        return _endpoint_error(endpoint)
    if served_name:
        return (
            "served_name selects a model on an attached endpoint; "
            "a managed deployment is served under its own name"
        )
    if process and not MODEL_RUNTIME_PROCESS.match(process):
        return "process must be a short name of letters, digits, '.', '_' or '-'"
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
    if budget.get("max_tokens") or (budget.get("overflow") or "reject") != "reject":
        return (
            f"deployment '{name}': decision models reject over-length input and "
            "never truncate; remove input"
        )
    return None


def validate_decision_model_references(
    config: UserConfig, deployments: dict
) -> list[ValidationError]:
    errors = []
    for name, profile in iter_routing_profiles(config):
        prefix = "routing" if name == "default" else f"recipes.{name}.routing"
        for rule in profile.signals.decision or []:
            message = _reference_error(deployments, rule.deployment)
            if message:
                errors.append(
                    ValidationError(
                        message, field=f"{prefix}.signals.decision.{rule.name}"
                    )
                )
        rules = {rule.name: rule for rule in profile.signals.decision or []}
        for decision in profile.decisions:
            errors.extend(_selector_errors(prefix, decision, deployments))
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
        if rule.question.type != "choice" and condition.label is not None:
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' condition '{rule.name}' is a "
                    f"{rule.question.type} question and takes no label",
                    field=field,
                )
            )
        elif (
            rule.question.type == "choice" and condition.label not in rule.option_keys()
        ):
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' choice condition '{rule.name}' "
                    "requires a declared choice key as its label",
                    field=field,
                )
            )
    return errors


def _selector_errors(prefix, decision, deployments) -> list[ValidationError]:
    algorithm = decision.algorithm
    if algorithm is None or algorithm.decision is None:
        return []
    field = f"{prefix}.decisions.{decision.name}.algorithm.decision"
    message = _reference_error(deployments, algorithm.decision.deployment)
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
