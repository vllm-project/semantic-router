"""Model-runtime checks that do not load local or remote model resources."""

from pathlib import PurePosixPath

from cli.model_runtime_defaults import effective_model_deployments
from cli.models import UserConfig
from cli.validation_error import ValidationError
from cli.validator_pii_window import validate_pii_windows

DEVICE_SELECTOR_PARTS = 2


def validate_model_runtime_references(config: UserConfig) -> list[ValidationError]:
    catalog = (config.global_ or {}).get("model_catalog", {})
    guard = catalog.get("modules", {}).get("prompt_guard", {})
    errors = []
    if guard.get("protocol"):
        errors.append(
            ValidationError(
                "prompt_guard.protocol is retired; run 'vllm-sr config migrate' "
                "to select a named backend",
                field="global.model_catalog.modules.prompt_guard.protocol",
            )
        )
    deployments = effective_model_deployments(config)
    external_names = {item.get("name") for item in catalog.get("external", [])}
    for name, deployment in deployments.items():
        message = _deployment_error(name, deployment, external_names)
        if message:
            errors.append(
                ValidationError(
                    message, field=f"global.model_catalog.deployments.{name}"
                )
            )
    profiles = [("routing", config.routing)] + [
        (f"recipes.{recipe.name}.routing", recipe.routing) for recipe in config.recipes
    ]
    for prefix, profile in profiles:
        for decision in profile.decisions:
            for plugin in decision.plugins or []:
                data = plugin.configuration
                if (
                    plugin.type.value == "rag"
                    and data.get("enabled")
                    and data.get("rerank") is not None
                    and "rag.reranker" not in profile.model_bindings
                ):
                    errors.append(
                        ValidationError(
                            "Neural rerank requires recipe-local model_bindings.rag.reranker",
                            field=f"{prefix}.decisions.{decision.name}.plugins",
                        )
                    )
        for consumer, binding in profile.model_bindings.items():
            if binding.deployment not in deployments:
                errors.append(
                    ValidationError(
                        f"Unknown model deployment '{binding.deployment}'",
                        field=f"{prefix}.model_bindings.{consumer}.deployment",
                    )
                )
                continue
            deployment = deployments[binding.deployment]
            message = _binding_error(consumer, binding, deployment, profile)
            if message:
                errors.append(
                    ValidationError(
                        message, field=f"{prefix}.model_bindings.{consumer}"
                    )
                )
    errors.extend(validate_pii_windows(config, deployments))
    return errors


def _binding_error(consumer, binding, deployment, profile=None):
    provider = deployment.get("provider") or "candle"
    if binding.operating_point is not None and not consumer.startswith("classifier."):
        return "operating_point is only supported by generic classifier bindings"
    contracts = {
        "prompt_guard": "label_distribution.v1",
        "domain_classifier": "label_distribution.v1",
        "fact_check_classifier": "label_distribution.v1",
        "feedback_detector": "label_distribution.v1",
        "modality_detector": "label_distribution.v1",
        "pii_classifier": "token_spans.v1",
        "hallucination_detector": "token_spans.v1",
        "hallucination_explainer": "text_pair_distribution.v1",
        "embedding": "embedding.v1",
        "complexity": "score.v1",
        "rag.reranker": "relevance_scores.v1",
    }
    if binding.pair_scorer is not None and consumer != "rag.reranker":
        return "pair_scorer selection is only supported by rag.reranker"
    if consumer == "rag.reranker":
        if provider == "http" or binding.adapter != "vela_reranker":
            return "Reranker requires a local vela_reranker adapter"
        if binding.mapping_path or (provider == "candle" and binding.head):
            return (
                "Reranker has no label mapping or separate Candle classification head"
            )
        if ((deployment.get("input") or {}).get("overflow") or "reject") != "reject":
            return "Reranker requires reject overflow for complete tokenizer pairs"
    if consumer.startswith("safety."):
        matches = []
        for rule in profile.signals.safety if profile else []:
            if consumer == f"safety.{rule.name}":
                matches.append("label_distribution.v1")
            if rule.hazard and consumer == f"safety.{rule.name}.hazard":
                matches.append("label_scores.v1")
        if len(matches) != 1:
            return "Safety binding must identify exactly one head in the same recipe"
        contracts[consumer] = matches[0]
        if binding.mapping_path:
            return "Safety labels define the mapping; mapping_path is not supported"
        if provider == "http" and binding.adapter != "http_classify":
            return "Safety HTTP head requires http_classify adapter"
    if consumer.startswith("classifier."):
        contracts[consumer] = "label_distribution.v1"
        name = consumer.removeprefix("classifier.")
        rule = (
            next(
                (
                    rule
                    for rule in (profile.signals.classifiers or [])
                    if rule.name == name
                ),
                None,
            )
            if profile
            else None
        )
        if rule is None:
            return "Generic classifier binding requires an existing rule in the same recipe"
        if binding.mapping_path:
            return "Generic classifier labels define the mapping; mapping_path is not supported"
        if binding.contract == "label_scores.v1":
            contracts[consumer] = binding.contract
            if (
                binding.operating_point is None
                or provider not in {"candle", "ort"}
                or (provider == "candle" and binding.head)
                or rule.type == "llm"
            ):
                return "Independent scores require operating_point and a complete Candle or qualified ORT artifact"
            budget = deployment.get("input") or {}
            if (
                budget.get("max_tokens", 0) <= 0
                or (budget.get("overflow") or "reject") != "reject"
            ):
                return "operating_point requires an explicit document budget and reject overflow"
            precision = deployment.get("precision") or "native"
            if precision != "native" and (provider != "candle" or precision != "fp32"):
                return "operating_point requires Candle float32 or qualified ORT native execution"
        elif binding.operating_point is not None:
            return "operating_point requires label_scores.v1"
        if rule.type == "llm":
            if provider != "http" or binding.adapter != "http_chat":
                return (
                    "LLM classifier binding requires HTTP http_chat scored extraction"
                )
        elif provider == "http" and binding.adapter != "http_classify":
            return "Sequence classifier binding requires http_classify adapter"
    if consumer not in contracts:
        return f"Unknown task consumer '{consumer}'"
    expected = contracts[consumer]
    if (
        consumer == "prompt_guard"
        and provider == "http"
        and binding.adapter == "http_chat"
    ):
        expected = "label_decision.v1"
    if consumer == "complexity" and binding.contract == "label_distribution.v1":
        expected = binding.contract
    if binding.contract != expected:
        return f"Contract must be '{expected}' for {consumer}"
    if not binding.adapter.strip():
        return "Adapter is required"
    if consumer == "complexity" and provider != "http":
        return "Complexity requires an HTTP score or distribution adapter"
    budget = deployment.get("input") or {}
    if provider == "http":
        if binding.head:
            return "Remote task cannot bind a local head"
        if consumer in {
            "fact_check_classifier",
            "feedback_detector",
            "modality_detector",
            "hallucination_explainer",
        }:
            return f"{consumer} has no HTTP task adapter"
        if consumer != "embedding" and (
            budget.get("max_tokens", 0) != 0
            or (budget.get("overflow") or "reject") != "reject"
        ):
            return (
                "HTTP classifier adapters cannot enforce local tokenizer input budgets"
            )
        if consumer == "hallucination_detector" and binding.adapter != "http_chat":
            return "Hallucination detector requires http_chat adapter"
    if provider == "ort" and consumer in {
        "hallucination_detector",
        "hallucination_explainer",
    }:
        return f"{consumer} has no ORT task adapter"
    return None


def project_classifier_rule(rule, bindings, deployments):
    """Resolve execution selectors without mutating canonical rule policy."""
    binding = bindings.get(f"classifier.{rule.name}")
    if binding is None or binding.deployment not in deployments:
        return rule
    deployment = deployments[binding.deployment]
    updates = {"model": None, "model_path": None, "use_cpu": False}
    if deployment.get("provider") == "http":
        updates["model"] = deployment.get("external_model")
        updates["type"] = "llm" if rule.type == "llm" else "sequence_classifier"
    else:
        updates["type"] = "local"
        updates["model_path"] = deployment.get("artifact")
        updates["use_cpu"] = (deployment.get("device") or "cpu") == "cpu"
    return rule.model_copy(update=updates)


def _deployment_error(name, deployment, external_names):
    if not name or name.strip() != name:
        return "Deployment name must be non-empty and trimmed"
    provider = deployment.get("provider")
    if provider in {"candle", "ort"}:
        if not (deployment.get("artifact") or "").strip() or deployment.get(
            "external_model"
        ):
            return "Local deployment requires artifact and cannot set external_model"
        device = deployment.get("device") or "cpu"
        if device != "cpu":
            parts = device.split(":")
            allowed = {"migraphx", "rocm"} if provider == "ort" else {"cuda", "metal"}
            if (
                len(parts) != DEVICE_SELECTOR_PARTS
                or parts[0] not in allowed
                or not parts[1].isdigit()
            ):
                return f"Device '{device}' is incompatible with provider '{provider}'"
        if (deployment.get("precision") or "native") not in {"native", "fp32", "fp16"}:
            return "Precision must be native, fp32 or fp16"
    elif provider == "http":
        if (
            deployment.get("artifact")
            or not (deployment.get("external_model") or "").strip()
        ):
            return "HTTP deployment requires external_model and cannot set artifact"
        if deployment.get("device") or deployment.get("precision"):
            return (
                "External service device and precision are not controlled by the router"
            )
        if deployment["external_model"] not in external_names:
            return f"Unknown external model '{deployment['external_model']}'"
    else:
        return f"Unsupported provider '{provider}'"
    profile = deployment.get("custom_ops_profile") or "none"
    if profile != "none" and (
        profile != "ck_flash_attention"
        or provider != "ort"
        or not (deployment.get("device") or "cpu").startswith("rocm:")
    ):
        return "custom_ops_profile requires ck_flash_attention on an ORT rocm:index deployment"
    cache_dir = deployment.get("compilation_cache_dir")
    if cache_dir is not None and cache_dir != "":
        if provider != "ort" or not (deployment.get("device") or "cpu").startswith(
            "migraphx:"
        ):
            return "compilation_cache_dir requires an ORT migraphx:index deployment"
        if (
            not isinstance(cache_dir, str)
            or cache_dir.strip() != cache_dir
            or "\x00" in cache_dir
            or not PurePosixPath(cache_dir).is_absolute()
        ):
            return "compilation_cache_dir must be an absolute, trimmed path without null bytes"
    budget = deployment.get("input") or {}
    if budget.get("max_tokens", 0) < 0:
        return "input.max_tokens must not be negative"
    if (budget.get("overflow") or "reject") not in {"reject", "truncate", "window"}:
        return "input.overflow must be reject, truncate or window"
    return None
