"""Recipe-scoped content safety model reference validation."""

from cli.config_contract import iter_routing_profiles
from cli.config_schema import schema_document
from cli.model_runtime_defaults import effective_model_deployments
from cli.models import UserConfig
from cli.validation_error import ValidationError
from cli.validator_classifier import _external_model_endpoint_errors, _external_models


def validate_safety_contracts(config: UserConfig) -> list[ValidationError]:
    errors: list[ValidationError] = []
    models = _external_models(config)
    catalog = (config.global_ or {}).get("model_catalog") or {}
    local_heads = (catalog.get("modules") or {}).get("safety") or {}
    deployments = effective_model_deployments(config)
    system = catalog.get("system") or {}
    system_keys = schema_document()["$defs"]["CanonicalSystemModels"]["properties"]
    for profile, routing in iter_routing_profiles(config):
        prefix = "routing" if profile == "default" else f"recipes.{profile}.routing"
        for rule in routing.signals.safety:
            heads = [
                (rule.model, "safety", f"{prefix}.signals.safety.{rule.name}.model")
            ]
            if rule.hazard:
                heads.append(
                    (
                        rule.hazard.model,
                        "hazard",
                        f"{prefix}.signals.safety.{rule.name}.hazard.model",
                    )
                )
            for declared_name, kind, field in heads:
                name = declared_name
                local = local_heads.get(kind) or {}
                consumer = f"safety.{rule.name}" + (
                    ".hazard" if kind == "hazard" else ""
                )
                binding = routing.model_bindings.get(consumer)
                if binding:
                    deployment = deployments.get(binding.deployment)
                    if not deployment:
                        continue  # The shared binding validator reports this reference.
                    if deployment.get("provider") == "http":
                        if not name and local.get("window") is not None:
                            errors.append(
                                ValidationError(
                                    "Remote safety head cannot use local token windows",
                                    field=field,
                                )
                            )
                        name = deployment.get("external_model")
                    else:
                        local = {
                            **local,
                            "model_id": deployment.get("artifact"),
                            "max_sequence_length": (deployment.get("input") or {}).get(
                                "max_tokens", 0
                            ),
                        }
                        if name:
                            local.pop("window", None)
                        name = None
                if not name:
                    # Omitted heads use the built-in module. Validate explicit
                    # deployment overrides here; artifact/label compatibility
                    # is checked by the native loader at runtime initialization.
                    limit = local.get("max_sequence_length", 0)
                    if (
                        not isinstance(limit, int)
                        or isinstance(limit, bool)
                        or limit < 0
                    ):
                        errors.append(
                            ValidationError(
                                "Safety max_sequence_length must be a nonnegative integer (0 uses the default)",
                                field=f"global.model_catalog.modules.safety.{kind}.max_sequence_length",
                            )
                        )
                    elif local.get("window") is not None:
                        window = local["window"]
                        size, overlap = window.get("size", 0), window.get("overlap", 0)
                        if (
                            not isinstance(size, int)
                            or isinstance(size, bool)
                            or size <= 0
                            or size > (limit or 512)
                            or not isinstance(overlap, int)
                            or isinstance(overlap, bool)
                            or overlap < 0
                            or overlap >= size
                        ):
                            errors.append(
                                ValidationError(
                                    "Safety window must fit the effective input budget with nonnegative overlap smaller than size",
                                    field=field,
                                )
                            )
                    ref = local.get("model_ref", kind)
                    if not local.get("model_id") and ref and ref not in system_keys:
                        errors.append(
                            ValidationError(
                                f"Unknown safety model_ref '{ref}'", field=field
                            )
                        )
                    elif (
                        not local.get("model_id") and ref in system and not system[ref]
                    ):
                        errors.append(
                            ValidationError(
                                f"Safety model_ref '{ref}' has no configured artifact",
                                field=field,
                            )
                        )
                    continue
                model = models.get(name)
                if model is None:
                    errors.append(
                        ValidationError(
                            f"Safety '{rule.name}' references unknown external model '{name}'",
                            field=field,
                        )
                    )
                elif model.get("model_role") != "classification":
                    errors.append(
                        ValidationError(
                            "Safety models must have model_role=classification",
                            field=field,
                        )
                    )
                else:
                    errors.extend(
                        _external_model_endpoint_errors(
                            rule.name,
                            model,
                            field,
                            classifier_kind="Safety classifier",
                            require_model_name=False,
                        )
                    )
    return errors
