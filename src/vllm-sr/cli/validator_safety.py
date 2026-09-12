"""Recipe-scoped content safety model reference validation."""

from cli.config_contract import iter_routing_profiles
from cli.config_schema import schema_document
from cli.models import UserConfig
from cli.validation_error import ValidationError
from cli.validator_classifier import _external_model_endpoint_errors, _external_models


def validate_safety_contracts(config: UserConfig) -> list[ValidationError]:
    errors: list[ValidationError] = []
    models = _external_models(config)
    catalog = (config.global_ or {}).get("model_catalog") or {}
    local_heads = (catalog.get("modules") or {}).get("safety") or {}
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
            for name, kind, field in heads:
                if not name:
                    # Omitted heads use the built-in module. Validate explicit
                    # deployment overrides here; artifact/label compatibility
                    # is checked by the native loader at runtime initialization.
                    local = local_heads.get(kind) or {}
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
