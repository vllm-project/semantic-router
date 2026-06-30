"""Validation helpers for workflows algorithm configuration."""

from cli.validation_error import ValidationError


def validate_static_workflow_roles(decision, workflows_cfg) -> list[ValidationError]:
    errors: list[ValidationError] = []
    roles = workflows_cfg.roles or []
    if not roles:
        return [
            ValidationError(
                f"Decision '{decision.name}' uses workflows mode=static but does not set roles",
                field=f"decisions.{decision.name}.algorithm.workflows.roles",
            )
        ]

    allowed_models = {
        model_ref.model for model_ref in getattr(decision, "modelRefs", []) or []
    }
    previous_access_ids: set[str] = set()
    for role_index, role in enumerate(roles):
        role_id = _workflow_role_id(role.name, role_index)
        if not role.name or not role.name.strip():
            errors.append(
                ValidationError(
                    f"Decision '{decision.name}' workflows role at index {role_index} has empty name",
                    field=f"decisions.{decision.name}.algorithm.workflows.roles.{role_index}.name",
                )
            )
        seen_models: set[str] = set()
        for model_index, model in enumerate(role.models or []):
            if not model or not model.strip():
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' workflows role '{role.name}' has empty model",
                        field=f"decisions.{decision.name}.algorithm.workflows.roles.{role_index}.models.{model_index}",
                    )
                )
                continue
            if model in seen_models:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' workflows role '{role.name}' duplicates model '{model}'",
                        field=f"decisions.{decision.name}.algorithm.workflows.roles.{role_index}.models",
                    )
                )
            seen_models.add(model)
            if model not in allowed_models:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' workflows role '{role.name}' references model '{model}' outside modelRefs",
                        field=f"decisions.{decision.name}.algorithm.workflows.roles.{role_index}.models.{model_index}",
                    )
                )

        seen_access: set[str] = set()
        for access_index, access_target in enumerate(role.access_list or []):
            if not access_target or not access_target.strip():
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' workflows role '{role.name}' has empty access_list entry",
                        field=f"decisions.{decision.name}.algorithm.workflows.roles.{role_index}.access_list.{access_index}",
                    )
                )
                continue
            normalized_target = access_target.strip()
            if normalized_target in seen_access:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' workflows role '{role.name}' duplicates access_list target '{normalized_target}'",
                        field=f"decisions.{decision.name}.algorithm.workflows.roles.{role_index}.access_list",
                    )
                )
            seen_access.add(normalized_target)
            if normalized_target not in previous_access_ids:
                errors.append(
                    ValidationError(
                        f"Decision '{decision.name}' workflows role '{role.name}' access_list references unknown or future role/agent '{normalized_target}'",
                        field=f"decisions.{decision.name}.algorithm.workflows.roles.{role_index}.access_list.{access_index}",
                    )
                )
        _register_workflow_access_ids(previous_access_ids, role_id, role.models or [])

    return errors


def validate_workflow_final_model(decision, workflows_cfg) -> list[ValidationError]:
    final_model = getattr(workflows_cfg.final, "model", None)
    if not final_model:
        return []
    allowed_models = {
        model_ref.model for model_ref in getattr(decision, "modelRefs", []) or []
    }
    if final_model in allowed_models:
        return []
    return [
        ValidationError(
            f"Decision '{decision.name}' workflows final model '{final_model}' is outside modelRefs",
            field=f"decisions.{decision.name}.algorithm.workflows.final.model",
        )
    ]


def _workflow_role_id(name: str | None, index: int) -> str:
    role_id = (name or "").strip().lower().replace(" ", "-").replace("_", "-")
    if role_id:
        return role_id
    return f"role-{index + 1}"


def _register_workflow_access_ids(
    access_ids: set[str], role_id: str, models: list[str]
) -> None:
    access_ids.add(role_id)
    for model_index, model in enumerate(models):
        normalized_model = (model or "").strip()
        if normalized_model:
            access_ids.add(_workflow_agent_id(role_id, model_index, normalized_model))


def _workflow_agent_id(role_id: str, model_index: int, model: str) -> str:
    return f"{role_id}:{model_index}:{model}"
