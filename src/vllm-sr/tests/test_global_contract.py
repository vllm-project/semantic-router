from __future__ import annotations

import pytest

from cli.global_contract import _GLOBAL_OBJECT_FIELDS, validate_global_structure
from cli.models import UserConfig
from pydantic import ValidationError

from test_human_authoring_contract import human_config


@pytest.mark.parametrize(
    "segments",
    [segments for segments, _ in _GLOBAL_OBJECT_FIELDS],
    ids=lambda segments: ".".join(("global", *segments)),
)
def test_final_global_object_boundaries_reject_unknown_fields(
    segments: tuple[str, ...],
) -> None:
    document: dict[str, object] = {}
    target = document
    for segment in segments:
        child: dict[str, object] = {}
        target[segment] = child
        target = child
    target["unexpected"] = True

    with pytest.raises(ValueError, match=r"global.*unexpected"):
        validate_global_structure(document)


@pytest.mark.parametrize(
    "segments",
    [segments for segments, _ in _GLOBAL_OBJECT_FIELDS if segments],
    ids=lambda segments: ".".join(("global", *segments)),
)
def test_final_global_object_boundaries_reject_scalar_shapes(
    segments: tuple[str, ...],
) -> None:
    document: dict[str, object] = {}
    target = document
    for segment in segments[:-1]:
        child: dict[str, object] = {}
        target[segment] = child
        target = child
    target[segments[-1]] = "not-an-object"

    with pytest.raises(ValueError, match=r"must be an object"):
        validate_global_structure(document)


def test_user_config_applies_nested_global_contract() -> None:
    payload = human_config()
    payload["global"]["services"] = {"access": {"enforcement": {"unexpected": "value"}}}

    with pytest.raises(
        ValidationError,
        match=r"global\.services\.access\.enforcement\.unexpected",
    ):
        UserConfig.model_validate(payload)


def test_named_backend_credentials_use_the_final_nested_contract() -> None:
    payload = human_config()
    payload["global"]["services"] = {
        "backend_credentials": {
            "private": {
                "credential_adapter_id": "bearer",
                "secret_env": "PRIVATE_API_KEY",
                "unexpected": True,
            }
        }
    }

    with pytest.raises(
        ValidationError,
        match=r"global\.services\.backend_credentials\.private\.unexpected",
    ):
        UserConfig.model_validate(payload)
