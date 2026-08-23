from __future__ import annotations

import pytest
from cli.container_management_listener import _validate_management_access


def test_managed_listener_requires_router_native_auth() -> None:
    _validate_management_access(
        {"remote_exposure": True, "auth": {"mode": "router"}}, "managed"
    )

    with pytest.raises(ValueError, match="requires management API auth mode router"):
        _validate_management_access({"auth": {"mode": "disabled"}}, "managed")


@pytest.mark.parametrize(
    "auth",
    [
        {"mode": "router", "tokens": [{"env": "TOKEN", "role": "admin"}]},
        {"mode": "router", "roles": {"admin": ["*"]}},
    ],
)
def test_managed_listener_rejects_static_authority(auth: dict) -> None:
    with pytest.raises(ValueError, match="does not accept.*tokens or roles"):
        _validate_management_access({"auth": auth}, "managed")


def test_standalone_listener_rejects_router_native_auth() -> None:
    with pytest.raises(ValueError, match="standalone management API auth mode"):
        _validate_management_access({"auth": {"mode": "router"}}, "standalone")
