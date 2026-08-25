from __future__ import annotations

import pytest
from cli.container_management_listener import _validate_management_access


def test_enabled_management_listener_requires_router_native_auth() -> None:
    _validate_management_access(
        {"remote_exposure": True, "auth": {"mode": "router"}},
        True,
    )

    with pytest.raises(ValueError, match="requires auth mode router"):
        _validate_management_access({"auth": {"mode": "disabled"}}, True)


@pytest.mark.parametrize(
    "auth",
    [
        {"mode": "router", "tokens": [{"env": "TOKEN", "role": "admin"}]},
        {"mode": "router", "roles": {"admin": ["*"]}},
    ],
)
def test_router_authenticated_listener_rejects_static_authority(auth: dict) -> None:
    with pytest.raises(ValueError, match=r"does not accept.*tokens or roles"):
        _validate_management_access({"auth": auth}, True)


def test_file_backed_listener_rejects_router_native_auth() -> None:
    with pytest.raises(ValueError, match="file-backed management API auth mode"):
        _validate_management_access({"auth": {"mode": "router"}}, False)
