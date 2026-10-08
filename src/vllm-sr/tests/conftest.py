"""Shared fixtures for the CLI unit tests."""

import pytest
from cli import router_validation


@pytest.fixture(autouse=True)
def _no_router_image_validation(monkeypatch, request):
    """Keep `config validate` from running a local Router image in unit tests.

    Tests that cover the Router's validation mark themselves `router_image`.
    """

    if request.node.get_closest_marker("router_image"):
        return

    def unavailable(_explicit=None):
        raise router_validation.RouterValidationUnavailableError(
            "Router image validation is off in unit tests"
        )

    monkeypatch.setattr(router_validation, "validation_image", unavailable)
