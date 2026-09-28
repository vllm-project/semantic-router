from __future__ import annotations

import json

from cli.commands.recipe import recipe
from cli.router_management_client import RouterResponse
from click.testing import CliRunner


def test_recipe_list_includes_collection_etag(monkeypatch) -> None:
    class _Client:
        @staticmethod
        def list_recipes() -> RouterResponse:
            return RouterResponse(
                payload={"etag": '"recipe-etag"', "recipes": []},
                etag='"recipe-etag"',
            )

    monkeypatch.setattr(
        "cli.commands.recipe._client",
        lambda _endpoint, _timeout, _token_env: _Client(),
    )

    result = CliRunner().invoke(recipe, ["list"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "etag": '"recipe-etag"',
        "recipes": [],
    }
