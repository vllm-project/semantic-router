"""The CLI refuses the backend ref settings the Router removed, with the
Router's own one-line hint instead of a generic schema error."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.models import REMOVED_BACKEND_REF_FIELDS  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402

ROUTER_SOURCE = (
    REPO_ROOT / "src/semantic-router/pkg/config/removed_backend_ref_fields.go"
)
HINT = "was removed; gateway response fields are accepted by default, so delete it"


def _gateway_config(extra: dict) -> dict:
    return {
        "version": "v0.3",
        "providers": {
            "models": [
                {
                    "name": "claude-via-gateway",
                    "provider_model_id": "example-claude",
                    "api_format": "anthropic",
                    "backend_refs": [
                        {
                            "name": "gateway",
                            "provider": "anthropic-compatible",
                            "base_url": "https://gateway.example.com/anthropic",
                            **extra,
                        }
                    ],
                }
            ]
        },
        "routing": {
            "modelCards": [{"name": "claude-via-gateway"}],
            "decisions": [
                {
                    "name": "default",
                    "priority": 0,
                    "rules": {"operator": "AND", "conditions": []},
                    "modelRefs": [{"model": "claude-via-gateway"}],
                }
            ],
        },
    }


@pytest.mark.parametrize("field", REMOVED_BACKEND_REF_FIELDS)
def test_parser_names_a_removed_gateway_opt_in(tmp_path: Path, field: str) -> None:
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(_gateway_config({field: True}), sort_keys=False))

    with pytest.raises(ConfigParseError) as refused:
        parse_user_config(str(path), log_summary=False)

    assert str(refused.value) == (
        f"providers.models[claude-via-gateway].backend_refs[0]: {field} {HINT}"
    )


def test_gateway_backend_parses_without_a_setting(tmp_path: Path) -> None:
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump(_gateway_config({}), sort_keys=False))

    parsed = parse_user_config(str(path), log_summary=False)

    assert parsed.providers.models[0].backend_refs[0].provider == "anthropic-compatible"


def test_removed_fields_and_hint_match_the_router() -> None:
    source = ROUTER_SOURCE.read_text(encoding="utf-8")
    declared = re.search(r"removedBackendRefFields = \[\]string\{([^}]*)\}", source)
    assert declared is not None
    assert tuple(re.findall(r'"([a-z_]+)"', declared.group(1))) == (
        REMOVED_BACKEND_REF_FIELDS
    )
    assert f'"providers.models[%s].backend_refs[%d]: %s {HINT}"' in source
