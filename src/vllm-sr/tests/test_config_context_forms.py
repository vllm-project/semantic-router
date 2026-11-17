"""The CLI refuses the removed context forms with the Router's own lines, and
the Router's migration leaves none of them behind."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_context_forms import removed_context_forms  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config  # noqa: E402

GOLDEN = REPO_ROOT / "src/semantic-router/pkg/config/testdata/migrate"
CASES = (
    "removed-field-rag-min-confidence",
    "removed-field-rag-recall",
    "removed-field-memory-recall",
    "removed-field-compression-scoring",
    "removed-plugin-rag",
    "removed-plugin-memory",
    "removed-plugin-context-compression",
    "removed-field-rag-reranker-binding",
    "invalid-value-rerank-provider",
    "invalid-value-rag-mcp-backend",
)


def _golden(case: str, name: str) -> str:
    return (GOLDEN / case / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("case", CASES)
def test_the_cli_refuses_a_removed_context_form_with_the_routers_line(case):
    before = yaml.safe_load(_golden(case, "before.yaml"))
    assert _golden(case, "hint.txt").strip() in removed_context_forms(before)


@pytest.mark.parametrize("case", CASES)
def test_the_routers_migration_leaves_no_removed_context_form(case):
    assert removed_context_forms(yaml.safe_load(_golden(case, "after.yaml"))) == []


def test_parsing_refuses_a_removed_context_form(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(_golden("removed-plugin-rag", "before.yaml"), encoding="utf-8")
    with pytest.raises(
        ConfigParseError,
        match="removed_plugin_type: plugin rag is removed",
    ):
        parse_user_config(str(path), log_summary=False)
