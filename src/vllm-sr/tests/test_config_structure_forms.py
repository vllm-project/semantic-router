"""The CLI refuses the removed global switches with the Router's own lines, and
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

from cli.config_structure_forms import removed_structure_forms  # noqa: E402

GOLDEN = REPO_ROOT / "src/semantic-router/pkg/config/testdata/migrate"
CASES = (
    "removed-field-response-cache-switch",
    "removed-field-memory-switch",
    "removed-field-router-replay-switch",
    "removed-section-prompt-compression",
)


def _golden(case: str, name: str) -> str:
    return (GOLDEN / case / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("case", CASES)
def test_the_cli_refuses_a_removed_switch_with_the_routers_line(case):
    before = yaml.safe_load(_golden(case, "before.yaml"))
    assert _golden(case, "hint.txt").strip() in removed_structure_forms(before)


@pytest.mark.parametrize("case", CASES)
def test_the_routers_migration_leaves_no_removed_switch(case):
    assert removed_structure_forms(yaml.safe_load(_golden(case, "after.yaml"))) == []
