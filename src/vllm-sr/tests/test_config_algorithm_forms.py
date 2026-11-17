"""The removed algorithm forms as the CLI reads them: refused with the
Router's own lines, and gone after `vllm-sr config migrate`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_algorithm_forms import removed_algorithm_forms  # noqa: E402
from cli.parser import ConfigParseError, parse_user_config_data  # noqa: E402

GOLDEN = REPO_ROOT / "src/semantic-router/pkg/config/testdata/migrate"
CASES = sorted(
    path.name
    for path in GOLDEN.iterdir()
    if path.name.startswith("removed-algorithm-")
    or path.name
    in {"removed-field-executor-on-error", "removed-field-selector-on-error"}
)


def _golden(case: str, name: str) -> dict:
    return yaml.safe_load((GOLDEN / case / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES)
def test_parser_refuses_a_removed_algorithm_form_with_the_router_line(
    case: str,
) -> None:
    hint = (GOLDEN / case / "hint.txt").read_text(encoding="utf-8").strip()

    with pytest.raises(ConfigParseError) as refused:
        parse_user_config_data(_golden(case, "before.yaml"), log_summary=False)

    assert hint in str(refused.value).splitlines()


@pytest.mark.parametrize("case", CASES)
def test_migrated_algorithms_have_no_removed_forms(case: str) -> None:
    assert removed_algorithm_forms(_golden(case, "after.yaml")) == []


def test_lines_follow_the_routers_order_in_every_routing_scope() -> None:
    document = {
        "global": {"router": {"model_selection": {"enabled": True, "ml": {}}}},
        "recipes": [
            {
                "name": "default",
                "decisions": [
                    {
                        "name": "pinned",
                        "algorithm": {"type": "static", "on_error": "skip"},
                    },
                    {
                        "name": "fused",
                        "algorithm": {"type": "fusion", "fusion": {"on_error": "skip"}},
                    },
                ],
            },
            {
                "name": "native",
                "decisions": [{"name": "cheap", "algorithm": {"type": "knn"}}],
            },
        ],
        "entrypoints": [
            {"model_names": ["vllm-sr/auto"], "recipe": "default", "default": True}
        ],
    }

    lines = [line.split(": ", 2)[:2] for line in removed_algorithm_forms(document)]

    assert lines == [
        [
            "recipes[native].decisions[cheap].algorithm",
            "removed_algorithm_type",
        ],
        ["recipes[default].decisions[fused].algorithm", "removed_field"],
        ["recipes[default].decisions[pinned].algorithm", "removed_field"],
        ["global.router.model_selection.ml", "removed_field"],
    ]
