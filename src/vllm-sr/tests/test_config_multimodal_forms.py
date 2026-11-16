"""The CLI refuses the removed multimodal forms with the Router's own lines,
and the Router's migration leaves none of them behind."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.config_multimodal_forms import removed_multimodal_forms  # noqa: E402
from cli.models import DecisionOutput  # noqa: E402

GOLDEN = REPO_ROOT / "src/semantic-router/pkg/config/testdata/migrate"
CASES = (
    "removed-field-model-modality",
    "removed-field-image-route",
    "removed-field-image-route-two-models",
    "removed-field-image-route-chat-api",
    "removed-plugin-image-gen",
)
# The cases without a mechanical rewrite have no after.yaml.
MIGRATED = tuple(case for case in CASES if (GOLDEN / case / "after.yaml").exists())


def _golden(case: str, name: str) -> str:
    return (GOLDEN / case / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("case", CASES)
def test_the_cli_refuses_a_removed_multimodal_form_with_the_routers_line(case):
    before = yaml.safe_load(_golden(case, "before.yaml"))
    assert _golden(case, "hint.txt").strip() in removed_multimodal_forms(before)


@pytest.mark.parametrize("case", MIGRATED)
def test_the_routers_migration_leaves_no_removed_multimodal_form(case):
    after = yaml.safe_load(_golden(case, "after.yaml"))
    assert removed_multimodal_forms(after) == []


def test_a_canonical_image_route_is_not_a_removed_form():
    config = {
        "providers": {
            "models": [
                {
                    "name": "painter",
                    "capabilities": ["image_generation"],
                    "modalities": {"input": ["text"], "output": ["image"]},
                }
            ]
        },
        "recipes": [
            {
                "name": "default",
                "decisions": [
                    {
                        "name": "image",
                        "rules": {"type": "modality", "name": "wants_image"},
                        "modelRefs": [{"model": "painter"}],
                    }
                ],
            }
        ],
    }
    assert removed_multimodal_forms(config) == []


@pytest.mark.parametrize(
    "output",
    [{"kind": "image", "action": "generate"}, {"kind": ["text", "image"]}, {}],
)
def test_an_output_contract_loads(output):
    DecisionOutput.model_validate(output)


@pytest.mark.parametrize(
    "output",
    [
        {"kind": "picture"},
        {"kind": ["image", "image"]},
        {"kind": []},
        {"kind": "text", "action": "edit"},
        {"kind": "image", "action": "draw"},
    ],
)
def test_an_output_contract_names_only_known_outputs(output):
    with pytest.raises(ValueError):
        DecisionOutput.model_validate(output)
