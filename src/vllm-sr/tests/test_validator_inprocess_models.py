"""`vllm-sr validate` mirrors the Router's backends check for in-process model calls."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.parser import parse_user_config  # noqa: E402
from cli.validator import validate_user_config  # noqa: E402

# Behind an external gateway (`listeners: []`) a provider model may go without
# backend_refs; the Router still calls a Looper's models itself.
CONFIG = """
version: v0.3
listeners: []
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs:
        - name: a-backend
          provider: vllm
          endpoint: 127.0.0.1:8000
    - name: b
      provider_model_id: b
routing:
  modelCards:
    - name: a
      loras:
        - name: a-lora
    - name: b
  decisions:
    - name: panel
      priority: 10
      rules:
        operator: AND
        conditions: []
{decision}
"""


def _errors(tmp_path, decision: str) -> list[str]:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG.format(decision=decision))
    errors = validate_user_config(parse_user_config(str(path)), log_summary=False)
    return [error.message for error in errors if "in process" in error.message]


def _refused(model: str) -> str:
    return (
        f'decision "panel": the Router calls model "{model}" in process, but it has '
        "no backend; give it providers.models[].backend_refs"
    )


@pytest.mark.parametrize(
    ("decision", "missing"),
    [
        pytest.param(
            """      modelRefs: [{model: a}, {model: b}]
      algorithm: {type: ratings}""",
            "b",
            id="a Looper candidate",
        ),
        pytest.param(
            """      modelRefs: [{model: a}]
      algorithm:
        type: fusion
        fusion: {model: b, analysis_models: [a]}""",
            "b",
            id="the Fusion judge",
        ),
        pytest.param(
            """      modelRefs: [{model: a}]
      algorithm:
        type: workflows
        workflows: {mode: dynamic, planner: {model: b}}""",
            "b",
            id="the Flow planner",
        ),
        pytest.param(
            """      modelRefs: [{model: a}, {model: b}]
      algorithm:
        type: prompt
        prompt: {model: b, instructions: Pick one.}""",
            "b",
            id="the prompt helper",
        ),
        pytest.param(
            """      modelRefs: [{model: b}]
      plugins:
        - type: context_compression
          configuration: {enabled: true, recovery: {enabled: true, store: response_cache}}""",
            "b",
            id="context recovery",
        ),
    ],
)
def test_a_model_called_in_process_needs_backend_refs(tmp_path, decision, missing):
    assert _errors(tmp_path, decision) == [_refused(missing)]


@pytest.mark.parametrize(
    "decision",
    [
        pytest.param(
            """      modelRefs: [{model: a}, {model: a-lora}]
      algorithm: {type: ratings}""",
            id="a LoRA of a served model",
        ),
        pytest.param(
            """      modelRefs: [{model: a}, {model: b}]
      algorithm: {type: static}""",
            id="a decision that only routes",
        ),
    ],
)
def test_the_router_needs_no_backend_for_models_it_does_not_call(tmp_path, decision):
    assert _errors(tmp_path, decision) == []
