"""Opt-in model-backed ROCm graph parity; requires an external verified artifact.

Set the DECISION_QWEN_GRAPH_TEST_* inputs only in an isolated validation run.
No model snapshot, host path, or revision is embedded in this test.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import median

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.artifacts import open_verified_artifact  # noqa: E402
from decision_runtime.catalog_adapter import (  # noqa: E402
    resolve_decision_runtime_model,
)
from decision_runtime.contracts import (  # noqa: E402
    SystemOneBatchRequest,
    SystemOneRequest,
)
from decision_runtime.physical_batching import DecisionRow, _request_rows  # noqa: E402
from decision_runtime.qwen35_rocm_binder import (  # noqa: E402
    assert_qwen_rocm_graph_replay_safe,
)
from decision_runtime.qwen35_torch import _collate  # noqa: E402
from decision_runtime.row_executor import TorchDecisionRowExecutor  # noqa: E402
from decision_runtime.runtime_factory import _load_family  # noqa: E402


def _inputs():
    names = ("MODEL", "REVISION", "ARTIFACT_ROOT", "CONTENT_ID", "REQUEST_PATH")
    values = {
        name: os.environ.get(f"DECISION_QWEN_GRAPH_TEST_{name}") for name in names
    }
    if any(value is None for value in values.values()):
        pytest.skip("external Qwen ROCm graph validation inputs are absent")
    return values


def _rows(request_path: Path, model_id: str):
    raw = json.loads(request_path.read_bytes())
    if raw.get("model") != model_id:
        raise ValueError("sealed test request model differs")
    if "states" not in raw:
        return _request_rows(SystemOneRequest.model_validate(raw))
    request = SystemOneBatchRequest.model_validate(raw)
    return tuple(
        DecisionRow(model_id, state.state, question_id, question)
        for state in request.states
        for question_id, question in request.questions.items()
    )


def _windows(encoded):
    options = []
    for offset in range(0, len(encoded) - 7, 8):
        rows = encoded[offset : offset + 8]
        padded = ((max(row.input_tokens for row in rows) + 31) // 32) * 32
        if padded <= 256:
            layout = (padded, any(row.input_tokens != padded for row in rows))
            options.append((layout, rows))
    for index, (layout, first) in enumerate(options):
        for second_layout, second in options[index + 1 :]:
            if layout == second_layout and tuple(
                row.input_ids for row in first
            ) != tuple(row.input_ids for row in second):
                return first, second
    pytest.fail("configured request has no two changed-content short B8 windows")


def _valid_bits(torch, logits, batch, temperature):
    valid = batch["candidate_mask"]
    values = logits.float()[valid]
    probabilities = (logits.float() / temperature).softmax(-1)
    assert bool(torch.isfinite(values).all())
    assert bool(torch.isfinite(probabilities[valid]).all())
    return values.view(torch.int32), probabilities[valid].view(torch.int32)


def test_short_b8_graph_replays_changed_content_with_bitwise_eager_parity():
    inputs = _inputs()
    model_id = inputs["MODEL"]
    resolved = resolve_decision_runtime_model(
        model_id, revision=inputs["REVISION"], backend="rocm", target="gfx942"
    )
    artifact = open_verified_artifact(
        Path(inputs["ARTIFACT_ROOT"]),
        resolved,
        expected_content_id=inputs["CONTENT_ID"],
    )
    runtime = _load_family(
        resolved,
        artifact,
        "rocm",
        physical_batch_size=8,
        enable_rocm_graph=True,
    )
    assert runtime.rocm_profile_binding is not None
    assert runtime.rocm_graphs is not None
    encoded = TorchDecisionRowExecutor(runtime, resolved.profile)._encode_rows(
        _rows(Path(inputs["REQUEST_PATH"]), model_id)
    )
    first, second = _windows(encoded)
    torch = runtime.torch

    for window in (first, second, first):
        batch = _collate(torch, window, runtime.tokenizer, device=runtime.device)
        with (
            runtime.rocm_graphs.lock,
            torch.inference_mode(),
            torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16),
        ):
            ordinary = runtime.model(**batch)
            candidate = runtime.rocm_graphs.logits(batch)
            torch.cuda.synchronize(runtime.device)
            expected = _valid_bits(torch, ordinary, batch, runtime.temperature)
            actual = _valid_bits(torch, candidate, batch, runtime.temperature)
            assert torch.equal(expected[0], actual[0])
            assert torch.equal(expected[1], actual[1])
        predictions = runtime.predict_encoded(window)
        assert len(predictions) == 8
        assert tuple(item.question_id for item in predictions) == tuple(
            item.question_id for item in window
        )

    assert len(runtime.rocm_graphs._graphs) == 1
    assert runtime.rocm_graphs._capture_attempts == 1

    # Diagnostic only: this source/config/cache guard runs on *every* replay.
    # Its cost belongs in end-to-end HTTP A/B; do not substitute this number.
    padded_tokens = ((max(row.input_tokens for row in first) + 31) // 32) * 32
    guard_samples_ms = []
    for _ in range(30):
        started = time.perf_counter_ns()
        assert_qwen_rocm_graph_replay_safe(
            runtime.rocm_profile_binding,
            physical_batch_size=8,
            padded_tokens=padded_tokens,
        )
        guard_samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    print(f"qwen_rocm_graph_replay_guard_median_ms={median(guard_samples_ms):.6f}")
