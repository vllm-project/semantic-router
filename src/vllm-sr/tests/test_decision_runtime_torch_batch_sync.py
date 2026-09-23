"""Dependency-free checks for Decision runtime batched Torch result transfer."""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime import qwen35_torch, vela_torch  # noqa: E402
from decision_runtime.qwen35_inputs import EncodedQwenRow  # noqa: E402
from decision_runtime.vela_inputs import EncodedVelaRow  # noqa: E402


class FakeTensor:
    def __init__(self, values, torch):
        self.values = list(values)
        self.torch = torch

    def __getitem__(self, index):
        return FakeTensor(self.values[index], self.torch)

    def __bool__(self):
        raise AssertionError("per-row tensor-to-bool synchronization")

    def float(self):
        return self

    def __truediv__(self, denominator):
        return FakeTensor((value / denominator for value in self.values), self.torch)

    def softmax(self, dimension):
        assert dimension == -1
        if not all(math.isfinite(value) for value in self.values):
            return FakeTensor([math.nan] * len(self.values), self.torch)
        maximum = max(self.values)
        weights = [math.exp(value - maximum) for value in self.values]
        total = sum(weights)
        return FakeTensor((weight / total for weight in weights), self.torch)

    def tolist(self):
        self.torch.host_transfers += 1
        return list(self.values)


class FakeTorch:
    bfloat16 = "bfloat16"

    def __init__(self):
        self.host_transfers = 0
        self.cat_calls = 0

    def inference_mode(self):
        return nullcontext()

    def autocast(self, **kwargs):
        assert kwargs == {"device_type": "cuda", "dtype": self.bfloat16}
        return nullcontext()

    def cat(self, tensors):
        self.cat_calls += 1
        return FakeTensor(
            (value for tensor in tensors for value in tensor.values), self
        )


def _runtime(family, monkeypatch, logits, *, temperature=1.0):
    torch = FakeTorch()

    class FakeModel:
        def __call__(self, *args, **kwargs):
            return tuple(FakeTensor(values, torch) for values in logits)

    if family == "vela":
        monkeypatch.setattr(vela_torch, "_collate", lambda *args, **kwargs: {})
        runtime = vela_torch.VelaTorchRuntime(
            model=FakeModel(),
            tokenizer=None,
            torch=torch,
            device=SimpleNamespace(type="cuda"),
            max_length=1024,
            backend="rocm",
        )
        rows = (
            EncodedVelaRow("first", "choice", (1, 2), (0, 1), ("a", "b"), 1),
            EncodedVelaRow(
                "second", "choice", (3, 4, 5), (0, 1, 2), ("c", "d", "e"), 2
            ),
        )
        error_type = vela_torch.VelaRuntimeError
    else:
        monkeypatch.setattr(qwen35_torch, "_collate", lambda *args, **kwargs: {})
        runtime = qwen35_torch.Qwen35TorchRuntime(
            model=FakeModel(),
            tokenizer=None,
            torch=torch,
            device=SimpleNamespace(type="cuda"),
            temperature=temperature,
            max_length=1024,
            backend="rocm",
            rocm_profile_binding=None,
        )
        rows = (
            EncodedQwenRow("first", "choice", (1, 2), (0, 1), 1, "0" * 64),
            EncodedQwenRow("second", "choice", (3, 4, 5), (0, 1, 2), 2, "1" * 64),
        )
        error_type = qwen35_torch.Qwen35RuntimeError
    return runtime, rows, torch, error_type


@pytest.mark.parametrize("family", ("vela", "qwen"))
def test_predict_encoded_transfers_once_and_preserves_row_order(
    family, monkeypatch: pytest.MonkeyPatch
) -> None:
    padding = math.nan if family == "vela" else -math.inf
    temperature = 2.0 if family == "qwen" else 1.0
    runtime, rows, torch, _ = _runtime(
        family,
        monkeypatch,
        ((0.0, 2.0, padding), (3.0, 1.0, 0.0)),
        temperature=temperature,
    )

    predictions = runtime.predict_encoded(rows)

    assert [item.question_id for item in predictions] == ["first", "second"]
    assert [item.type for item in predictions] == ["choice", "choice"]
    assert [item.input_tokens for item in predictions] == [2, 3]
    assert [len(item.probabilities) for item in predictions] == [2, 3]
    first_weights = [math.exp(value / temperature) for value in (0.0, 2.0)]
    second_weights = [math.exp(value / temperature) for value in (3.0, 1.0, 0.0)]
    assert predictions[0].probabilities == pytest.approx(
        [weight / sum(first_weights) for weight in first_weights]
    )
    assert predictions[1].probabilities == pytest.approx(
        [weight / sum(second_weights) for weight in second_weights]
    )
    assert torch.host_transfers == 1
    assert torch.cat_calls == 1


@pytest.mark.parametrize("family", ("vela", "qwen"))
@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_predict_encoded_rejects_nonfinite_valid_candidate_logits(
    family, invalid, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, rows, torch, error_type = _runtime(
        family, monkeypatch, ((0.0, 2.0, -math.inf), (3.0, invalid, 0.0))
    )

    with pytest.raises(error_type, match="non-finite .* candidate logits"):
        runtime.predict_encoded(rows)

    assert torch.host_transfers == 1
    assert torch.cat_calls == 1
