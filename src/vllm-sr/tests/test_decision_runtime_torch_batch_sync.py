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


class FakeMask:
    def __init__(self, values):
        self.values = [list(row) for row in values]

    def __invert__(self):
        return FakeMask(([not value for value in row] for row in self.values))


class FakeMatrix:
    def __init__(self, values, torch):
        self.values = [list(row) for row in values]
        self.torch = torch

    def float(self):
        return self

    def masked_fill(self, mask, value):
        self.torch.masked_fill_calls += 1
        return FakeMatrix(
            (
                [
                    value if hidden else item
                    for item, hidden in zip(row, flags, strict=True)
                ]
                for row, flags in zip(self.values, mask.values, strict=True)
            ),
            self.torch,
        )

    def __truediv__(self, denominator):
        return FakeMatrix(
            ([value / denominator for value in row] for row in self.values),
            self.torch,
        )

    def softmax(self, dimension):
        assert dimension == -1
        self.torch.softmax_calls += 1
        output = []
        for row in self.values:
            if any(not math.isfinite(value) and value != -math.inf for value in row):
                output.append([math.nan] * len(row))
                continue
            finite = [value for value in row if math.isfinite(value)]
            if not finite:
                output.append([math.nan] * len(row))
                continue
            maximum = max(finite)
            weights = [
                math.exp(value - maximum) if value != -math.inf else 0.0
                for value in row
            ]
            total = sum(weights)
            output.append([weight / total for weight in weights])
        return FakeMatrix(output, self.torch)

    def tolist(self):
        self.torch.host_transfers += 1
        return [list(row) for row in self.values]


class FakeTorch:
    bfloat16 = "bfloat16"

    def __init__(self):
        self.host_transfers = 0
        self.cat_calls = 0
        self.masked_fill_calls = 0
        self.softmax_calls = 0

    def inference_mode(self):
        return nullcontext()

    def autocast(self, **kwargs):
        assert kwargs == {"device_type": "cuda", "dtype": self.bfloat16}
        return nullcontext()

    def cat(self, tensors):
        self.cat_calls += 1
        tensors = tuple(tensors)
        if isinstance(tensors[0], FakeMatrix):
            return FakeMatrix(
                (row for tensor in tensors for row in tensor.values), self
            )
        return FakeTensor(
            (value for tensor in tensors for value in tensor.values), self
        )


def _runtime(family, monkeypatch, logits, *, temperature=1.0, qwen_rows=None):
    torch = FakeTorch()

    class FakeModel:
        def __call__(self, *args, **kwargs):
            if family == "vela":
                return tuple(FakeTensor(values, torch) for values in logits)
            return FakeMatrix(logits, torch)

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
        monkeypatch.setattr(
            qwen35_torch,
            "_collate",
            lambda _, items, *args, **kwargs: {
                "candidate_mask": FakeMask(
                    (
                        [
                            index < len(row.candidate_positions)
                            for index in range(len(logits[0]))
                        ]
                        for row in items
                    )
                )
            },
        )
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
        rows = qwen_rows or (
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
    if family == "qwen":
        assert torch.masked_fill_calls == 1
        assert torch.softmax_calls == 1


@pytest.mark.parametrize("temperature", (0.7, 1.3))
def test_qwen_batched_softmax_matches_rowwise_reference_with_mixed_candidates(
    temperature: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = (
        EncodedQwenRow("first", "choice", (1, 2, 3), (0, 1), 2, "0" * 64),
        EncodedQwenRow("second", "choice", (4, 5, 6, 7), (0, 1, 2), 3, "1" * 64),
        EncodedQwenRow(
            "third", "choice", (8, 9, 10, 11, 12), (0, 1, 2, 3), 4, "2" * 64
        ),
    )
    logits = (
        (1.0, 1.000001, math.nan, 100.0),
        (3.0, 2.999999, 3.000002, 100.0),
        (0.4, 0.400002, 0.399998, 0.400001),
    )
    runtime, _, torch, _ = _runtime(
        "qwen", monkeypatch, logits, temperature=temperature, qwen_rows=rows
    )

    predictions = runtime.predict_encoded(rows)

    for row, raw, prediction in zip(rows, logits, predictions, strict=True):
        # This is the old rowwise computation over real candidates only.
        selected = raw[: len(row.candidate_positions)]
        scaled = [value / temperature for value in selected]
        peak = max(scaled)
        weights = [math.exp(value - peak) for value in scaled]
        reference = [weight / sum(weights) for weight in weights]
        assert prediction.probabilities == pytest.approx(reference, abs=1e-12)
        assert max(
            range(len(prediction.probabilities)),
            key=prediction.probabilities.__getitem__,
        ) == max(range(len(reference)), key=reference.__getitem__)
    assert [prediction.question_id for prediction in predictions] == [
        "first",
        "second",
        "third",
    ]
    assert torch.masked_fill_calls == 1
    assert torch.softmax_calls == 1
    assert torch.host_transfers == 1
    assert torch.cat_calls == 1


def test_qwen_cpu_torch_matches_rowwise_softmax_with_padded_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    rows = (
        EncodedQwenRow("first", "choice", (1, 2, 3), (0, 1), 2, "0" * 64),
        EncodedQwenRow("second", "choice", (4, 5, 6, 7), (0, 1, 2), 3, "1" * 64),
        EncodedQwenRow(
            "third", "choice", (8, 9, 10, 11, 12), (0, 1, 2, 3), 4, "2" * 64
        ),
    )
    logits = torch.tensor(
        [
            [1.0, 1.000001, math.nan, 100.0],
            [3.0, 2.999999, 3.000002, 100.0],
            [0.4, 0.400002, 0.399998, 0.400001],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False], [True] * 4]
    )
    monkeypatch.setattr(
        qwen35_torch,
        "_collate",
        lambda *args, **kwargs: {"candidate_mask": mask},
    )
    temperature = 1.3
    runtime = qwen35_torch.Qwen35TorchRuntime(
        model=lambda **kwargs: logits,
        tokenizer=None,
        torch=torch,
        device=torch.device("cpu"),
        temperature=temperature,
        max_length=1024,
        backend="cpu",
        rocm_profile_binding=None,
    )

    predictions = runtime.predict_encoded(rows)

    for index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        count = len(row.candidate_positions)
        reference = (logits[index, :count].float() / temperature).softmax(-1).tolist()
        assert prediction.probabilities == pytest.approx(reference, abs=1e-7)
        assert max(range(count), key=prediction.probabilities.__getitem__) == max(
            range(count), key=reference.__getitem__
        )


@pytest.mark.parametrize("family", ("vela", "qwen"))
@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_predict_encoded_rejects_nonfinite_valid_candidate_logits(
    family, invalid, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, rows, torch, error_type = _runtime(
        family, monkeypatch, ((0.0, 2.0, -math.inf), (3.0, invalid, 0.0))
    )

    with pytest.raises(error_type, match=r"non-finite .* candidate logits"):
        runtime.predict_encoded(rows)

    assert torch.host_transfers == 1
    assert torch.cat_calls == 1
