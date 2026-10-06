"""The Vela 2.0 heads against their formulas: the marker head (0.3B) and the span head (decoders)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from vllm_srun.heads.marker import MarkerHead  # noqa: E402
from vllm_srun.heads.span import SpanHead  # noqa: E402

HIDDEN = torch.Generator().manual_seed(1)
Q_INDEX = torch.tensor([[0, 1, 2, 6], [1, 0, 3, 9]])
OPT_INDEX = torch.tensor([[0, 2, 0], [0, 7, 0], [1, 4, 1]])
UNIT_INDEX = torch.tensor([[0, 5], [1, 7]])
ENT_INDEX = torch.tensor([[0, 8], [1, 9], [1, 2]])


def randomized(module, seed: int):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
    return module.eval()


def marker_inputs():
    hidden = torch.randn(2, 10, 8, generator=torch.Generator().manual_seed(1))
    return hidden, Q_INDEX, OPT_INDEX, UNIT_INDEX, ENT_INDEX


@pytest.mark.parametrize("readout", ["cosine", "mlp", "both"])
def test_marker_head_scores_options_and_words(readout: str) -> None:
    head = randomized(MarkerHead(8, 4, readout), 0)
    hidden, *indices = marker_inputs()
    with torch.no_grad():
        options, words = head(hidden, *indices)
        expected = []
        for row, marker, owner in OPT_INDEX.tolist():
            source, query, start, end = Q_INDEX[owner].tolist()
            value = torch.zeros(())
            if readout != "mlp":
                option = head.w_o(head.norm(hidden[row, marker]))
                option = F.normalize(
                    option + head.w_q(head.norm(hidden[source, query])), dim=-1
                )
                pooled = hidden[source, start:end].mean(0)
                part = F.normalize(head.w_p(head.norm(pooled)), dim=-1)
                value = value + (option * part).sum() / head.log_tau.exp()
            if readout != "cosine":
                value = value + head.cls_mlp(hidden[row, marker]).squeeze()
            expected.append(value)
        word = head.w_t(head.norm(hidden[UNIT_INDEX[:, 0], UNIT_INDEX[:, 1]]))
        label = head.w_e(head.norm(hidden[ENT_INDEX[:, 0], ENT_INDEX[:, 1]]))
        pairs = F.normalize(word, dim=-1) @ F.normalize(label, dim=-1).T
    torch.testing.assert_close(options, torch.stack(expected))
    torch.testing.assert_close(words, pairs / head.log_tau_span.exp())


def test_marker_head_reads_in_fp32_under_autocast() -> None:
    head = randomized(MarkerHead(8, 4, "both"), 0)
    hidden, *indices = marker_inputs()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        options, words = head(hidden.bfloat16(), *indices)
    assert options.dtype == words.dtype == torch.float32
    with pytest.raises(ValueError):
        MarkerHead(8, 4, "attention")


@pytest.mark.parametrize("labels", [1, 3, 5])
def test_span_head_scores_words_against_labels(labels: int) -> None:
    head = randomized(SpanHead(8, projection=4, slots=3), 2)
    generator = torch.Generator().manual_seed(3)
    words = torch.randn(6, 8, generator=generator)
    names = torch.randn(labels, 8, generator=generator)
    with torch.no_grad():
        scores = head(words, names)
        word = head.wn(words)
        word = word - word.mean(0, keepdim=True)
        label = head.ln(names) + head.slot(torch.arange(labels).clamp(max=2))
        if labels > 1:
            label = label - label.mean(0, keepdim=True)
        bilinear = head.K(word) @ head.Q(label).T / math.sqrt(4)
        mlp = head.v(F.gelu(head.M(word)[:, None] + head.N(label)[None])).squeeze(-1)
    assert scores.shape == (6, labels)
    torch.testing.assert_close(scores, bilinear + mlp)
