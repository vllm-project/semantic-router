"""What every task head shares: the ModernBERT classifier, rows of one packed forward, the head contract.

A task model's heads read the hidden states of one shared forward. The
family deduplicates work items by token IDs, runs a single packed forward over
the distinct sequences with the union of the exits the heads need, and hands
each head the rows of its own items (``Rows``), so one pass serves every head
and every bundled task of the model.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
    "silu": F.silu,
}
HEAD_PREFIXES = {
    "head.dense.": "dense.",
    "head.norm.": "norm.",
    "classifier.": "classifier.",
}


class ClassifierHead(nn.Module):
    """``classifier(norm(act(dense(x))))``: ModernBERT's prediction head and classifier, in FP32."""

    def __init__(self, config: dict[str, Any], labels: int):
        super().__init__()
        hidden = config["hidden_size"]
        activation = config.get("classifier_activation", "gelu")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported classifier activation {activation!r}")
        self.act = ACTIVATIONS[activation]
        self.dense = nn.Linear(hidden, hidden, bias=bool(config.get("classifier_bias")))
        self.norm = nn.LayerNorm(
            hidden,
            eps=config.get("norm_eps", 1e-5),
            bias=bool(config.get("norm_bias", False)),
        )
        self.classifier = nn.Linear(hidden, labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(self.act(self.dense(hidden_states))))

    @classmethod
    def load(
        cls, files: Iterable[Path], config: dict[str, Any], labels: int
    ) -> ClassifierHead:
        """The head and classifier tensors of a task checkpoint (``head.*``, ``classifier.*``)."""
        from safetensors import safe_open

        head = cls(config, labels)
        expected = set(head.state_dict())
        state: dict[str, torch.Tensor] = {}
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
                    prefix = next(
                        (p for p in HEAD_PREFIXES if name.startswith(p)), None
                    )
                    if prefix is not None:
                        local = HEAD_PREFIXES[prefix] + name[len(prefix) :]
                        state[local] = handle.get_tensor(name).float()
        if set(state) != expected:
            missing, extra = sorted(expected - set(state)), sorted(
                set(state) - expected
            )
            raise ValueError(
                f"classifier tensors differ: missing {missing}, unexpected {extra}"
            )
        head.load_state_dict(state)
        return head.eval()


@dataclass
class Rows:
    """Hidden states of one packed forward by layer exit (``[tokens, hidden]``).

    Sequence ``i`` occupies rows ``starts[i] : starts[i] + lengths[i]``.
    ``outputs`` holds an engine's named graph outputs instead, one row per
    sequence, when it runs graphs with heads baked in.
    """

    hidden: dict[int, torch.Tensor]
    starts: list[int]
    lengths: list[int]
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def first(self, sequences: Sequence[int], layer: int) -> torch.Tensor:
        """The first token's row of each sequence (CLS pooling)."""
        hidden = self.hidden[layer]
        index = torch.tensor([self.starts[s] for s in sequences], device=hidden.device)
        return hidden.index_select(0, index)

    def mean(self, sequences: Sequence[int], layer: int) -> torch.Tensor:
        """The mean of every row of each sequence, special tokens included (mean pooling)."""
        hidden = self.hidden[layer]
        return torch.stack(
            [
                hidden[self.starts[s] : self.starts[s] + self.lengths[s]].sum(0)
                / self.lengths[s]
                for s in sequences
            ]
        )

    def last(self, sequences: Sequence[int], layer: int) -> torch.Tensor:
        """The last token's row of each sequence (last-token pooling)."""
        hidden = self.hidden[layer]
        index = torch.tensor(
            [self.starts[s] + self.lengths[s] - 1 for s in sequences],
            device=hidden.device,
        )
        return hidden.index_select(0, index)

    def tokens(self, sequences: Sequence[int], layer: int) -> torch.Tensor:
        """Every row of the sequences, concatenated in order."""
        hidden = self.hidden[layer]
        index = torch.cat(
            [
                torch.arange(self.starts[s], self.starts[s] + self.lengths[s])
                for s in sequences
            ]
        )
        return hidden.index_select(0, index.to(hidden.device))


@dataclass(frozen=True)
class Item:
    """One forward's worth of work for one head: framed token IDs and the exit it reads."""

    ids: tuple[int, ...]
    head: str
    layer: int
    cache_key: str | None = None


def cache_key(identity: str, head: str, layer: int, ids: Sequence[int]) -> str:
    """A content hash of everything an item's result depends on."""
    digest = hashlib.sha256(f"{identity}\0{head}\0{layer}\0".encode())
    digest.update(",".join(map(str, ids)).encode())
    return digest.hexdigest()


@dataclass
class Prepared:
    """One input made ready for a head: its items, usage and what ``result`` needs."""

    items: list[Item]
    usage: dict[str, Any]
    state: Any = None


@dataclass
class HeadOptions:
    """The request options a head honours, already validated against its limits."""

    overflow: str
    max_tokens: int
    window: tuple[int, int] | None = None
    threshold: float | None = None
    return_tokens: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Head(ABC):
    """One readout of a model's shared forward: the exit it reads and a batched ``readout``.

    The family runs one forward for the items of every head and hands each
    head the rows of its own items (on the model's worker, one batched call
    per forward).
    """

    kind: ClassVar[str]
    surface: ClassVar[str]

    def __init__(self, name: str, layer: int):
        self.name = name
        self.layer = layer

    @abstractmethod
    def readout(self, rows: Rows, sequences: Sequence[int]) -> list[Any]:
        """Per sequence, this head's result; must not keep references to ``rows``."""

    def to(self, device: torch.device) -> Head:
        """Move the head's tensors to the backbone's device."""
        return self


class TaskHead(Head):
    """A classify head: ``prepare`` turns one input into items (on the request
    thread) and ``result`` assembles one input's API result from their readouts.
    """

    surface: ClassVar[str] = "classify"

    def __init__(self, name: str, labels: Sequence[str], layer: int):
        super().__init__(name, layer)
        self.labels = tuple(labels)

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """The head's card fields (``HeadInfo`` keyword arguments)."""

    @abstractmethod
    def prepare(self, value: Any, options: HeadOptions, identity: str) -> Prepared:
        """Validate and tokenize one input; raises ``ValueError`` for a bad input."""

    @abstractmethod
    def result(self, prepared: Prepared, values: Sequence[Any]) -> dict[str, Any]:
        """One input's API result from its items' readouts, in item order."""
