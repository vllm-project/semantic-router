"""Fixed tokenizer pair framing, shared by sequence and grounded heads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

PAIR_PROBES = (("", "x"), ("Hello, world!", "中文 🚀"), ("<eos> a", "b <bos>"))


@dataclass(frozen=True)
class PairEnvelope:
    """The special tokens a tokenizer puts around and between the two sequences of a pair."""

    prefix: tuple[int, ...]
    middle: tuple[int, ...]
    suffix: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.prefix) + len(self.middle) + len(self.suffix)

    def frame(self, first: Sequence[int], second: Sequence[int]) -> list[int]:
        return [*self.prefix, *first, *self.middle, *second, *self.suffix]

    @classmethod
    def of(cls, tokenizer: Any) -> PairEnvelope:
        """The tokenizer's pair framing; refused when it depends on the texts."""
        encoding = tokenizer.encode("a", "b", add_special_tokens=True)
        ids, sequence = list(encoding.ids), list(encoding.sequence_ids)
        first = [index for index, s in enumerate(sequence) if s == 0]
        second = [index for index, s in enumerate(sequence) if s == 1]
        if not first or not second:
            raise ValueError("the tokenizer does not encode pairs")
        envelope = cls(
            tuple(ids[: first[0]]),
            tuple(ids[first[-1] + 1 : second[0]]),
            tuple(ids[second[-1] + 1 :]),
        )
        for a, b in PAIR_PROBES:
            framed = envelope.frame(
                tokenizer.encode(a, add_special_tokens=False).ids,
                tokenizer.encode(b, add_special_tokens=False).ids,
            )
            if framed != tokenizer.encode(a, b, add_special_tokens=True).ids:
                raise ValueError("the tokenizer's pair framing is not a fixed envelope")
        return envelope
