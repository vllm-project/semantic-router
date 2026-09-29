"""F1 ``hs1_quote_check``: the contract between the family module and its evidence kinds.

A kind builder has the signature::

    def build(rng: random.Random, base: str, target: int | None, length: str) -> F1World

``base`` is the type of the kind's own question (choice / noul / score).
``target`` is the gold the world must have (0/1 for noul, a level for score,
None for choice). ``length`` is ``short`` or ``long`` (long worlds must supply
enough distractor records and filler to reach 3,000-7,500 characters once the
family module assembles them). A builder raises ``core.GenerationError`` when a
draw cannot meet its constraints; the family module then retries with a fresh
generator.

The family module (``f1_quote``) wraps a world into the right / wrong pair:
it renders the quote (speaker, channel, hedging, position), assembles the
state, writes the Noul-verify question, and builds the Items.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from v2.data.hs1.core import GenerationError, RecheckMismatch

MECHANISMS = (
    "overlooked_condition",
    "stale_value",
    "arithmetic_slip",
    "entity_swap",
    "short_chain",
    "boundary_misread",
    "scope_misapplied",
)


@dataclasses.dataclass(frozen=True)
class Claim:
    """A conclusion a person might state about the world's question.

    ``answer`` answers the base question in the world's encoding. The
    ``conclusion`` is one declarative sentence without quotation marks or a
    final full stop (e.g. "order 4471 arrived inside its promised window").
    The ``rationale`` is one clause citing one or two values (e.g. "it was
    scanned as delivered on 14 March, three business days after dispatch").
    Right and wrong claims must use the same templates so that the quote text
    alone cannot tell them apart.
    """

    answer: int
    conclusion: str
    rationale: str
    mechanism: str  # "correct" or one of MECHANISMS


@dataclasses.dataclass(frozen=True)
class F1World:
    kind: str
    base: str
    subject: str
    question: str
    choices: tuple[str, ...]
    gold: int
    recheck: int
    evidence: tuple[str, ...]
    distractors: tuple[str, ...]
    filler: tuple[str, ...]
    right: Claim
    wrong: Claim
    decisive: tuple[str, ...]
    facts: Mapping[str, Any]
    variant: str
    roles: tuple[str, ...]

    def check(self) -> None:
        if self.base not in ("choice", "noul", "score"):
            raise GenerationError(f"bad base {self.base}")
        if self.gold != self.recheck:
            raise RecheckMismatch(
                f"{self.kind}: oracle {self.gold} != recheck {self.recheck}"
            )
        if self.right.mechanism != "correct" or self.right.answer != self.gold:
            raise GenerationError(f"{self.kind}: right claim must equal gold")
        if self.wrong.mechanism not in MECHANISMS or self.wrong.answer == self.gold:
            raise GenerationError(f"{self.kind}: wrong claim must differ from gold")
        if self.base == "noul":
            if (
                self.choices
                or self.gold not in (0, 1)
                or self.wrong.answer != 1 - self.gold
            ):
                raise GenerationError(f"{self.kind}: bad noul world")
        elif self.base == "choice":
            if not 3 <= len(self.choices) <= 5 or len(set(self.choices)) != len(
                self.choices
            ):
                raise GenerationError(f"{self.kind}: choice needs 3-5 distinct options")
            if not 0 <= self.wrong.answer < len(self.choices):
                raise GenerationError(f"{self.kind}: wrong answer not offered")
        else:
            if not 3 <= len(self.choices) <= 5:
                raise GenerationError(f"{self.kind}: score needs 3-5 levels")
            if (
                not 0 <= self.wrong.answer < len(self.choices)
                or abs(self.wrong.answer - self.gold) > 2
            ):
                raise GenerationError(f"{self.kind}: wrong level must be 1-2 away")
        if not self.evidence or not self.roles:
            raise GenerationError(f"{self.kind}: evidence and roles required")
        text = "\n\n".join(self.evidence + self.distractors)
        for needle in self.decisive:
            if needle not in text:
                raise GenerationError(
                    f"{self.kind}: decisive fact not rendered: {needle!r}"
                )
        for claim in (self.right, self.wrong):
            if not claim.conclusion.strip() or not claim.rationale.strip():
                raise GenerationError(f"{self.kind}: empty claim text")
            if (
                claim.conclusion.rstrip().endswith((".", "!", "?"))
                or '"' in claim.conclusion
            ):
                raise GenerationError(
                    f"{self.kind}: conclusion must have no final stop or quotes"
                )
