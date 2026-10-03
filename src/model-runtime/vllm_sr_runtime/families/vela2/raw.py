"""What a row's model outputs reduce to before calibration (shared by both members)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RawSpan:
    """Word x label logits of a span question over its whole target part.

    ``offsets`` are the words' code-point ranges; words no window covered
    carry the logit -30. ``alias`` maps a trained label back to the caller's.
    """

    labels: list[str]
    offsets: np.ndarray
    logits: np.ndarray
    head: str = "router"
    head_reason: str | None = None
    alias: dict[str, str] | None = None


@dataclass
class RawRow:
    """One row's outputs: option logits by question ID ([ABS] last when shown) and its span.

    ``tokens`` are the full token counts of the row's parts (before any cut),
    ``windows`` the number of windows a long part was read in (0 when none)
    and ``input_tokens`` every token the row's sequences fed the model.
    """

    logits: dict[str, np.ndarray] = field(default_factory=dict)
    span: RawSpan | None = None
    tokens: dict[str, int] = field(default_factory=dict)
    windows: int = 0
    input_tokens: int = 0
