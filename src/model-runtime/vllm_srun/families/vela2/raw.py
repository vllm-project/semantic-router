"""What a row's model outputs reduce to before calibration (shared by both members)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class RawSpan:
    """Word x label logits of a span question over its whole target part.

    ``offsets`` are the words' code-point ranges; words no window covered
    carry the logit -30. ``alias`` maps a trained label back to the caller's.
    """

    labels: list[str]
    offsets: NDArray[np.int32]
    logits: NDArray[np.float64]
    head: str = "router"
    alias: dict[str, str] | None = None


@dataclass
class RawRow:
    """One row's outputs: option logits by question ID ([ABS] last when shown) and its span.

    ``tokens`` are the full token counts of the row's parts (before any cut),
    ``windows`` the number of windows a long part was read in (0 when none)
    and ``input_tokens`` every token the row's sequences fed the model.
    """

    logits: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    span: RawSpan | None = None
    tokens: dict[str, int] = field(default_factory=dict)
    windows: int = 0
    input_tokens: int = 0
