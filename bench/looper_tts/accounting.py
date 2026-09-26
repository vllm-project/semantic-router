"""Fixed-budget admission and per-call accounting for Looper experiments.

The benchmark contract deliberately keeps runtime accounting out of the
manifest.  This module is the small, deterministic runtime layer shared by the
live executor and its fake-provider tests:

* every attempted call reserves its prompt estimate plus output ceiling before
  dispatch;
* provider usage replaces that reservation when it contains a usable total (an
  explicit total or a prompt/completion pair);
* missing usage or usage without a billable prompt/completion pair is charged
  at the reservation and remains visible as unknown evidence; and
* all counters are protected so ReMoM/Fusion parallel calls cannot oversubscribe
  a cell's envelope.

No provider-specific response is imported here.  Adapters only need to return
the three OpenAI-compatible usage fields, preserving ``None`` for fields that
were not present on the wire.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Usage:
    """Token usage with explicit field presence.

    ``None`` means the provider omitted the field.  A value of zero is valid
    evidence and is therefore retained as a value rather than converted to
    ``None``.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> Usage:
        if not isinstance(value, Mapping):
            return cls()

        def integer(name: str) -> int | None:
            raw = value.get(name)
            if raw is None or isinstance(raw, bool):
                return None
            if isinstance(raw, int):
                return raw if raw >= 0 else None
            # A few compatible clients decode JSON numbers as floats. Accept an
            # integral finite float while rejecting lossy/non-finite values.
            if (
                isinstance(raw, float)
                and math.isfinite(raw)
                and raw.is_integer()
                and raw >= 0
            ):
                return int(raw)
            return None

        return cls(
            prompt_tokens=integer("prompt_tokens"),
            completion_tokens=integer("completion_tokens"),
            total_tokens=integer("total_tokens"),
        )

    def as_record(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @property
    def complete(self) -> bool:
        return (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
        )

    @property
    def budget_total(self) -> int | None:
        """Return the best known total for budget settlement."""
        if self.total_tokens is not None:
            return self.total_tokens
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None


@dataclass(frozen=True)
class Pricing:
    """USD rates from one manifest model entry."""

    input_per_million: float | None = None
    output_per_million: float | None = None

    @classmethod
    def from_model(cls, model: Mapping[str, Any]) -> Pricing:
        raw = model.get("pricing", {})
        if not isinstance(raw, Mapping):
            return cls()
        return cls(
            input_per_million=_finite_rate(raw.get("input_per_million")),
            output_per_million=_finite_rate(raw.get("output_per_million")),
        )


def _finite_rate(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0:
        return float(value)
    return None


def cost_for_usage(usage: Usage, pricing: Pricing) -> float | None:
    """Calculate USD cost only when both billable token fields are known."""
    if (
        usage.prompt_tokens is None
        or usage.completion_tokens is None
        or pricing.input_per_million is None
        or pricing.output_per_million is None
    ):
        return None
    return (
        usage.prompt_tokens * pricing.input_per_million
        + usage.completion_tokens * pricing.output_per_million
    ) / 1_000_000.0


def estimate_prompt_tokens(messages: Any) -> int:
    """Make a stable, tokenizer-free prompt estimate for admission.

    The provider's usage remains authoritative.  Compact JSON plus a four-byte
    byte-pair approximation is intentionally conservative enough for admission
    and works for arbitrary OpenAI-compatible message content.
    """
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return max(1, math.ceil(len(encoded) / 4.0))


def estimate_total_tokens(messages: Any, max_output_tokens: int) -> int:
    if type(max_output_tokens) is not int or max_output_tokens < 0:
        raise ValueError("max_output_tokens must be non-negative")
    return estimate_prompt_tokens(messages) + max_output_tokens


@dataclass(frozen=True)
class BudgetSnapshot:
    calls: int
    active_calls: int
    tokens: int
    reserved_tokens: int
    unknown_usage_calls: int
    exhausted: bool


class BudgetExhaustedError(RuntimeError):
    """Raised when a call cannot be admitted under the cell's envelope."""

    def __init__(self, reason: str, snapshot: BudgetSnapshot):
        self.reason = reason
        self.snapshot = snapshot
        super().__init__("budget exhausted: " + reason)


@dataclass(frozen=True)
class Reservation:
    call_id: str
    estimated_tokens: int


@dataclass(frozen=True)
class Settlement:
    call_id: str
    estimated_tokens: int
    charged_tokens: int
    usage: Usage
    usage_known: bool
    usage_source: str
    exhausted: bool


class BudgetLedger:
    """Thread-safe per-cell budget ledger.

    The call ceiling counts admitted calls, including failed and unusable
    responses.  Unknown usage is charged at the original reservation so an
    endpoint that omits usage cannot bypass the token ceiling.
    """

    def __init__(self, max_calls: int, max_total_tokens: int):
        if type(max_calls) is not int or max_calls <= 0:
            raise ValueError("max_calls must be a positive integer")
        if type(max_total_tokens) is not int or max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be a positive integer")
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self._lock = threading.RLock()
        self._calls = 0
        self._active_calls = 0
        self._tokens = 0
        self._reserved_tokens = 0
        self._unknown_usage_calls = 0
        self._exhausted = False
        self._reservations: dict[str, Reservation] = {}

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                calls=self._calls,
                active_calls=self._active_calls,
                tokens=self._tokens,
                reserved_tokens=self._reserved_tokens,
                unknown_usage_calls=self._unknown_usage_calls,
                exhausted=self._exhausted,
            )

    def reserve(self, call_id: str, estimated_tokens: int) -> Reservation:
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("call_id must be a non-empty string")
        if type(estimated_tokens) is not int or estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be a positive integer")
        with self._lock:
            if call_id in self._reservations:
                raise ValueError("duplicate active call_id: " + call_id)
            if self._calls >= self.max_calls:
                self._exhausted = True
                raise BudgetExhaustedError("maximum calls reached", self.snapshot())
            if (
                self._tokens + self._reserved_tokens + estimated_tokens
                > self.max_total_tokens
            ):
                self._exhausted = True
                raise BudgetExhaustedError(
                    "estimated token reservation exceeds limit", self.snapshot()
                )
            reservation = Reservation(call_id, estimated_tokens)
            self._reservations[call_id] = reservation
            self._calls += 1
            self._active_calls += 1
            self._reserved_tokens += estimated_tokens
            return reservation

    def settle(self, reservation: Reservation, usage: Any = None) -> Settlement:
        if not isinstance(reservation, Reservation):
            raise TypeError("settle expects a Reservation")
        parsed = usage if isinstance(usage, Usage) else Usage.from_mapping(usage)
        with self._lock:
            active = self._reservations.pop(reservation.call_id, None)
            if active is None:
                raise ValueError(
                    "unknown or already settled call_id: " + reservation.call_id
                )
            self._active_calls -= 1
            self._reserved_tokens -= active.estimated_tokens
            self._reserved_tokens = max(0, self._reserved_tokens)
            charged = parsed.budget_total
            known = charged is not None
            if not known:
                charged = active.estimated_tokens
                self._unknown_usage_calls += 1
            charged = max(0, int(charged))
            self._tokens += charged
            if self._tokens > self.max_total_tokens:
                self._exhausted = True
            return Settlement(
                call_id=active.call_id,
                estimated_tokens=active.estimated_tokens,
                charged_tokens=charged,
                usage=parsed,
                usage_known=known,
                usage_source="provider" if known else "reservation",
                exhausted=self._exhausted,
            )

    def close(self) -> BudgetSnapshot:
        """Mark an unfinished execution exhausted and return its snapshot."""
        with self._lock:
            if self._reservations:
                self._exhausted = True
            return self.snapshot()
