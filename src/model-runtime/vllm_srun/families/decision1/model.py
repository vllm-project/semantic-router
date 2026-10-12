"""Loaded Decision 1.0 models: one class per runtime over shared System One handling.

``Decision1Model`` owns what both runtimes share: the request rules and
presets, the bundled runtime's admission rule (one over-long question fails
every question of the request, nothing is truncated), the physical batches of
the ``exact`` profile and the answers. A subclass renders a question and runs
a batch on its engine model.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import torch

from ...errors import (
    INVALID_QUESTION,
    MAX_LENGTH_EXCEEDED,
    QuestionError,
    question_error,
)
from ...heads.candidate import CandidateHead, forward_logits
from ...heads.typed import TypeReadout
from ...plugins.base import EngineModel, ModelInfo
from ...plugins.decisions import DecisionModel, RenderedItem, RequestPlan
from ...systemone import canonical
from ...text import segments
from ...text.tokenizer import Tokenizer
from . import qwen, vela
from .answers import answer
from .questions import KINDS, NoulDefaults, Row, check_request, parse


class Decision1Model(DecisionModel[RenderedItem, list[float] | None]):
    """A loaded Decision 1.0 model; ``render``, ``physical_batches`` and ``run`` are per runtime."""

    noul_defaults: ClassVar[NoulDefaults]
    fuse_bundled_jobs: ClassVar[bool] = False

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        tokenizer: Tokenizer,
        presets: dict[str, dict[str, Any]],
    ):
        self.info = info
        self.engine_model = engine_model
        self.tokenizer = tokenizer
        self.presets = presets

    @abstractmethod
    def render(
        self, row: Row, state: str, tokens: Callable[[str], list[int]]
    ) -> RenderedItem:
        """One question's model input; raises ``QuestionError``."""

    @abstractmethod
    def physical_batches(self, items: list[RenderedItem]) -> list[list[int]]:
        """How the released runtime splits a request's items into forwards."""

    def tokens(self) -> Callable[[str], list[int]]:
        """The tokenizer for one request's rendering."""
        return self.tokenizer.encode

    def exact_batches(self, items: list[RenderedItem]) -> list[list[int]]:
        """The released physical batches; each stays within the forward token budget."""
        return self.physical_batches(items)

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[RenderedItem]:
        check_request(state, questions)
        text = state if isinstance(state, str) else canonical(state)
        rows: list[Row] = []
        errors: dict[str, dict[str, Any]] = {}
        for question_id, question in questions.items():
            try:
                rows.append(
                    parse(question_id, question, self.noul_defaults, self.presets)
                )
            except QuestionError as exc:
                kind = question.get("type") if isinstance(question, dict) else None
                errors[question_id] = question_error(
                    kind if kind in KINDS else None, exc, INVALID_QUESTION
                )
        tokens = self.tokens()
        items: list[RenderedItem] = []
        for row in rows:
            try:
                items.append(self.render(row, text, tokens))
            except QuestionError as exc:
                if exc.code != MAX_LENGTH_EXCEEDED:
                    errors[row.question_id] = question_error(row.kind, exc)
                    continue
                for failed in rows:
                    errors[failed.question_id] = {
                        "type": failed.kind,
                        "error": MAX_LENGTH_EXCEEDED,
                    }
                items = []
                break
        return RequestPlan(
            complete_inputs=frozenset(
                key
                for key, question in questions.items()
                if isinstance(question, dict)
                and question.get("require_full_input") is True
                and key not in errors
            ),
            question_ids=list(questions),
            items=items,
            errors=errors,
            input_tokens=sum(len(item.ids) for item in items),
        )

    def answer(self, item: RenderedItem, logits: list[float] | None) -> dict[str, Any]:
        """``logits`` holds the item's candidate probabilities (the runtimes normalize on the device)."""
        return answer(item.task_type, item.keys, item.descriptions, logits)


class VelaDecisionModel(Decision1Model):
    """Kai, Lex and Route: three ModernBERT layer stacks over one embedding, typed heads (``vela.py``)."""

    noul_defaults = vela.NOUL_DEFAULTS

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        tokenizer: Tokenizer,
        presets: dict[str, dict[str, Any]],
        *,
        readout: TypeReadout,
        special: dict[str, int],
        exit_layer: int,
    ):
        super().__init__(info, engine_model, tokenizer, presets)
        self.readout = readout.to(engine_model.device)
        self.special = special
        self.exit_layer = exit_layer

    def tokens(self) -> Callable[[str], list[int]]:
        return vela.token_cache(self.tokenizer.encode)

    def render(
        self, row: Row, state: str, tokens: Callable[[str], list[int]]
    ) -> RenderedItem:
        return vela.render(
            row, state, tokens, self.special, self.info.limits["max_input_tokens"]
        )

    def physical_batches(self, items: list[RenderedItem]) -> list[list[int]]:
        return vela.physical_batches(items)

    def forward_token_budget(self) -> int | None:
        """Coalesced batches on CPUs stay within ``vela.CPU_BATCH_TOKENS``; exact keeps the released batches."""
        return vela.CPU_BATCH_TOKENS if self.engine_model.device.type == "cpu" else None

    def run(self, items: list[RenderedItem]) -> list[list[float] | None]:
        with torch.inference_mode():
            scores = vela.marker_logits(
                self.engine_model.encode,
                self.readout,
                items,
                self.special["pad"],
                self.engine_model.device,
                self.exit_layer,
            )
            if not torch.isfinite(scores).all():
                return [None] * len(items)
            return [
                scores[slot, : len(item.keys)].softmax(-1).cpu().tolist()
                for slot, item in enumerate(items)
            ]

    def run_approximate(self, items: list[RenderedItem]) -> list[list[float] | None]:
        """Each question type's layer stack over that type's rows only, packed without padding.

        The released runtime runs every type's stack over the whole padded
        batch; rows of other types and padding are wasted work, but dropping
        them changes GEMM and attention shapes, so this is for approximate
        profiles: up to three times fewer row passes on a mixed request and no
        padding across coalesced requests.
        """
        results: list[list[float] | None] = [None] * len(items)
        with torch.inference_mode():
            for kind in KINDS:
                rows = [
                    slot for slot, item in enumerate(items) if item.task_type == kind
                ]
                if not rows:
                    continue
                group = [items[slot] for slot in rows]
                scores = vela.packed_marker_logits(
                    self.engine_model.encode,
                    self.readout,
                    kind,
                    group,
                    self.engine_model.device,
                    self.exit_layer,
                )
                finite = bool(torch.isfinite(scores).all())
                for position, (slot, item) in enumerate(zip(rows, group, strict=True)):
                    results[slot] = (
                        scores[position, : len(item.keys)].softmax(-1).cpu().tolist()
                        if finite
                        else None
                    )
        return results


class QwenDecisionModel(Decision1Model):
    """Eos, Sol, Nox and Lux: a Qwen3.5 backbone and the shared candidate head (``qwen.py``)."""

    noul_defaults = qwen.NOUL_DEFAULTS

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        tokenizer: Tokenizer,
        presets: dict[str, dict[str, Any]],
        *,
        head: CandidateHead,
        temperatures: dict[str, float],
        null_choice_as_key: bool,
    ):
        super().__init__(info, engine_model, tokenizer, presets)
        self.head = head.to(engine_model.device)
        self.temperatures = temperatures
        self.null_choice_as_key = null_choice_as_key

    def render(
        self, row: Row, state: str, tokens: Callable[[str], list[int]]
    ) -> RenderedItem:
        return qwen.render(
            row,
            state,
            tokens,
            self.info.limits["max_input_tokens"],
            self.null_choice_as_key,
        )

    def physical_batches(self, items: list[RenderedItem]) -> list[list[int]]:
        return qwen.physical_batches(items, self.forward_token_budget())

    def forward_token_budget(self) -> int | None:
        return self.engine_model.max_forward_tokens()

    def run(self, items: list[RenderedItem]) -> list[list[float] | None]:
        return self.run_shared(items, 0)

    def run_shared(
        self, items: list[RenderedItem], shared_prefix: int
    ) -> list[list[float] | None]:
        batch = segments.collate(items, self.tokenizer.pad_id, qwen.PAD_MULTIPLE)
        scores = forward_logits(
            self.engine_model,
            self.head,
            batch,
            [len(item.ids) for item in items],
            shared_prefix,
        )
        with torch.inference_mode():
            return qwen.probabilities(scores, items, self.temperatures)
