"""System One requests over any engine that returns candidate logits.

Mirrors ``Decision2.system_one`` and ``QwenDecision.system_one`` of the package
runtime: the same validation, the same per-question errors, over-budget input
is never truncated, and answers keep the caller's question order. The only
difference is execution: each valid question is one engine request carrying
its token IDs and positions, and a request's questions run concurrently, so
the engine batches them with other traffic instead of padding them together.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .gather import POSITIONS_KEY
from .package import Decision2Package

EncodeFn = Callable[[list[int], dict[str, Any], str], Awaitable[list[float]]]


class SystemOneService:
    def __init__(self, package: Decision2Package, encode: EncodeFn):
        self.package = package
        self._encode = encode

    def prepare(
        self, state: Any, questions: Any
    ) -> tuple[
        dict[str, dict[str, Any]], list[tuple[str, dict[str, Any], dict[str, Any]]], int
    ]:
        """Validate and tokenize; returns (error answers, jobs, input tokens). CPU only."""
        runtime = self.package.runtime
        if (
            not isinstance(questions, dict)
            or not questions
            or any(not isinstance(key, str) or not key for key in questions)
        ):
            raise ValueError("questions must be a nonempty mapping of question IDs")
        if not runtime.api_json_payload(state):
            raise ValueError("state must be text, an object, or an array")
        try:
            runtime.canonical(state)
        except (TypeError, ValueError) as exc:
            raise ValueError("state must contain JSON data") from exc
        item = {"id": "request", "state": state}
        answers: dict[str, dict[str, Any]] = {}
        jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        tokens = 0
        for qid, question in questions.items():
            try:
                row = runtime.question_to_row(item, qid, question)
                encoded = runtime.encode(
                    row, self.package.tokenizer, self.package.max_input_tokens
                )
            except ValueError as exc:
                reason = (
                    "max_length_exceeded"
                    if "exceeds max_length" in str(exc)
                    else "invalid_question"
                )
                answers[qid] = {
                    "type": (
                        question.get("type") if isinstance(question, dict) else None
                    ),
                    "error": reason,
                }
                continue
            tokens += len(encoded["ids"])
            jobs.append((qid, row, encoded))
        return answers, jobs, tokens

    def answer(
        self, row: dict[str, Any], encoded: dict[str, Any], logits: list[float]
    ) -> dict[str, Any]:
        runtime = self.package.runtime
        try:
            if len(logits) != len(encoded["keys"]):
                raise ValueError("wrong number of candidate logits")
            values = [float(value) for value in logits]
            if self.package.score_bias is not None and row["task_type"] == "score":
                values = runtime.apply_score_bias(
                    self.package.score_bias, values, len(row["options"])
                )
            return runtime.product_answer(
                row["task_type"],
                encoded["keys"],
                values,
                self.package.temperatures[row["task_type"]],
                [option["description"] for option in row["options"]],
            )
        except ValueError:
            return {"type": row["task_type"], "error": "invalid_model_output"}

    async def system_one(
        self, *, state: Any, questions: Any, request_id: str
    ) -> dict[str, Any]:
        answers, jobs, tokens = await asyncio.to_thread(self.prepare, state, questions)
        results = await asyncio.gather(
            *(
                self._encode(
                    encoded["ids"],
                    {
                        POSITIONS_KEY: {
                            "candidate_positions": encoded["candidate_positions"],
                            "query_position": encoded["query_position"],
                        }
                    },
                    f"{request_id}-{index}",
                )
                for index, (_, _, encoded) in enumerate(jobs)
            )
        )
        for (qid, row, encoded), logits in zip(jobs, results):
            answers[qid] = self.answer(row, encoded, logits)
        return {
            "model": self.package.model_name,
            "answers": {qid: answers[qid] for qid in questions},
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }
