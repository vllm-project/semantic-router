"""A Decision Index kit engine that answers through the runtime, for Index records of a served model.

The kit's runner loads it with this directory on ``PYTHONPATH``:

    python3 -m decision_index run --engine index_engine:RuntimeIndexEngine \\
        --option model=vllm-sr/Decision-2.0-Eos-0.8B --option revision=REV --option device=rocm:0 \\
        --rows ROWS.jsonl.gz --out RUN_DIR --compact

The model loads as ``vllm-sr-runtime serve`` loads it (resolution, verification, placement, pinned
kernel choices, the golden check that gates readiness), offline from the Hugging Face cache, and
answers each row as one request on the ``exact`` profile through the scheduler, as
``cross_process.py`` does. A question the model refuses (``max_length_exceeded``,
``invalid_question``) makes the row unsupported. Any other defect is an error: a missing or extra
answer, another model identity, a probability outside [0, 1], options that do not sum to 1 within
1e-5, or a choice other than the first most probable option (ties within 1e-8, in the caller's
order).

Paths stay out of the options the kit records: ``INDEX_ENGINE_CACHE_DIR`` (the Hugging Face
cache), ``INDEX_ENGINE_AUTOTUNE_CACHE`` and ``INDEX_ENGINE_RECEIPT`` (where ``close`` writes what
ran: versions, the golden check, the kernel choices and the engine's receipt) come from the
environment.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cross_process import autotune_entries, kernel_choices, versions
from vllm_sr_runtime.config import ModelConfig, ServeConfig
from vllm_sr_runtime.errors import INVALID_QUESTION, MAX_LENGTH_EXCEEDED
from vllm_sr_runtime.runtime import Runtime

REFUSALS = {INVALID_QUESTION, MAX_LENGTH_EXCEEDED}
TIE = 1e-8
SUM_TOLERANCE = 1e-5
PROFILE = "exact"


class RefusalError(ValueError):
    """The model declined a question of the row."""


def _probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def check(
    response: dict[str, Any], questions: dict[str, dict[str, Any]], model: str
) -> dict[str, Any]:
    """The response unchanged if it answers every question validly; raises ``RefusalError`` or ``ValueError``."""
    if response.get("model") != model:
        raise ValueError("the runtime answered under another model identity")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("the runtime returned missing or extra answers")
    errors = {
        a["error"] for a in answers.values() if isinstance(a, dict) and "error" in a
    }
    if errors - REFUSALS:
        raise ValueError(f"the runtime answered {sorted(errors - REFUSALS)}")
    if errors:
        raise RefusalError(",".join(sorted(errors)))
    for key, question in questions.items():
        answer = answers[key]
        if answer.get("type") != question.get("type"):
            raise ValueError("the runtime returned the wrong answer type")
        if question["type"] == "noul":
            if not _probability(answer.get("noul")):
                raise ValueError("the runtime returned an invalid Noul probability")
            continue
        options = list(question.get("criteria") or {})
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(options):
            raise ValueError("the runtime omitted or added option probabilities")
        values = [probabilities[option] for option in options]
        if not all(_probability(value) for value in values) or not math.isclose(
            sum(values), 1.0, rel_tol=0, abs_tol=SUM_TOLERANCE
        ):
            raise ValueError("the runtime returned an invalid option distribution")
        top = max(values)
        first = next(
            o for o, v in zip(options, values, strict=True) if abs(v - top) <= TIE
        )
        if answer.get("choice") != first:
            raise ValueError("the choice is not the first most probable option")
    return response


class RuntimeIndexEngine:
    """Duck-typed kit engine: the row's original state and questions in, the runtime's answers out."""

    name = "vllm-sr-runtime-exact"
    latency = (
        "In-process request wall time through the runtime's scheduler on the exact profile: "
        "planning, the request's forwards, answers and validation; excludes model loading."
    )

    def __init__(self, *, model: str, revision: str, device: str = "rocm:0") -> None:
        self.autotune_cache = os.environ.get("INDEX_ENGINE_AUTOTUNE_CACHE")
        self.tuned_before = autotune_entries(self.autotune_cache)
        self.service = Runtime(
            ServeConfig(
                models=(ModelConfig(model=model, revision=revision, device=device),),
                cache_dir=os.environ.get("INDEX_ENGINE_CACHE_DIR"),
                offline=True,
                autotune_cache=self.autotune_cache,
            )
        )
        self.service.load()
        self.served = self.service.lookup(None)
        assert self.served.model is not None and self.served.placement is not None
        info = self.served.model.info
        self.model_id = self.served.served_id
        self.provenance = {
            "kind": "vllm-sr-runtime-builtin",
            "model_id": info.repo,
            "revision": info.revision,
            "model_sha256": info.model_sha256,
            "profile": PROFILE,
            "policy": (
                "Original state and questions as one exact-profile request through the runtime's "
                "scheduler; a refused question makes the row unsupported; malformed answers are errors."
            ),
        }

    def __call__(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], None]:
        from decision_index.engines import Unsupported

        prepared = self.service.prepare(
            "decisions",
            {
                "state": state,
                "questions": questions,
                "options": {"profile": PROFILE, "return_meta": False},
            },
        )
        results = self.served.submit_items(prepared.plan.items, None, PROFILE).result()
        response = self.service.finish(prepared, results, 0.0, 0.0)
        try:
            return check(response, questions, self.model_id), None
        except RefusalError as exc:
            raise Unsupported(str(exc)) from exc

    def warmup(self) -> None:
        question = {
            "warmup": {
                "type": "choice",
                "instructions": "Which color is named?",
                "criteria": {"red": "red", "blue": "blue"},
            }
        }
        for _ in range(2):
            self("The color is red.", question)

    def synchronize(self) -> None:
        import torch

        if self.served.placement.device.accelerator != "cpu":
            torch.cuda.synchronize()

    def runtime_info(self) -> dict[str, Any]:
        placement = self.served.placement
        return {
            "device": placement.device.label,
            "device_name": placement.device.name,
            "golden": self.served.health.golden.describe(),
            "kernel_choices": kernel_choices(),
            "versions": versions(),
        }

    def runtime(self) -> dict[str, Any]:
        return self.runtime_info()

    def close(self) -> None:
        receipt = os.environ.get("INDEX_ENGINE_RECEIPT")
        if receipt:
            engine = self.served.model.engine_model
            document = {
                "schema": "model-runtime-index-engine/1",
                **self.provenance,
                **self.runtime_info(),
                "autotune_entries": {
                    "before": self.tuned_before,
                    "after": autotune_entries(self.autotune_cache),
                },
                "fast_path": engine.receipt() if hasattr(engine, "receipt") else None,
            }
            Path(receipt).write_text(
                json.dumps(document, indent=1) + "\n", encoding="utf-8"
            )
        self.service.stop()
