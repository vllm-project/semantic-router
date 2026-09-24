"""Adapt owned Decision family runtimes to the physical row scheduler."""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import suppress
from typing import Any

from .backend import (
    BackendContractError,
    BackendInputTooLargeError,
    BackendUnavailableError,
)
from .model_inputs import (
    QWEN_DEFAULT_NO,
    QWEN_DEFAULT_YES,
    VELA_DEFAULT_NO,
    VELA_DEFAULT_YES,
    build_model_input,
)
from .physical_batching import (
    DecisionRow,
    DecisionRowResult,
    PreparedDecisionRow,
)
from .qwen35_inputs import EncodedQwenRow, encode_qwen_rows
from .runtime_profile import RuntimeProfile
from .vela_inputs import EncodedVelaRow, encode_vela_rows


class TorchDecisionRowExecutor:
    """Prepare complete model inputs and score compatible physical batches.

    The physical scheduler is the only caller of ``predict_rows``, so a resident
    model has one forward in flight. Tokenization and inference run off the
    event loop to keep health and admission responsive during model work.
    """

    def __init__(self, runtime: Any, profile: RuntimeProfile) -> None:
        if profile.family not in {"vela", "qwen3.5"}:
            raise ValueError("unsupported Decision model family")
        if runtime.max_length != profile.max_input_tokens:
            raise ValueError("resident model input limit differs from its profile")
        self._runtime = runtime
        self._profile = profile
        # Preparation may occupy every worker in the event loop's default
        # pool during a burst. Keep the resident model's single forward off
        # that queue so prepared physical batches can start promptly.
        self._inference_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="decision-inference"
        )
        # Qwen row preparation does substantial Python work around the
        # tokenizer. Limit simultaneous preparations so they cannot occupy
        # every CPU thread while a physical GPU forward is launching.
        self._preparation_limit = (
            asyncio.Semaphore(4) if profile.family == "qwen3.5" else None
        )
        self._closed = False

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            self._inference_pool.shutdown(wait=True, cancel_futures=True)

    async def ready(self) -> bool:
        return not self._closed

    async def prepare_rows(
        self, rows: tuple[DecisionRow, ...]
    ) -> tuple[PreparedDecisionRow, ...]:
        if self._closed:
            raise BackendUnavailableError("Decision row executor is closed")
        if not rows:
            raise BackendContractError("a Decision request contains no rows")
        try:
            if self._preparation_limit is None:
                encoded = await _finish_thread_operation(self._encode_rows, rows)
            else:
                async with self._preparation_limit:
                    if self._closed:
                        raise BackendUnavailableError("Decision row executor is closed")
                    encoded = await _finish_thread_operation(self._encode_rows, rows)
        except ValueError as error:
            if "max_length" in str(error) or "no room for state" in str(error):
                raise BackendInputTooLargeError(str(error)) from error
            raise BackendContractError("Decision row preparation failed") from error
        family = self._profile.family
        return tuple(
            PreparedDecisionRow(
                row=row,
                batch_key=(
                    f"{family}:{row.question.type}" if family == "vela" else family
                ),
                payload=payload,
            )
            for row, payload in zip(rows, encoded, strict=True)
        )

    def _encode_rows(self, rows: tuple[DecisionRow, ...]):
        family = self._profile.family
        if family == "vela":
            defaults = (VELA_DEFAULT_NO, VELA_DEFAULT_YES)
            explicit_null = "use_default"
        else:
            defaults = (QWEN_DEFAULT_NO, QWEN_DEFAULT_YES)
            explicit_null = "preserve_json_null"
        inputs = tuple(
            build_model_input(
                question_id=row.question_id,
                state=row.state,
                question=row.question,
                choice_null_description=(
                    self._profile.prompt_policy.choice_null_description
                ),
                noul_default_false=defaults[0],
                noul_default_true=defaults[1],
                noul_explicit_null=explicit_null,
            )
            for row in rows
        )
        if family == "vela":
            return encode_vela_rows(
                inputs,
                self._runtime.tokenizer,
                max_length=self._profile.max_input_tokens,
            )
        return encode_qwen_rows(
            inputs,
            self._runtime.tokenizer,
            max_length=self._profile.max_input_tokens,
        )

    async def predict_rows(
        self, rows: tuple[PreparedDecisionRow, ...]
    ) -> tuple[DecisionRowResult, ...]:
        if self._closed:
            raise BackendUnavailableError("Decision row executor is closed")
        if not rows or any(row.batch_key != rows[0].batch_key for row in rows):
            raise BackendContractError("Decision physical batch is incompatible")
        payload_type = (
            EncodedVelaRow if self._profile.family == "vela" else EncodedQwenRow
        )
        if any(
            not isinstance(item.payload, payload_type)
            or item.payload.question_id != item.row.question_id
            or item.payload.type != item.row.question.type
            for item in rows
        ):
            raise BackendContractError("Decision prepared row identity changed")
        payloads = tuple(item.payload for item in rows)
        predictions = await _finish_thread_operation(
            self._runtime.predict_encoded, payloads, executor=self._inference_pool
        )
        if not isinstance(predictions, tuple) or len(predictions) != len(rows):
            raise BackendContractError("Decision model changed the physical batch")
        results = []
        for item, prediction in zip(rows, predictions, strict=True):
            if (
                prediction.question_id != item.row.question_id
                or prediction.type != item.row.question.type
            ):
                raise BackendContractError("Decision model changed question identity")
            results.append(
                DecisionRowResult(
                    question_id=prediction.question_id,
                    type=prediction.type,
                    probabilities=prediction.probabilities,
                    input_tokens=prediction.input_tokens,
                )
            )
        return tuple(results)


async def _finish_thread_operation(
    operation, payloads, *, executor: Executor | None = None
):
    """Do not abandon a worker thread when its awaiting coroutine is cancelled."""

    if executor is None:
        task = asyncio.create_task(asyncio.to_thread(operation, payloads))
    else:
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(
            loop.run_in_executor(executor, operation, payloads)
        )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except Exception:
                break
        with suppress(Exception):
            task.result()
        raise
