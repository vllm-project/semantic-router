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
from .family_registry import family_adapter
from .physical_batching import (
    DecisionRow,
    DecisionRowResult,
    PreparedDecisionRow,
)
from .runtime_profile import RuntimeProfile


class TorchDecisionRowExecutor:
    """Prepare complete model inputs and score compatible physical batches.

    The physical scheduler is the only caller of ``predict_rows``, so a resident
    model has one forward in flight. Tokenization and inference run off the
    event loop to keep health and admission responsive during model work.
    """

    def __init__(self, runtime: Any, profile: RuntimeProfile) -> None:
        self._adapter = family_adapter(profile.family)
        if runtime.max_length != profile.max_input_tokens:
            raise ValueError("resident model input limit differs from its profile")
        self._runtime = runtime
        self._profile = profile
        # Families with expensive parallel preparation can reserve a forward
        # worker so inference is not queued behind the default executor.
        self._inference_pool = (
            ThreadPoolExecutor(
                max_workers=self._adapter.inference_workers,
                thread_name_prefix="decision-inference",
            )
            if self._adapter.inference_workers is not None
            else None
        )
        self._closed = False

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            if self._inference_pool is not None:
                self._inference_pool.shutdown(wait=True, cancel_futures=True)

    async def ready(self) -> bool:
        return not self._closed

    async def prepare_rows(
        self, rows: tuple[DecisionRow, ...]
    ) -> tuple[PreparedDecisionRow, ...]:
        if not rows:
            raise BackendContractError("a Decision request contains no rows")
        try:
            encoded = await _finish_thread_operation(self._encode_rows, rows)
        except ValueError as error:
            if "max_length" in str(error) or "no room for state" in str(error):
                raise BackendInputTooLargeError(str(error)) from error
            raise BackendContractError("Decision row preparation failed") from error
        return tuple(
            PreparedDecisionRow(
                row=row,
                batch_key=self._adapter.batch_key(row.question.type),
                payload=payload,
            )
            for row, payload in zip(rows, encoded, strict=True)
        )

    def _encode_rows(self, rows: tuple[DecisionRow, ...]):
        return self._adapter.encode_rows(rows, self._runtime.tokenizer, self._profile)

    async def predict_rows(
        self, rows: tuple[PreparedDecisionRow, ...]
    ) -> tuple[DecisionRowResult, ...]:
        if self._closed:
            raise BackendUnavailableError("Decision row executor is closed")
        if not rows or any(row.batch_key != rows[0].batch_key for row in rows):
            raise BackendContractError("Decision physical batch is incompatible")
        payload_type = self._adapter.encoded_row_type
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
