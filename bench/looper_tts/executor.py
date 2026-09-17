"""Execute a planned Looper TTS matrix with auditable call accounting.

The executor intentionally lives beside the versioned benchmark contract.  It
does not mutate the semantic-router production configuration: each manifest
arm is expanded into the same four stage families (direct, confidence,
ReMoM, and Fusion) against an injected OpenAI-compatible provider.  An
injected provider also makes the whole path testable without credentials or a
running model server.

PR2 owns generation and accounting.  Candidate replay, repeated sampling and
native grading are separate phases and therefore remain outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import requests

from . import SCHEMA_VERSION
from .accounting import (
    BudgetExhausted,
    BudgetLedger,
    BudgetSnapshot,
    Pricing,
    Settlement,
    Usage,
    cost_for_usage,
    estimate_prompt_tokens,
    estimate_total_tokens,
)
from .plan import validate_plan
from .records import validate_records
from .validation import ContractError, digest, load_json, write_json


RUNTIME_SCHEMA_VERSION = "looper-tts-runtime.v1"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_OUTPUT_TOKENS = 1024


@dataclass
class ProviderResponse:
    """Normalized result returned by a provider adapter."""

    content: str = ""
    reasoning: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Any = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class Provider(Protocol):
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        stage: str,
    ) -> ProviderResponse: ...


class RequestsProvider:
    """One-attempt OpenAI-compatible HTTP provider.

    Retries belong to :class:`LooperTTSExecutor` so each retry receives its
    own call ID, reservation, usage row, latency and raw artifact.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        headers: Optional[Mapping[str, str]] = None,
    ):
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint:
            raise ValueError("provider endpoint is required")
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("provider timeout must be a positive integer")
        self.endpoint = (
            endpoint
            if endpoint.endswith("/chat/completions")
            else endpoint + "/chat/completions"
        )
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if headers:
            self.headers.update(dict(headers))

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        stage: str,
    ) -> ProviderResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        headers = dict(self.headers)
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        started = time.perf_counter()
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            return ProviderResponse(
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=str(error),
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != requests.codes.ok:
            return ProviderResponse(
                latency_ms=latency_ms,
                error="HTTP {}: {}".format(response.status_code, response.text[:500]),
            )
        try:
            body = response.json()
        except ValueError as error:
            return ProviderResponse(
                latency_ms=latency_ms, error="invalid JSON response: " + str(error)
            )
        if not isinstance(body, Mapping):
            return ProviderResponse(
                latency_ms=latency_ms, error="provider response must be an object"
            )
        content, reasoning = _response_text(body)
        return ProviderResponse(
            content=content,
            reasoning=reasoning,
            usage=Usage.from_mapping(body.get("usage")),
            raw=dict(body),
            latency_ms=latency_ms,
        )


class DeterministicProvider:
    """Offline provider for smoke tests and CI.

    Answers depend only on model, stage, seed and messages.  ``fail_models``
    exercises error accounting; ``omit_usage`` exercises the unknown-usage
    path.  It is a provider adapter, not benchmark evidence.
    """

    def __init__(
        self,
        fail_models: Optional[Iterable[str]] = None,
        omit_usage: bool = False,
        completion_tokens: int = 8,
    ):
        self.fail_models = set(fail_models or ())
        self.omit_usage = omit_usage
        self.completion_tokens = max(0, int(completion_tokens))
        self.calls: List[Tuple[str, str]] = []
        self._lock = threading.Lock()

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        stage: str,
    ) -> ProviderResponse:
        with self._lock:
            self.calls.append((stage, model))
        if model in self.fail_models:
            return ProviderResponse(
                error="deterministic provider failure for " + model, latency_ms=0.0
            )
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        if stage == "verify":
            content = json.dumps({"confidence": 1.0, "reason": "deterministic fixture"})
        else:
            marker = hashlib.sha256(
                (model + "\0" + stage + "\0" + str(seed) + "\0" + prompt).encode(
                    "utf-8"
                )
            ).hexdigest()[:12]
            content = "deterministic answer " + marker
        prompt_tokens = estimate_prompt_tokens(messages)
        completion = min(self.completion_tokens, max_tokens)
        usage: Any = {}
        if not self.omit_usage:
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion,
                "total_tokens": prompt_tokens + completion,
            }
        raw = {
            "id": "deterministic-"
            + hashlib.sha256(content.encode("utf-8")).hexdigest()[:12],
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        return ProviderResponse(
            content=content,
            usage=Usage.from_mapping(usage),
            raw=raw,
            latency_ms=0.0,
        )


def _response_text(body: Mapping[str, Any]) -> Tuple[str, str]:
    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return "", ""
    if not isinstance(message, Mapping):
        return "", ""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return str(content), str(reasoning)


@dataclass
class _CallOutcome:
    response: Optional[ProviderResponse]
    call_ids: List[str]
    error: Optional[str] = None


@dataclass
class _ItemState:
    experiment_id: str
    cell_id: str
    item_id: str
    budget: Mapping[str, Any]
    ledger: BudgetLedger
    calls: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    candidate_scores: List[Dict[str, Any]] = field(default_factory=list)
    attempt_numbers: Dict[Tuple[str, str], int] = field(default_factory=dict)
    budget_error: Optional[str] = None
    budget_exhausted: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


class LooperTTSExecutor:
    """Run each planned item and emit normalized records plus a receipt."""

    def __init__(
        self,
        plan: Mapping[str, Any],
        provider: Provider,
        output_dir: Path,
        retries: int = 0,
        max_output_tokens: Optional[int] = None,
        max_workers: int = 8,
    ):
        validate_plan(dict(plan))
        if type(retries) is not int or retries < 0:
            raise ValueError("retries must be non-negative")
        if max_output_tokens is not None and (
            type(max_output_tokens) is not int or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be positive when set")
        if type(max_workers) is not int or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        self.plan = plan
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.retries = retries
        self.max_output_tokens = max_output_tokens
        self.max_workers = max(1, max_workers)
        config = plan["config"]
        self.models = {model["id"]: model for model in config["models"]}
        self.arms = {arm["id"]: arm for arm in config["arms"]}
        self.items = {item["id"]: item for item in config["dataset"]["items"]}
        self.budgets = {budget["id"]: budget for budget in config["budgets"]}
        self.scorer_id = config["scorer"]["id"]
        self._receipt_cells: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._receipt_cells.clear()
        records: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.plan["experiment_id"],
            "evidence_kind": self.plan["config"]["dataset"]["evidence_kind"],
            "calls": [],
            "results": [],
        }
        for cell in self.plan["matrix"]:
            for item_id in cell["item_ids"]:
                item = self.items[item_id]
                try:
                    result, calls, receipt = self._run_item(cell, item)
                except Exception as error:  # keep the paired matrix intact
                    result = self._terminal_error(
                        cell, item, str(error), budget_exhausted=False
                    )
                    calls = []
                    receipt = {
                        "cell_id": cell["id"],
                        "item_id": item["id"],
                        "status": "error",
                        "error": str(error)[:1000],
                        "budget": None,
                        "calls": [],
                    }
                records["calls"].extend(calls)
                records["results"].append(result)
                self._receipt_cells.append(receipt)
        # This validation catches executor bugs before files are published.
        validate_records(records, self.plan)
        write_json(self.output_dir / "records.json", records)
        write_json(self.output_dir / "runtime_receipt.json", self._runtime_receipt())
        return records

    def _run_item(
        self, cell: Mapping[str, Any], item: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        budget = self.budgets[cell["budget_id"]]
        state = _ItemState(
            experiment_id=self.plan["experiment_id"],
            cell_id=cell["id"],
            item_id=item["id"],
            budget=budget,
            ledger=BudgetLedger(budget["max_calls"], budget["max_total_tokens"]),
        )
        arm = self.arms[cell["arm_id"]]
        prompt = item["prompt"]
        algorithm = arm["algorithm"]
        try:
            if algorithm == "direct":
                final = self._run_direct(state, arm, prompt, cell["seed"])
            elif algorithm == "confidence":
                final = self._run_confidence(state, arm, prompt, cell["seed"])
            elif algorithm == "remom":
                final = self._run_remom(state, arm, prompt, cell["seed"])
            elif algorithm == "fusion":
                final = self._run_fusion(state, arm, prompt, cell["seed"])
            else:
                final = _CallOutcome(None, [], "unsupported algorithm " + algorithm)
        except Exception as error:  # preserve already-paid calls in the cohort
            final = _CallOutcome(
                None,
                [call["id"] for call in state.calls],
                "executor failure: " + str(error),
            )

        # Parallel ReMoM/Fusion workers finish in provider-dependent order.
        # Canonicalize the evidence before exposing it so replay joins and
        # deterministic fake runs do not depend on thread scheduling.
        with state.lock:
            state.calls.sort(key=_call_sort_key)
            state.events.sort(key=_event_sort_key)

        snapshot = state.ledger.snapshot()
        if state.budget_exhausted or snapshot.exhausted:
            state.budget_exhausted = True
        answer = final.response.content if final.response is not None else None
        if not answer and final.response is not None:
            answer = final.response.reasoning or None
        if state.budget_exhausted:
            status = "budget_exhausted"
            error = state.budget_error or final.error or "budget exhausted"
        elif answer is not None and not final.error:
            status = "success"
            error = None
        else:
            status = "error"
            error = final.error or "algorithm produced no usable answer"
        result = {
            "id": digest(
                {
                    "experiment_id": self.plan["experiment_id"],
                    "cell_id": cell["id"],
                    "item_id": item["id"],
                    "kind": "result",
                }
            ),
            "experiment_id": self.plan["experiment_id"],
            "cell_id": cell["id"],
            "item_id": item["id"],
            "status": status,
            "final_answer": answer,
            "score": None,
            "scorer_id": self.scorer_id,
            "call_ids": [call["id"] for call in state.calls],
            "candidate_scores": list(state.candidate_scores),
            "panel_sha256": getattr(final, "panel_sha256", None),
            "budget_status": "exhausted" if state.budget_exhausted else "within",
            "error": error,
        }
        receipt = {
            "cell_id": cell["id"],
            "item_id": item["id"],
            "status": status,
            "budget": {
                "max_calls": budget["max_calls"],
                "max_total_tokens": budget["max_total_tokens"],
                "max_output_tokens": self._max_output_tokens(budget),
                "snapshot": _snapshot_record(snapshot),
            },
            "calls": list(state.events),
        }
        return result, list(state.calls), receipt

    def _run_direct(
        self, state: _ItemState, arm: Mapping[str, Any], prompt: str, seed: int
    ) -> _CallOutcome:
        model_id = arm["model_ids"][0]
        return self._call(state, "generate", model_id, _messages(prompt), seed)

    def _run_confidence(
        self, state: _ItemState, arm: Mapping[str, Any], prompt: str, seed: int
    ) -> _CallOutcome:
        threshold = float(arm["parameters"]["threshold"])
        last: Optional[_CallOutcome] = None
        for index, model_id in enumerate(arm["model_ids"]):
            generated = self._call(
                state,
                "generate",
                model_id,
                _messages(prompt),
                _derived_seed(seed, "generate", index),
            )
            last = generated
            response = generated.response
            if response is None:
                if state.budget_exhausted:
                    return generated
                continue
            candidate_call = generated.call_ids[-1] if generated.call_ids else None
            verification = self._call(
                state,
                "verify",
                model_id,
                _messages(_verification_prompt(prompt, response.content)),
                _derived_seed(seed, "verify", index),
            )
            score = _confidence_score(verification.response)
            if candidate_call is not None:
                state.candidate_scores.append(
                    {"call_id": candidate_call, "score": score}
                )
            if score is not None and score >= threshold:
                return _CallOutcome(
                    response, generated.call_ids + verification.call_ids
                )
            if state.budget_exhausted:
                return _CallOutcome(
                    response,
                    generated.call_ids + verification.call_ids,
                    state.budget_error,
                )
            last = _CallOutcome(response, generated.call_ids + verification.call_ids)
        return last or _CallOutcome(
            None, [], "confidence cascade produced no candidate"
        )

    def _run_remom(
        self, state: _ItemState, arm: Mapping[str, Any], prompt: str, seed: int
    ) -> _CallOutcome:
        schedule = list(arm["parameters"]["breadth"]) + [1]
        references: List[str] = []
        all_candidates: List[ProviderResponse] = []
        all_call_ids: List[str] = []
        final_response: Optional[ProviderResponse] = None
        for round_index, width in enumerate(schedule):
            is_final = round_index == len(schedule) - 1
            stage = "synthesize" if is_final else "generate"
            reference_text = ""
            if references:
                reference_text = "\n\nReference responses:\n" + "\n---\n".join(
                    references
                )
            round_prompt = prompt + reference_text
            model_ids = [
                arm["model_ids"][i % len(arm["model_ids"])] for i in range(width)
            ]
            outcomes = self._parallel_calls(
                state,
                stage,
                model_ids,
                round_prompt,
                _derived_seed(seed, stage, round_index),
            )
            usable = [
                outcome.response
                for outcome in outcomes
                if outcome.response and _usable(outcome.response)
            ]
            for outcome in outcomes:
                all_call_ids.extend(outcome.call_ids)
                if outcome.response is not None and not is_final:
                    all_candidates.append(outcome.response)
            if not usable:
                if state.budget_exhausted:
                    return _CallOutcome(
                        final_response, all_call_ids, state.budget_error
                    )
                return _CallOutcome(
                    final_response,
                    all_call_ids,
                    "ReMoM round produced no usable response",
                )
            references = [response.content for response in usable if response.content]
            if is_final:
                final_response = next(
                    (response for response in usable if response.content), usable[0]
                )
            if not is_final:
                for outcome in outcomes:
                    if (
                        outcome.response is not None
                        and _usable(outcome.response)
                        and outcome.call_ids
                    ):
                        state.candidate_scores.append(
                            {"call_id": outcome.call_ids[-1], "score": None}
                        )
            if state.budget_exhausted and not is_final:
                break
        if final_response is None:
            final_response = next(
                (response for response in reversed(all_candidates) if response.content),
                None,
            )
        return _CallOutcome(
            final_response,
            all_call_ids,
            state.budget_error if state.budget_exhausted else None,
        )

    def _run_fusion(
        self, state: _ItemState, arm: Mapping[str, Any], prompt: str, seed: int
    ) -> _CallOutcome:
        params = arm["parameters"]
        panel_ids = list(params["panel_model_ids"])
        panel_outcomes = self._parallel_calls(
            state,
            "generate",
            panel_ids,
            prompt,
            _derived_seed(seed, "panel", 0),
        )
        panel_entries = [
            (index, outcome.response)
            for index, outcome in enumerate(panel_outcomes)
            if outcome.response and _usable(outcome.response)
        ]
        panel = [response for _, response in panel_entries]
        call_ids = [
            call_id for outcome in panel_outcomes for call_id in outcome.call_ids
        ]
        for outcome in panel_outcomes:
            if outcome.response is not None:
                call_id = outcome.call_ids[-1] if outcome.call_ids else None
                if call_id:
                    state.candidate_scores.append({"call_id": call_id, "score": None})
        if not panel:
            return _CallOutcome(
                None,
                call_ids,
                state.budget_error or "Fusion panel produced no usable response",
            )
        panel_block = "\n---\n".join(
            "[{}]\n{}".format(panel_ids[index], response.content)
            for index, response in panel_entries
        )
        panel_hash = digest(
            [
                {"model": panel_ids[index], "content": response.content}
                for index, response in panel_entries
            ]
        )
        judge = self._call(
            state,
            "judge",
            params["judge_model_id"],
            _messages(_judge_prompt(prompt, panel_block)),
            _derived_seed(seed, "judge", 0),
        )
        call_ids.extend(judge.call_ids)
        if judge.response is None:
            return _CallOutcome(None, call_ids, state.budget_error or judge.error)
        synthesis = self._call(
            state,
            "synthesize",
            params["synthesis_model_id"],
            _messages(_synthesis_prompt(prompt, panel_block, judge.response.content)),
            _derived_seed(seed, "synthesize", 0),
        )
        call_ids.extend(synthesis.call_ids)
        if synthesis.response is None:
            return _CallOutcome(
                judge.response, call_ids, state.budget_error or synthesis.error
            )
        outcome = _CallOutcome(
            synthesis.response,
            call_ids,
            state.budget_error if state.budget_exhausted else None,
        )
        # _CallOutcome is intentionally small; attach the optional panel hash
        # for the normalized result without changing its public constructor.
        setattr(outcome, "panel_sha256", panel_hash)
        return outcome

    def _parallel_calls(
        self,
        state: _ItemState,
        stage: str,
        model_ids: Sequence[str],
        prompt: str,
        seed: int,
    ) -> List[_CallOutcome]:
        if len(model_ids) <= 1:
            attempt_base = self._reserve_attempt_block(state, stage, model_ids[0])
            return [
                self._call(
                    state,
                    stage,
                    model_ids[0],
                    _messages(prompt),
                    seed,
                    attempt_base,
                )
            ]
        workers = min(self.max_workers, len(model_ids))
        results: List[Optional[_CallOutcome]] = [None] * len(model_ids)
        attempt_bases = [
            self._reserve_attempt_block(state, stage, model_id)
            for model_id in model_ids
        ]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._call,
                    state,
                    stage,
                    model_id,
                    _messages(prompt),
                    _derived_seed(seed, model_id, index),
                    attempt_bases[index],
                ): index
                for index, model_id in enumerate(model_ids)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as error:
                    results[index] = _CallOutcome(None, [], str(error))
        return [
            result or _CallOutcome(None, [], "parallel call did not complete")
            for result in results
        ]

    def _call(
        self,
        state: _ItemState,
        stage: str,
        model_id: str,
        messages: Sequence[Mapping[str, str]],
        seed: int,
        attempt_base: Optional[int] = None,
    ) -> _CallOutcome:
        model = self.models[model_id]
        max_tokens = self._max_output_tokens(state.budget)
        call_ids: List[str] = []
        last_error: Optional[str] = None
        if attempt_base is None:
            attempt_base = self._reserve_attempt_block(state, stage, model_id)
        for retry_index in range(self.retries + 1):
            attempt = attempt_base + retry_index
            call_id = digest(
                {
                    "experiment_id": state.experiment_id,
                    "cell_id": state.cell_id,
                    "item_id": state.item_id,
                    "stage": stage,
                    "model_id": model_id,
                    "attempt": attempt,
                }
            )
            estimate = estimate_total_tokens(messages, max_tokens)
            try:
                reservation = state.ledger.reserve(call_id, estimate)
            except BudgetExhausted as error:
                state.budget_exhausted = True
                state.budget_error = str(error)
                return _CallOutcome(None, call_ids, str(error))
            call_ids.append(call_id)
            started = time.perf_counter()
            try:
                response = self.provider.chat(
                    messages=messages,
                    model=model["model"],
                    temperature=float(model["sampling"]["temperature"]),
                    top_p=float(model["sampling"]["top_p"]),
                    max_tokens=max_tokens,
                    seed=seed,
                    stage=stage,
                )
            except Exception as error:  # adapter failures are paid attempts
                response = ProviderResponse(error=str(error))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not isinstance(response, ProviderResponse):
                response = _coerce_provider_response(response)
            if response.content is None:
                response.content = ""
            elif not isinstance(response.content, str):
                response.content = str(response.content)
            if response.reasoning is None:
                response.reasoning = ""
            elif not isinstance(response.reasoning, str):
                response.reasoning = str(response.reasoning)
            if response.error is not None and not isinstance(response.error, str):
                response.error = str(response.error)
            if not isinstance(response.usage, Usage):
                response.usage = Usage.from_mapping(response.usage)
            response.latency_ms = _normalized_latency(response.latency_ms, elapsed_ms)
            settlement = state.ledger.settle(reservation, response.usage)
            if settlement.exhausted:
                state.budget_exhausted = True
                state.budget_error = "provider usage exceeded token budget"
            call = self._call_record(
                state, call_id, stage, model_id, attempt, response, settlement
            )
            with state.lock:
                state.calls.append(call)
                state.events.append(self._event_record(call, settlement, model))
            if response.error:
                last_error = response.error[:1000]
                if state.budget_exhausted or retry_index >= self.retries:
                    break
                continue
            return _CallOutcome(response, call_ids)
        return _CallOutcome(
            None, call_ids, last_error or state.budget_error or "provider call failed"
        )

    def _reserve_attempt_block(
        self, state: _ItemState, stage: str, model_id: str
    ) -> int:
        """Allocate deterministic attempt IDs before parallel work starts."""
        with state.lock:
            key = (stage, model_id)
            base = state.attempt_numbers.get(key, 0) + 1
            state.attempt_numbers[key] = base + self.retries
            return base

    def _max_output_tokens(self, budget: Mapping[str, Any]) -> int:
        if self.max_output_tokens is not None:
            return self.max_output_tokens
        return max(
            1,
            min(
                DEFAULT_MAX_OUTPUT_TOKENS,
                budget["max_total_tokens"] // budget["max_calls"],
            ),
        )

    def _call_record(
        self,
        state: _ItemState,
        call_id: str,
        stage: str,
        model_id: str,
        attempt: int,
        response: ProviderResponse,
        settlement: Settlement,
    ) -> Dict[str, Any]:
        raw_path = None
        if response.error is None:
            raw_path = self._write_raw(state.cell_id, state.item_id, call_id, response)
        return {
            "id": call_id,
            "experiment_id": state.experiment_id,
            "cell_id": state.cell_id,
            "item_id": state.item_id,
            "stage": stage,
            "model_id": model_id,
            "attempt": attempt,
            "status": "error" if response.error else "success",
            "usage": response.usage.as_record(),
            "latency_ms": response.latency_ms,
            "raw_output_path": raw_path,
            "error": response.error[:1000] if response.error else None,
            "cache_id": None,
        }

    def _write_raw(
        self, cell_id: str, item_id: str, call_id: str, response: ProviderResponse
    ) -> str:
        # IDs are contract data and may contain path separators. Hashing the
        # directory components keeps raw artifacts inside the chosen output
        # root while retaining deterministic joins through the call record.
        relative = (
            Path("raw")
            / _artifact_component(cell_id)
            / _artifact_component(item_id)
            / (call_id + ".json")
        )
        destination = self.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = response.raw
        if raw is None:
            raw = {
                "content": response.content,
                "reasoning": response.reasoning,
                "usage": response.usage.as_record(),
            }
        try:
            encoded = json.dumps(raw, indent=2, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            encoded = json.dumps(
                {"content": response.content, "usage": response.usage.as_record()},
                ensure_ascii=False,
            )
        destination.write_text(encoded + "\n", encoding="utf-8")
        return relative.as_posix()

    @staticmethod
    def _event_record(
        call: Mapping[str, Any], settlement: Settlement, model: Mapping[str, Any]
    ) -> Dict[str, Any]:
        cost = cost_for_usage(settlement.usage, Pricing.from_model(model))
        return {
            "call_id": call["id"],
            "stage": call["stage"],
            "model_id": call["model_id"],
            "status": call["status"],
            "attempt": call["attempt"],
            "estimated_tokens": settlement.estimated_tokens,
            "charged_tokens": settlement.charged_tokens,
            "usage_source": settlement.usage_source,
            "usage_known": settlement.usage_known,
            "cost_usd": cost,
            "latency_ms": call["latency_ms"],
        }

    def _runtime_receipt(self) -> Dict[str, Any]:
        totals = {
            "calls": 0,
            "tokens": 0,
            "unknown_usage_calls": 0,
            "cost_usd": 0.0,
            "cost_known_calls": 0,
            "cost_unknown_calls": 0,
        }
        for cell in self._receipt_cells:
            snapshot = (
                cell.get("budget", {}).get("snapshot") if cell.get("budget") else None
            )
            if snapshot:
                totals["calls"] += snapshot["calls"]
                totals["tokens"] += snapshot["tokens"]
                totals["unknown_usage_calls"] += snapshot["unknown_usage_calls"]
            for event in cell.get("calls", []):
                if event["cost_usd"] is not None:
                    totals["cost_usd"] += event["cost_usd"]
                    totals["cost_known_calls"] += 1
                else:
                    totals["cost_unknown_calls"] += 1
        if totals["cost_unknown_calls"]:
            totals["cost_usd"] = None
        return {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "experiment_id": self.plan["experiment_id"],
            "config_sha256": self.plan["config_sha256"],
            "execution": {
                "retries": self.retries,
                "max_output_tokens": self.max_output_tokens,
                "effective_output_tokens_by_budget": {
                    budget_id: self._max_output_tokens(budget)
                    for budget_id, budget in self.budgets.items()
                },
                "provider": self.provider.__class__.__name__,
            },
            "totals": totals,
            "cells": self._receipt_cells,
        }

    def _terminal_error(
        self,
        cell: Mapping[str, Any],
        item: Mapping[str, Any],
        error: str,
        budget_exhausted: bool,
    ) -> Dict[str, Any]:
        return {
            "id": digest(
                {
                    "experiment_id": self.plan["experiment_id"],
                    "cell_id": cell["id"],
                    "item_id": item["id"],
                    "kind": "result",
                }
            ),
            "experiment_id": self.plan["experiment_id"],
            "cell_id": cell["id"],
            "item_id": item["id"],
            "status": "budget_exhausted" if budget_exhausted else "error",
            "final_answer": None,
            "score": None,
            "scorer_id": self.scorer_id,
            "call_ids": [],
            "candidate_scores": [],
            "panel_sha256": None,
            "budget_status": "exhausted" if budget_exhausted else "within",
            "error": error[:1000],
        }


def _messages(prompt: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _verification_prompt(prompt: str, answer: str) -> str:
    return (
        "Evaluate the candidate answer for the original question. Return only JSON "
        '{"confidence": number between 0 and 1}.\n\nOriginal question:\n'
        + prompt
        + "\n\nCandidate answer:\n"
        + answer
    )


def _judge_prompt(prompt: str, panel: str) -> str:
    return "Compare the panel answers and identify the correct answer. Return concise analysis.\n\nQuestion:\n{}\n\nPanel:\n{}".format(
        prompt, panel
    )


def _synthesis_prompt(prompt: str, panel: str, judge: str) -> str:
    return "Synthesize one final answer to the question using the panel and judge analysis.\n\nQuestion:\n{}\n\nPanel:\n{}\n\nJudge analysis:\n{}".format(
        prompt, panel, judge
    )


def _confidence_score(response: Optional[ProviderResponse]) -> Optional[float]:
    if response is None:
        return None
    text = response.content.strip()
    try:
        value = json.loads(text)
        score = value.get("confidence") if isinstance(value, Mapping) else None
    except (TypeError, ValueError):
        score = None
    if isinstance(score, bool):
        return None
    if isinstance(score, (int, float)) and math.isfinite(float(score)):
        return max(0.0, min(1.0, float(score)))
    return None


def _usable(response: ProviderResponse) -> bool:
    return bool(response.content.strip() or response.reasoning.strip())


def _derived_seed(seed: int, *parts: Any) -> int:
    payload = "\0".join([str(seed)] + [str(part) for part in parts]).encode("utf-8")
    # OpenAI-compatible seed values are signed 32-bit in several backends.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _snapshot_record(snapshot: BudgetSnapshot) -> Dict[str, Any]:
    return {
        "calls": snapshot.calls,
        "active_calls": snapshot.active_calls,
        "tokens": snapshot.tokens,
        "reserved_tokens": snapshot.reserved_tokens,
        "unknown_usage_calls": snapshot.unknown_usage_calls,
        "exhausted": snapshot.exhausted,
    }


_STAGE_ORDER = {
    "generate": 0,
    "verify": 1,
    "judge": 2,
    "synthesize": 3,
}


def _call_sort_key(call: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        _STAGE_ORDER.get(call.get("stage"), 99),
        str(call.get("model_id", "")),
        int(call.get("attempt", 0)),
        str(call.get("id", "")),
    )


def _event_sort_key(event: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        _STAGE_ORDER.get(event.get("stage"), 99),
        str(event.get("model_id", "")),
        int(event.get("attempt", 0)),
        str(event.get("call_id", "")),
    )


def _coerce_provider_response(value: Any) -> ProviderResponse:
    if isinstance(value, Mapping):
        content, reasoning = _response_text(value)
        return ProviderResponse(
            content=content,
            reasoning=reasoning,
            usage=Usage.from_mapping(value.get("usage")),
            raw=dict(value),
        )
    return ProviderResponse(error="provider returned unsupported response type")


def _normalized_latency(value: Any, fallback: float) -> float:
    """Keep provider supplied latency within the evidence number contract."""
    if isinstance(value, bool):
        return max(0.0, float(fallback))
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return max(0.0, float(fallback))
    if not math.isfinite(converted) or converted < 0:
        return max(0.0, float(fallback))
    return converted


def _artifact_component(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def execute_manifest(
    manifest_path: Path,
    output_dir: Path,
    endpoint: Optional[str] = None,
    api_key: str = "",
    retries: int = 0,
    max_output_tokens: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    fake: bool = False,
) -> Dict[str, Any]:
    """Load a saved manifest and execute it.

    ``fake=True`` is intentionally explicit; it produces synthetic smoke
    evidence and should never be used for a benchmark claim.
    """
    plan = load_json(manifest_path)
    validate_plan(plan)
    if fake:
        provider: Provider = DeterministicProvider()
    else:
        if not endpoint:
            raise ContractError("--endpoint is required unless --fake is set")
        provider = RequestsProvider(endpoint, api_key=api_key, timeout=timeout)
    return LooperTTSExecutor(
        plan,
        provider,
        output_dir=output_dir,
        retries=retries,
        max_output_tokens=max_output_tokens,
    ).run()
