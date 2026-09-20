"""Run ownership and sequentially durable bounded execution."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
from http import HTTPStatus

import requests

from cli.routing_preview import build_preview_request, case_request_fields

from .adapters import get_adapter
from .contracts import digest, plan, planned_cells
from .failures import failure_reason, failure_summary
from .provenance import capture_runner
from .report import BUCKETS, make_report
from .store import TERMINAL, now
from .transport import CallFailure, chat, effective_request

ARC_MAX_COLOR = 9


class ReviewedPlanChangedError(ValueError):
    """A new submission differs from its reviewed hash before any dispatch."""


def basic_grade(case, final):
    expected = case["answer"]
    if case["benchmark"] in {"mmlu-pro", "gpqa-diamond"}:
        text = final.strip()
        match = re.fullmatch(r"(?:ANSWER\s*:\s*)?\(?([A-J])\)?[.]?", text, re.I)
        answer = match.group(1).upper() if match else None
        return {
            "answer": answer,
            "correct": answer is not None and answer == str(expected).upper(),
            "score": float(answer is not None and answer == str(expected).upper()),
            "details": {"strict_format": match is not None},
        }
    if case["benchmark"] == "arc-agi-2":
        try:
            answer = json.loads(final)

            def valid_grid(grid):
                return (
                    isinstance(grid, list)
                    and bool(grid)
                    and bool(grid[0])
                    and all(
                        isinstance(row, list)
                        and len(row) == len(grid[0])
                        and all(type(x) is int and 0 <= x <= ARC_MAX_COLOR for x in row)
                        for row in grid
                    )
                )

            valid = (
                (
                    isinstance(answer, list)
                    and bool(answer)
                    and all(valid_grid(grid) for grid in answer)
                )
                if case.get("metadata", {}).get("output_format") == "grids"
                else valid_grid(answer)
            )
        except (ValueError, TypeError):
            answer = None
            valid = False
        correct = valid and answer == expected
        return {
            "answer": answer,
            "correct": correct,
            "score": float(correct),
            "details": {"valid_grid": valid},
        }
    raise ValueError("No basic grader for benchmark")


class Context:
    def __init__(self, engine, run_id, manifest, case, target, cancel, deadline):
        self.engine = engine
        self.store = engine.store
        self.run_id = run_id
        self.manifest = manifest
        self.case = case
        self.target = target
        self.limits = manifest["limits"]
        self.config = manifest.get("benchmark_options", {}).get(case["benchmark"], {})
        self._cancel = cancel
        self.deadline = min(deadline, time.monotonic() + self.limits["case_timeout_s"])
        self.calls = []
        self.call_slots = 0
        self.quality_failure = None
        self.artifact_dir = (
            self.store.root / "runs" / run_id / case["id"] / target["id"]
        )
        # IDs must not be used as paths without normalization.
        self.artifact_dir = (
            self.store.root / "runs" / run_id / digest([case["id"], target["id"]])[:24]
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def cancelled(self):
        return self._cancel.is_set() or time.monotonic() > self.deadline

    def call(self, messages, role="subject", target=None, extra_body=None):
        if self.cancelled():
            raise CallFailure("Run cancelled or wall-time budget exhausted")
        if self.quality_failure:
            raise CallFailure(
                "Case ended at the frozen output limit; no further calls are permitted"
            )
        selected = target or self.target
        if isinstance(selected, str):
            selected = next(
                (t for t in self.manifest["targets"] if t["id"] == selected),
                self.manifest.get("auxiliary_targets", {}).get(selected),
            )
            if selected is None:
                raise ValueError("Unknown auxiliary target reference")
        if role not in {"subject", "judge", "simulator"}:
            raise ValueError("Unknown call role")
        if role == "subject" and not any(
            call["role"] == "subject" for call in self.calls
        ):
            extra_body = {**case_request_fields(self.case), **(extra_body or {})}
        request_body = effective_request(
            selected, messages, self.manifest["sampling"], extra_body
        )
        reservation = 0
        if self.manifest["cost_policy"] == "require_priced":
            prices = list(selected.get("prices", {}).values())
            if not prices:
                raise CallFailure("Call has no frozen prices")
            count = (
                selected.get("max_inference_calls", 1)
                if selected["kind"] == "mom"
                else 1
            )
            input_bound = (
                len(json.dumps(request_body, ensure_ascii=False).encode())
                + 64 * len(messages)
                + 256
            )
            input_bound += (count - 1) * self.limits["max_output_chars"] * 4
            reservation = (
                count
                * (
                    input_bound
                    * max(
                        max(p["input"], p["cached_input"], p["cache_write"])
                        for p in prices
                    )
                    + request_body["max_tokens"] * max(p["output"] for p in prices)
                )
                / 1_000_000
            )
        with self.engine.lock:
            if self.call_slots >= self.limits["max_calls_per_case"]:
                raise CallFailure("Per-case call budget exhausted")
            committed = self.engine.spent.get(
                self.run_id, 0
            ) + self.engine.reserved.get(self.run_id, 0)
            if self.manifest["cost_policy"] == "require_priced" and (
                committed + reservation > self.limits["max_cost_usd"]
                or committed >= self.limits["max_cost_usd"]
            ):
                raise CallFailure(
                    "Insufficient cost budget for a bounded call reservation"
                )
            self.engine.reserved[self.run_id] = (
                self.engine.reserved.get(self.run_id, 0) + reservation
            )
            self.call_slots += 1
        limits = {
            **self.limits,
            "total_timeout_s": min(
                self.limits["total_timeout_s"],
                max(0.1, self.deadline - time.monotonic()),
            ),
        }
        call_id = self.store.start_call(
            self.run_id,
            self.case["id"],
            self.target["id"],
            role,
            {
                "model": selected["model"],
                "request": {
                    "messages": messages,
                    "sampling": self.manifest["sampling"],
                    "request_params": selected.get("request_params", {}),
                    "effective_body": request_body,
                    "extra_body": extra_body,
                },
            },
        )
        call_record = {"id": call_id, "role": role}
        self.calls.append(call_record)
        try:
            result = chat(
                selected,
                messages,
                self.manifest["sampling"],
                limits,
                self.cancelled,
                extra_body,
                self.artifact_dir / (call_id + ".sse"),
            )
            if (
                self.manifest["cost_policy"] == "require_priced"
                and result.get("cost_usd") is None
            ):
                raise CallFailure(
                    "Priced call returned incomplete usage or unknown model pricing",
                    result,
                )
            if (
                selected.get("max_inference_calls") is not None
                and result.get("inference_call_count") is not None
                and result["inference_call_count"] > selected["max_inference_calls"]
            ):
                raise CallFailure("Inference call count exceeded frozen bound", result)
            if not result.get("output_complete", True) and role != "subject":
                raise CallFailure(
                    "Auxiliary model output was truncated; grading is incomplete",
                    result,
                )
            self.store.finish_call(call_id, "completed", result)
            call_record.update(result)
            with self.engine.lock:
                self.engine.spent[self.run_id] = self.engine.spent.get(
                    self.run_id, 0
                ) + (result.get("cost_usd") or 0)
                self.engine.reserved[self.run_id] -= reservation
            if not result.get("output_complete", True):
                self.quality_failure = "output_limit"
            return result
        except CallFailure as exc:
            self.store.finish_call(
                call_id,
                "cancelled" if self.cancelled() else "failed",
                {**exc.partial, "error": str(exc)},
            )
            call_record.update(exc.partial)
            with self.engine.lock:
                self.engine.spent[self.run_id] = self.engine.spent.get(
                    self.run_id, 0
                ) + (exc.partial.get("cost_usd") or 0)
                # Unknown incurred cost keeps its reservation and stops the run.
                if exc.partial.get("cost_usd") is not None:
                    self.engine.reserved[self.run_id] -= reservation
            raise


class Engine:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.cancels = {}
        self.threads = {}
        self.spent = {}
        self.reserved = {}
        self.user_cancelled = set()
        self.first_failures = {}
        self.store.recover()

    def start(
        self,
        manifest,
        owner="local",
        request_key=None,
        recovery=False,
        *,
        actor_role="local",
    ):
        if manifest.get("recovery") and not recovery:
            raise ValueError("Recovery lineage requires the explicit recovery endpoint")
        frozen = plan(manifest)
        with self.store.lock:
            if request_key and (existing := self.store.request(owner, request_key)):
                if existing["manifest"]["plan_sha256"] != frozen["plan_sha256"]:
                    raise ValueError(
                        "idempotency key is already bound to a different plan"
                    )
                return existing
            if (
                "plan_sha256" in manifest
                and manifest["plan_sha256"] != frozen["plan_sha256"]
            ):
                raise ReviewedPlanChangedError(
                    "Reviewed plan changed; review a new frozen plan before starting"
                )
            run, created = self.store.create(
                frozen,
                owner,
                request_key,
                provenance=capture_runner(frozen),
                actor_role=actor_role,
            )
        if created:
            cancel = threading.Event()
            with self.lock:
                self.cancels[run["id"]] = cancel
                worker = threading.Thread(
                    target=self._run, args=(run["id"], frozen, cancel), daemon=True
                )
                self.threads[run["id"]] = worker
                worker.start()
        return self.store.get(run["id"])

    def cancel(self, run_id):
        run = self.store.get(run_id)
        if run["status"] in TERMINAL:
            return run
        with self.lock:
            cancel = self.cancels.get(run_id)
            if cancel:
                self.user_cancelled.add(run_id)
                cancel.set()
        self.store.event(run_id, "cancellation_requested", {})
        return self.store.get(run_id)

    def _case(self, run_id, manifest, case, target, cancel, deadline, enqueued_at):
        if cancel.is_set() or time.monotonic() > deadline:
            return
        queue_wait_s = max(0, time.monotonic() - enqueued_at)
        case_started_at = now()
        self.store.result(
            run_id,
            case["id"],
            target["id"],
            "running",
            {
                "benchmark": case["benchmark"],
                "queue_wait_s": queue_wait_s,
                "started_at": case_started_at,
            },
        )
        ctx = Context(self, run_id, manifest, case, target, cancel, deadline)
        ctx.queue_wait_s, ctx.started_at = queue_wait_s, case_started_at
        started = time.monotonic()
        try:
            if manifest["mode"] == "preview":
                headers = {"Content-Type": "application/json"}
                if target.get("preview_api_key_env") or target.get("api_key_env"):
                    key = os.environ.get(
                        target.get("preview_api_key_env") or target["api_key_env"]
                    )
                    if not key:
                        raise ValueError(
                            "Target credential environment variable is not set"
                        )
                    headers["Authorization"] = "Bearer " + key
                url = target["preview_url"]
                if target.get("config_hash"):
                    headers["X-SR-Bench-Expected-Config-Hash"] = target["config_hash"]
                payload = build_preview_request(
                    {
                        "messages": case["messages"],
                        **case_request_fields(case),
                        "model": target["model"],
                        "max_tokens": target.get("request_params", {}).get(
                            "max_tokens", manifest["sampling"]["max_tokens"]
                        ),
                        "options": {"trace": True},
                        "preview_context": manifest["preview_context"],
                    }
                )
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=min(
                        manifest["limits"]["total_timeout_s"],
                        manifest["limits"]["idle_timeout_s"],
                    ),
                )
                if response.status_code >= HTTPStatus.BAD_REQUEST:
                    raise ValueError(f"Preview HTTP {response.status_code}")
                routing = response.json()
                if (
                    target.get("config_hash")
                    and routing.get("config_hash") != target["config_hash"]
                ):
                    raise ValueError(
                        "Preview runtime identity acknowledgement missing or mismatched"
                    )
                result = {
                    "answer": None,
                    "correct": None,
                    "score": None,
                    "details": {"routing": routing},
                }
            else:
                result = get_adapter(case["benchmark"]).execute(case, ctx)
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("correct"), bool)
                    or result.get("score") not in {0, 1}
                ):
                    raise ValueError(
                        "Adapter result requires a terminal binary graded outcome"
                    )
            self._completed_case(ctx, result, started)
        except Exception as exc:
            if ctx.quality_failure:
                self._completed_case(ctx, {}, started)
                return
            # Exception strings from subprocesses/HTTP can contain secrets; only controlled errors escape.
            message = (
                str(exc)
                if isinstance(exc, (CallFailure, ValueError))
                else type(exc).__name__
            )
            self.store.result(
                run_id,
                case["id"],
                target["id"],
                "cancelled" if ctx.cancelled() else "failed",
                {
                    "benchmark": case["benchmark"],
                    "error": message,
                    "queue_wait_s": queue_wait_s,
                    "started_at": case_started_at,
                    "finished_at": now(),
                    "latency_s": time.monotonic() - started,
                    "partial": getattr(exc, "partial", {}),
                },
            )
            with self.lock:
                if run_id not in self.first_failures:
                    failure = {
                        "case_id": case["id"],
                        "target_id": target["id"],
                        "reason": failure_reason(message),
                        "inferred_from_saved_results": False,
                    }
                    self.first_failures[run_id] = failure
                    self.store.event(run_id, "failure_observed", failure)
            # Fail closed: no new cases are dispatched after a transport/harness failure.
            cancel.set()

    def _completed_case(self, ctx, result, started):
        if ctx.quality_failure:
            result = {
                "answer": None,
                "correct": False,
                "score": 0.0,
                "details": {
                    "quality_failure": ctx.quality_failure,
                    "output_complete": False,
                },
            }
        subject = [c for c in ctx.calls if c["role"] == "subject"]
        usage = (
            {k: sum(c["usage"][k] for c in subject) for k in BUCKETS}
            if subject and all(c.get("usage") is not None for c in subject)
            else None
        )
        cost = (
            sum(c["cost_usd"] for c in subject)
            if subject and all(c.get("cost_usd") is not None for c in subject)
            else None
        )
        data = {
            **result,
            "benchmark": ctx.case["benchmark"],
            "usage": usage,
            "cost_usd": cost,
            "latency_s": time.monotonic() - started,
            "subject_latency_s": sum(c.get("latency_s", 0) for c in subject),
            "ttft_s": subject[0].get("ttft_s") if subject else None,
            "call_count": len(ctx.calls),
            "queue_wait_s": ctx.queue_wait_s,
            "started_at": ctx.started_at,
            "finished_at": now(),
        }
        self.store.result(
            ctx.run_id, ctx.case["id"], ctx.target["id"], "completed", data
        )

    def _run(self, run_id, manifest, cancel):
        self.store.status(run_id, "running")
        enqueued_at = time.monotonic()
        deadline = enqueued_at + manifest["limits"]["max_run_seconds"]
        cases = {case["id"]: case for case in manifest["cases"]}
        targets = {target["id"]: target for target in manifest["targets"]}
        pending = iter(
            (cases[cell["case_id"]], targets[cell["target_id"]])
            for cell in planned_cells(manifest)
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=manifest["limits"]["concurrency"]
            ) as pool:
                active = set()

                def fill():
                    while (
                        not cancel.is_set()
                        and time.monotonic() < deadline
                        and len(active) < manifest["limits"]["concurrency"]
                    ):
                        pair = next(pending, None)
                        if pair is None:
                            break
                        active.add(
                            pool.submit(
                                self._case,
                                run_id,
                                manifest,
                                *pair,
                                cancel,
                                deadline,
                                enqueued_at,
                            )
                        )

                fill()
                while active:
                    done, active = concurrent.futures.wait(
                        active,
                        timeout=0.2,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        future.result()
                    if time.monotonic() >= deadline:
                        cancel.set()
                    fill()
            run = self.store.get(run_id)
            if run_id in self.user_cancelled:
                status = "cancelled"
            elif run["progress"]["failed"]:
                status = "failed"
            elif cancel.is_set():
                status = "cancelled"
            elif run["progress"]["completed"] == run["progress"]["total"]:
                status = "completed"
            else:
                status = "failed"
            failure = self.first_failures.get(run_id)
            self.store.status(
                run_id, status, failure_summary(failure) if failure else None
            )
            report = make_report(self.store, run_id)
            path = self.store.root / "runs" / run_id / "report.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.store.status(run_id, "failed", type(exc).__name__)
        finally:
            with self.lock:
                self.cancels.pop(run_id, None)
