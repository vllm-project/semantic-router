"""Drive and read one live sr-bench experiment through its public CLI."""

import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contract import ContractError
from .deployed_config import config_identity
from .loop_evidence import check, digest, sha, target


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def persist(directory, receipt):
    """Content-address each observation, including the run ID before waiting."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = encoded(receipt) + b"\n"
    path = directory / (digest(data) + ".json")
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        staged = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            staged.unlink(missing_ok=True)
            raise
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)
    return str(path.resolve())


def save_result(workspace, receipt):
    """Keep the raw machine evidence retrievable; the report helper is #5492."""
    receipt["receipt"] = persist(workspace / "sr-bench", receipt)


class SrBench:
    """Keep dispatch idempotent and leave credentials to vllm-sr."""

    def __init__(self, executable):
        self.executable = executable

    def command(self, command, *arguments):
        result = subprocess.run(  # noqa: PLW1510 - the wrapper checks the code
            [self.executable, "benchmark", command, *arguments],
            capture_output=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=120,
        )
        check(
            result.returncode == 0,
            f"vllm-sr benchmark {command} exited {result.returncode}",
        )
        try:
            document = json.loads(result.stdout)
        except ValueError as exc:
            raise ValueError(f"benchmark {command} returned invalid JSON") from exc
        check(isinstance(document, dict), f"benchmark {command} returned no object")
        return document

    def submit(self, manifest, request_key):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(encoded(manifest))
            return self.command(
                "run",
                "--manifest",
                str(path),
                "--detach",
                "--idempotency-key",
                request_key,
            )["id"]

    def rows(self, run_id, target_id, expected):
        cursor = 0
        rows = {}
        while True:
            page = self.command(
                "show", run_id, "--results", "--after", str(cursor), "--limit", "500"
            )
            for row in page["results"]:
                if row["target_id"] != target_id:
                    continue
                case = row["case_id"]
                check(
                    case in expected and case not in rows,
                    "unexpected or duplicate result cell",
                )
                check(
                    row["status"] == "completed" and isinstance(row["correct"], bool),
                    f"missing terminal grade for {case}",
                )
                rows[case] = row
            following = page["next_cursor"]
            if following is None:
                break
            check(
                type(following) is int and following > cursor,
                "invalid sr-bench results cursor",
            )
            cursor = following
        check(set(rows) == expected, "missing paired result cells")
        return rows


def subject(manifest, identity):
    subjects = [item for item in manifest["targets"] if item["id"] == identity]
    check(
        len(subjects) == 1, f"target {identity} missing or ambiguous in the frozen run"
    )
    return subjects[0]


def _selected_protocol(manifest, selected, both_mom):
    # candidate-plan/compare validate shared cases, graders and effective judge
    # targets. Compare ranks single models. The Lab additionally freezes the
    # actual subject's transport parameters, even when the baseline is a MoM.
    sampling = manifest["sampling"]
    check(
        isinstance(sampling, dict) and isinstance(manifest["limits"], dict),
        "missing frozen sampling or limits",
    )
    protocol = {
        "sampling": sampling,
        "request_params": {**sampling, **selected.get("request_params", {})},
        "limits": manifest["limits"],
        "output_policy": manifest.get("output_policy", "bounded"),
        "session_mode": selected.get("session_mode", "stateless"),
        "max_inference_calls": selected.get(
            "max_inference_calls", 1 if selected["kind"] == "single" else None
        ),
        "benchmark_options": manifest.get("benchmark_options", {}),
        "adapter_versions": manifest.get("adapter_versions", {}),
        "benchmark_weights": manifest.get("benchmark_weights", {}),
    }
    if protocol["output_policy"] == "native":
        check(
            bool(selected.get("native_limits")),
            "missing selected subject native_limits",
        )
        if both_mom:
            protocol["native_limits"] = selected["native_limits"]
        else:
            # A single-model arm only declares its own physical capacity. Use
            # sr-bench's per-physical-model inventory across arms in this case.
            limits = {}
            for item in [
                *manifest["targets"],
                *manifest.get("auxiliary_targets", {}).values(),
            ]:
                for model, capacity in item["native_limits"].items():
                    check(
                        limits.get(model, capacity) == capacity,
                        "conflicting native_limits",
                    )
                    limits[model] = capacity
            protocol["native_limits"] = limits
    return protocol


def _protocol_value(value):
    """Compare JSON numbers by value while keeping booleans distinct."""
    if isinstance(value, dict):
        return {k: _protocol_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_protocol_value(v) for v in value]
    if type(value) is float and value.is_integer():
        return int(value)
    return value


def validate_candidate(manifest, args, source, baseline, receipt):
    check(
        manifest["mode"] == "live"
        and manifest["baseline_run_id"] == args.baseline_run
        and manifest["case_sha256"] == baseline["case_sha256"]
        and manifest["profile"] == baseline["profile"]
        and manifest["seed"] == baseline["seed"]
        and manifest["cost_policy"] == baseline["cost_policy"]
        and manifest.get("execution_cells") is None
        and not manifest.get("recovery"),
        "candidate is not a full live evaluation of the frozen baseline",
    )
    check(len(manifest["targets"]) == 1, "expected one candidate target")
    selected = subject(manifest, args.candidate_target)
    parent = subject(baseline, args.baseline_target)
    both_mom = parent["kind"] == "mom"
    protocols = [
        _protocol_value(_selected_protocol(document, arm, both_mom))
        for document, arm in ((baseline, parent), (manifest, selected))
    ]
    original, proposed = protocols
    receipt["baseline_subject_protocol_sha256"] = digest(encoded(original))
    receipt["candidate_subject_protocol_sha256"] = digest(encoded(proposed))
    mismatches = sorted(
        k
        for k in original.keys() | proposed.keys()
        if encoded(original.get(k)) != encoded(proposed.get(k))
    )
    if "request_params" in mismatches:
        left, right = original["request_params"], proposed["request_params"]
        # Never retain the contents of nested template kwargs or stop strings.
        receipt["changed_request_fields"] = sorted(
            k
            for k in left.keys() | right.keys()
            if k not in left or k not in right or encoded(left[k]) != encoded(right[k])
        )
    check(not mismatches, "different frozen subject protocol: " + ", ".join(mismatches))
    check(
        selected["kind"] == "mom"
        and selected.get("config_hash") == args.active_runtime_hash
        and selected.get("capture_recipe") is True,
        "candidate target must capture the recipe and match the expected active runtime hash",
    )
    return target(source, selected["model"])["name"]


def verdict(baseline, candidate, guards, minimum):
    lost = sorted(
        case
        for case in baseline
        if baseline[case]["correct"] and not candidate[case]["correct"]
    )
    common = sorted(
        case
        for case in baseline
        if baseline[case]["correct"] and candidate[case]["correct"]
    )
    guard_losses = sorted(set(lost).intersection(guards))
    metrics = {
        "paired_cases": len(baseline),
        "baseline_correct": sum(row["correct"] for row in baseline.values()),
        "candidate_correct": sum(row["correct"] for row in candidate.values()),
        "common_successes": len(common),
        "lost_cases": lost,
        "guard_losses": guard_losses,
        "baseline_common_success_s": None,
        "candidate_common_success_s": None,
        "speedup_pct": None,
    }
    totals = []
    for rows in (baseline, candidate):
        values = [rows[case].get("latency_s") for case in common]
        if not values or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in values
        ):
            return (
                "INCONCLUSIVE",
                "missing positive whole-case timings on common successes",
                metrics,
            )
        totals.append(math.fsum(values))
    speedup = (totals[0] - totals[1]) / totals[0] * 100
    if not math.isfinite(speedup):
        return (
            "INCONCLUSIVE",
            "common-success speedup could not be represented",
            metrics,
        )
    metrics.update(
        baseline_common_success_s=totals[0],
        candidate_common_success_s=totals[1],
        speedup_pct=speedup,
    )
    if guard_losses or metrics["candidate_correct"] < metrics["baseline_correct"]:
        return "FAIL", "protected case lost or task success fell", metrics
    if speedup <= 0 or speedup < minimum:
        return "FAIL", "observed latency reduction did not meet the objective", metrics
    return "PASS", "observed quality and latency criteria met", metrics


def read_candidate(args, receipt):
    check(
        re.fullmatch(r"agent:\S+", args.proposer),
        "expected proposer agent:<id>",
    )
    check(
        0 <= args.min_speedup_pct <= MAX_SPEEDUP_PCT,
        f"expected a speedup threshold between 0 and {MAX_SPEEDUP_PCT}",
    )
    check(
        math.isfinite(args.wait_seconds) and args.wait_seconds >= 0,
        "invalid wait duration",
    )
    sha(args.active_runtime_hash)
    raw, source, deployed_sha = config_identity(args.candidate, args.deployed_config)
    receipt["candidate_sha256"] = digest(raw)
    receipt["deployed_source_sha256"] = deployed_sha
    return source


def read_baseline(args, bench, receipt):
    base_run = bench.command("show", args.baseline_run)
    baseline = base_run["manifest"]
    check(
        base_run["id"] == args.baseline_run
        and base_run["status"] == "completed"
        and baseline["mode"] == "live"
        and baseline.get("execution_cells") is None
        and not baseline.get("recovery")
        and any(t["kind"] == "single" for t in baseline["targets"]),
        "baseline must be a completed full live run containing a single-model target",
    )
    selected = subject(baseline, args.baseline_target)
    if selected["kind"] == "mom":
        sha(selected.get("config_hash"))
    receipt.update(
        baseline_kind=selected["kind"],
        baseline_model=selected["model"],
        baseline_runtime_hash=selected.get("config_hash"),
        baseline_plan_sha256=sha(baseline["plan_sha256"]),
        case_sha256=sha(baseline["case_sha256"]),
        profile=baseline["profile"],
    )
    cases = {case["id"] for case in baseline["cases"]}
    check(
        cases and len(cases) == len(baseline["cases"]),
        "missing or duplicate baseline cases",
    )
    guards = set(args.guard_case) if args.guard_case else cases
    check(guards <= cases, "guard case is absent from baseline")
    receipt["guard_cases"] = sorted(guards)
    baseline_rows = bench.rows(args.baseline_run, args.baseline_target, cases)
    return baseline, baseline_rows


def submit_candidate(args, bench, receipt, directory, source, baseline):
    plan = bench.command(
        "candidate-plan",
        args.baseline_run,
        "--target",
        args.candidate_target,
        "--mode",
        "live",
    )
    manifest = plan["manifest"]
    check(
        plan["status"] == "validated" and plan["model_requests"] == 0,
        "candidate plan was not validated",
    )
    check(
        sha(plan["plan_sha256"]) == manifest["plan_sha256"],
        "reviewed plan hash mismatch",
    )
    receipt["recipe"] = validate_candidate(manifest, args, source, baseline, receipt)
    receipt["candidate_plan_sha256"] = plan["plan_sha256"]
    # Bind retries to this candidate even if a later server plan drifts.
    key = digest(
        encoded(
            [
                args.baseline_run,
                args.candidate_target,
                receipt["candidate_sha256"],
                args.active_runtime_hash,
            ]
        )
    )
    receipt["idempotency_key"] = "recipe-lab-" + key
    receipt["reason"] = (
        "candidate plan reviewed; submission may be retried with the same key"
    )
    persist(directory, receipt)
    receipt["candidate_run"] = bench.submit(manifest, receipt["idempotency_key"])
    check(
        isinstance(receipt["candidate_run"], str) and receipt["candidate_run"],
        "submission returned no run ID",
    )
    receipt["reason"] = "candidate submitted; awaiting evidence"
    persist(directory, receipt)


def await_candidate(args, bench, receipt, source, baseline):
    run_id = receipt["candidate_run"]
    check(run_id != args.baseline_run, "baseline and candidate runs must be distinct")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        run = bench.command("show", run_id)
        check(run["id"] == run_id, "candidate run ID mismatch")
        manifest = run["manifest"]
        receipt["recipe"] = validate_candidate(
            manifest, args, source, baseline, receipt
        )
        plan_hash = sha(manifest["plan_sha256"])
        check(
            receipt.get("candidate_plan_sha256", plan_hash) == plan_hash,
            "submitted plan changed",
        )
        receipt["candidate_plan_sha256"] = plan_hash
        if run["status"] == "completed":
            break
        check(
            run["status"] in {"queued", "running"}, f"candidate run is {run['status']}"
        )
        remaining = deadline - time.monotonic()
        check(remaining > 0, "candidate is still active; reopen its run ID")
        time.sleep(min(1.0, remaining))
    report = bench.command("report", run_id)
    check(
        report["run_id"] == run_id and report["status"] == "completed",
        "candidate report mismatch",
    )
    provenance = report["provenance"]
    check(
        provenance["plan_sha256"] == receipt["candidate_plan_sha256"],
        "report plan hash mismatch",
    )
    runner = provenance.get("runner") or {}
    snapshot = runner.get("recipe_snapshots", {}).get(args.candidate_target) or {}
    check(
        snapshot.get("source_config_hash") == receipt["deployed_source_sha256"],
        "deployed source SHA-256 differs or is missing",
    )
    check(
        snapshot.get("generated_runtime_hash")
        == snapshot.get("active_runtime_hash")
        == args.active_runtime_hash,
        "deployed runtime identity differs or is missing",
    )


def _evaluate(args, bench, receipt, directory):
    source = read_candidate(args, receipt)
    baseline, baseline_rows = read_baseline(args, bench, receipt)
    if not args.candidate_run:
        submit_candidate(args, bench, receipt, directory, source, baseline)
    await_candidate(args, bench, receipt, source, baseline)
    run_id = receipt["candidate_run"]
    comparison = bench.command("compare", args.baseline_run, run_id)
    check(
        comparison["baseline_run_id"] == args.baseline_run
        and comparison["candidate_run_id"] == run_id,
        "sr-bench comparison references different runs",
    )
    receipt["sr_bench_selected_single"] = comparison["baseline_selected_target_id"]
    current = bench.rows(run_id, args.candidate_target, set(baseline_rows))
    result, reason, metrics = verdict(
        baseline_rows, current, receipt["guard_cases"], args.min_speedup_pct
    )
    receipt.update(verdict=result, reason=reason, metrics=metrics)


def evaluate(args):
    receipt = {
        "schema": "recipe-lab-sr-bench-run-v1",
        "verdict": "INCONCLUSIVE",
        "reason": "",
        "proposer": args.proposer,
        "baseline_run": args.baseline_run,
        "baseline_target": args.baseline_target,
        "candidate_run": args.candidate_run,
        "candidate_target": args.candidate_target,
        "active_runtime_hash": args.active_runtime_hash,
        "candidate_sha256": None,
        "min_speedup_pct": (
            args.min_speedup_pct if math.isfinite(args.min_speedup_pct) else None
        ),
        "guard_cases": sorted(set(args.guard_case)),
        "metrics": None,
        "scope": "Descriptive paired case outcomes and timing for this run; statistical significance is untested.",
    }
    directory = args.workspace / "sr-bench"
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _evaluate(args, SrBench(args.vllm_sr), receipt, directory)
    except yaml.YAMLError:
        receipt["reason"] = "invalid configuration YAML"
    except (
        KeyError,
        TypeError,
        AttributeError,
        ValueError,
        OverflowError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        receipt["reason"] = str(exc)
    try:
        save_result(args.workspace, receipt)
    except OSError as exc:
        receipt.update(
            verdict="INCONCLUSIVE",
            reason=f"could not save run receipt: {exc}",
            receipt=None,
        )
    return receipt


# The largest latency reduction a candidate may request, in percent.
MAX_SPEEDUP_PCT = 100


class SrbenchError(RuntimeError):
    """Raised when the sr-bench CLI cannot be executed or parsed."""


# Verified: the sr-bench service marks these run states terminal.
TERMINAL_RUN_STATES = frozenset({"completed", "failed", "cancelled", "interrupted"})


def cases_to_srbench_rows(
    cases: list[dict[str, Any]], benchmark: str = "recipe-lab"
) -> list[dict[str, Any]]:
    """Render recipe-lab cases as sr-bench dataset rows (X3).

    `case_id` becomes `id`, `prompt` becomes a single user message, and a
    string `truth` becomes `answer` (the row shape the real service accepts,
    verified against the R2 integration harness). Deterministic: same cases
    in the same order => byte-identical JSONL once serialized with fixed
    keys. Non-string (signal-fact) truths cannot be judged by sr-bench's
    answer matching and are rejected here, never silently dropped.
    """
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        truth = case.get("truth")
        if not isinstance(truth, str) or not truth:
            raise ContractError(
                f"case {index} ({case.get('case_id')!r}) has no string truth; "
                "an sr-bench dataset needs an answer to grade against"
            )
        rows.append(
            {
                "id": str(case["case_id"]),
                "benchmark": benchmark,
                "messages": [{"role": "user", "content": str(case["prompt"])}],
                "answer": truth,
            }
        )
    return rows


@dataclass
class SrbenchCli:
    bin: str = "vllm-sr"
    url: str | None = None
    store: str | None = None
    timeout_s: float = 3600.0

    def _base_args(self) -> list[str]:
        args = [self.bin, "benchmark"]
        if self.url:
            args += ["--url", self.url]
        if self.store:
            args += ["--store", self.store]
        return args

    def run_command(self, args: list[str]) -> dict[str, Any] | str:
        """Run one benchmark subcommand; parse JSON stdout when possible."""
        cmd = self._base_args() + args
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SrbenchError(
                f"sr-bench CLI failed ({' '.join(args)}): {exc}"
            ) from exc
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-500:]
            raise SrbenchError(
                f"sr-bench CLI returned {completed.returncode} ({' '.join(args)}): {tail}"
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return completed.stdout

    def compare(
        self, baseline_run_id: str, candidate_run_id: str
    ) -> dict[str, Any] | str:
        return self.run_command(["compare", baseline_run_id, candidate_run_id])

    def show(self, run_id: str) -> dict[str, Any] | str:
        """One run document (status, manifest, ...); no evidence paging."""
        return self.run_command(["show", run_id])

    def targets(self) -> list[dict[str, Any]]:
        """The operator-registered targets (`target list` -> GET /targets).

        R2: the registry is the only acceptable source for the auto-built
        single-model baseline arm - a submitted manifest target must match
        the registry field-for-field, so the registry document is embedded
        verbatim.
        """
        payload = self.run_command(["target", "list"])
        rows = payload.get("targets") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise SrbenchError("sr-bench target list did not return targets")
        return [row for row in rows if isinstance(row, dict)]

    def runs(self) -> list[dict[str, Any]]:
        """All run summaries (`runs` -> GET /runs), newest page of the store."""
        payload = self.run_command(["runs"])
        rows = payload.get("runs") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise SrbenchError("sr-bench runs did not return a list")
        return [row for row in rows if isinstance(row, dict)]

    def wait_for_run(
        self, run_id: str, *, poll_interval_s: float = 2.0
    ) -> dict[str, Any]:
        """Poll `show` until the run reaches a terminal state (I).

        The caller owns the loop, so each trial's outcome is observable and
        recordable. Returns the final run document; a non-dict payload, a
        missing status, or a non-terminal status that stops updating is an
        SrbenchError, never a guessed state.
        """
        while True:
            doc = self.show(run_id)
            if not isinstance(doc, dict):
                raise SrbenchError(
                    f"sr-bench show {run_id} did not return a run document"
                )
            status = doc.get("status")
            if status is None:
                raise SrbenchError(f"sr-bench run {run_id} carries no status")
            if status in TERMINAL_RUN_STATES:
                return doc
            time.sleep(poll_interval_s)

    def _submit_and_wait(self, payload: Any, what: str) -> dict[str, Any]:
        run_id = payload.get("id") if isinstance(payload, dict) else None
        if not run_id or not isinstance(run_id, str):
            raise SrbenchError(
                f"sr-bench did not return a run id for {what}: {payload!r}"
            )
        return self.wait_for_run(run_id)

    def run_manifest(self, manifest_path: str | Path) -> dict[str, Any]:
        """Submit one frozen manifest (`run --manifest --detach`) and wait."""
        payload = self.run_command(
            ["run", "--manifest", str(manifest_path), "--detach"]
        )
        return self._submit_and_wait(payload, f"run --manifest {manifest_path}")

    def preview(self, manifest_path: str | Path) -> dict[str, Any]:
        """Submit one preview manifest (routing only, no quality) and wait."""
        payload = self.run_command(
            ["preview", "--manifest", str(manifest_path), "--detach"]
        )
        return self._submit_and_wait(payload, f"preview --manifest {manifest_path}")

    def candidate_plan(
        self,
        baseline_run_id: str,
        *,
        targets: Iterable[str],
        mode: str = "live",
        name: str | None = None,
        hypothesis: str = "",
    ) -> dict[str, Any]:
        """Reuse a terminal baseline's frozen protocol for a new run (I).

        Two service steps, both verified against the service (I,
        cli/sr_bench/service.py): `candidate-plan` only reviews the protocol
        (response status "validated", model_requests 0 - the engine refuses
        to dispatch from the plan route), so the returned frozen manifest is
        then dispatched verbatim through `run --manifest --detach`. The
        engine enforces reviewed_plan_changed, which keeps this honest: any
        divergence between the reviewed plan and the dispatched manifest is
        a service-side 400. The run is waited to a terminal state here.
        """
        target_list = list(targets)
        if not target_list:
            raise SrbenchError("candidate-plan requires at least one target id")
        args = ["candidate-plan", baseline_run_id]
        for tgt in target_list:
            args += ["--target", tgt]
        args += ["--mode", mode]
        if name:
            args += ["--name", name]
        if hypothesis:
            args += ["--hypothesis", hypothesis]
        reviewed = self.run_command(args)
        manifest = reviewed.get("manifest") if isinstance(reviewed, dict) else None
        plan_sha = reviewed.get("plan_sha256") if isinstance(reviewed, dict) else None
        if (
            not isinstance(manifest, dict)
            or not isinstance(plan_sha, str)
            or not plan_sha
        ):
            raise SrbenchError(
                f"sr-bench candidate-plan did not return a reviewed plan for "
                f"{baseline_run_id}: {reviewed!r}"
            )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", encoding="utf-8", delete=False
        ) as handle:
            json.dump(manifest, handle)
            plan_path = Path(handle.name)
        try:
            payload = self.run_command(
                ["run", "--manifest", str(plan_path), "--detach"]
            )
        finally:
            plan_path.unlink(missing_ok=True)
        return self._submit_and_wait(payload, f"candidate-plan {baseline_run_id}")

    def experiment(self, args: list[str]) -> dict[str, Any] | str:
        """Group baselines/routing-checks/candidates (create, attach, show...)."""
        return self.run_command(["experiment", *args])

    def report(self, run_id: str, output_path: str | Path) -> dict[str, Any] | str:
        self.run_command(["report", run_id, "--output", str(output_path)])
        path = Path(output_path)
        if not path.is_file():
            raise SrbenchError(f"sr-bench report did not write {path}")
        text = path.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


def extract_metric(payload: Any, metric_key: str) -> float | None:
    """Depth-first search for `metric_key` with a numeric leaf value."""
    if isinstance(payload, dict):
        if metric_key in payload and isinstance(payload[metric_key], (int, float)):
            return float(payload[metric_key])
        for value in payload.values():
            found = extract_metric(value, metric_key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = extract_metric(item, metric_key)
            if found is not None:
                return found
    return None


def require_metric(payload: Any, metric_key: str) -> float:
    value = extract_metric(payload, metric_key)
    if value is None:
        raise ContractError(f"metric {metric_key!r} not found in sr-bench payload")
    return value


def extract_metric_values(payload: Any, metric_key: str) -> list[float]:
    """All numeric occurrences of `metric_key`, in traversal order."""
    values: list[float] = []
    if isinstance(payload, dict):
        if metric_key in payload and isinstance(payload[metric_key], (int, float)):
            values.append(float(payload[metric_key]))
        for value in payload.values():
            values.extend(extract_metric_values(value, metric_key))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(extract_metric_values(item, metric_key))
    return values
