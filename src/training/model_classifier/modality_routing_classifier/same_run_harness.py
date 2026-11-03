#!/usr/bin/env python3
"""Single-stream same-run harness for router-native candidates (#3856).

QSL → warmup (discarded) → one-in-flight classify → one JSON record per request.

Quality numbers are not defined here. This file emits per-row gold/pred labels
for the #3194 metric contract. Latency, peak RSS, and CPU seconds are the
harness outputs. Pair two runs from the same host with same_run_pair.py.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import http.client
import json
import os
import platform
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version as _package_version_lookup
from pathlib import Path
from typing import TypedDict
from urllib import error as urllib_error
from urllib import request as urllib_request

LABELS = ("AR", "DIFFUSION", "BOTH")
HOST_IDENTITY_KEYS = ("cpu_model", "core_count", "ram_gb")
HTTP_OK = 200
METRIC_CONTRACT = "https://github.com/vllm-project/semantic-router/issues/3194"


class ClassifyResult(TypedDict):
    output: str
    tokenize_ns: int
    forward_ns: int
    e2e_ns: int
    seq_len: int


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_id_for(text: str) -> str:
    """Stable under gold relabelling: hash the prompt text only."""
    return sha256_hex(text)[:16]


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 if sys.platform != "darwin" else 1024 * 1024
    return float(usage) / divisor


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _ram_gb() -> float:
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = float(line.split()[1])
                return round(kb / (1024 * 1024), 1)
    return 0.0


def _package_version(name: str) -> str | None:
    try:
        return _package_version_lookup(name)
    except Exception:
        return None


def host_fingerprint(binding: str) -> dict:
    torch_version = None
    try:
        import torch  # noqa: PLC0415 — optional heavy dep, only probed here

        torch_version = torch.__version__
    except Exception:
        torch_version = _package_version("torch")
    return {
        "cpu_model": _cpu_model(),
        "core_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "model_runtime_version": _package_version("vllm_srun"),
        "binding": binding,
        "platform": platform.platform(),
    }


def host_identity(host: dict | None) -> tuple:
    if not host:
        raise SystemExit("run is missing host fingerprint; refusing to pair")
    missing = [key for key in HOST_IDENTITY_KEYS if host.get(key) in (None, "")]
    if missing:
        raise SystemExit(f"host fingerprint missing {missing}; refusing to pair")
    return tuple(host[key] for key in HOST_IDENTITY_KEYS)


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def load_qsl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for index, raw_line in enumerate(handle):
            stripped = raw_line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            text = raw["text"]
            rows.append(
                {
                    "qsl_index": index,
                    "row_id": row_id_for(text),
                    "input_hash": sha256_hex(text),
                    "label": raw["label_name"],
                    "text": text,
                }
            )
    return rows


def load_hf_adapter(model_id: str, max_length: int):
    # Lazy: --binding model-runtime must not require torch/transformers installed.
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415  # fmt: skip
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    def classify(text: str) -> ClassifyResult:
        t_submit = time.perf_counter_ns()
        enc = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors="pt",
        )
        t_tok = time.perf_counter_ns()
        with torch.no_grad():
            logits = model(**enc).logits
        t_fwd = time.perf_counter_ns()
        class_id = int(logits.argmax(dim=-1).item())
        t_return = time.perf_counter_ns()
        output = id2label.get(
            class_id, LABELS[class_id] if class_id < len(LABELS) else str(class_id)
        )
        seq_len = int(enc["input_ids"].shape[-1])
        return {
            "output": output,
            "tokenize_ns": t_tok - t_submit,
            "forward_ns": t_fwd - t_tok,
            "e2e_ns": t_return - t_submit,
            "seq_len": seq_len,
        }

    return classify


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# /proc is Linux-only: absent on macOS entirely, and on Windows os.sysconf
# doesn't exist at all (AttributeError, not one of the exceptions caught
# below). Checked once at import time so _proc_cpu_and_rss can short-circuit
# before ever reaching a platform call that would misbehave differently per
# OS, and so load_model_runtime_adapter can decide once whether to attach
# helper_stats at all rather than attach it with fabricated zeros.
_PROC_FS_AVAILABLE = Path("/proc").is_dir()


def _proc_cpu_and_rss(pid: int) -> tuple[float, float]:
    """Read a subprocess's own cumulative CPU seconds and lifetime-peak RSS.

    model-runtime's /metrics has no process-level CPU/RSS (only app counters
    like request duration and queue depth — confirmed no ProcessCollector is
    attached), so unlike the old candle helper, which self-reported
    getrusage(RUSAGE_SELF) inside every JSON response, we read the same
    information externally from the OS's own per-process accounting.
    Returns (0.0, 0.0) if the process has already exited rather than raising,
    so run_single_stream's fold logic always gets numbers to work with. Only
    ever called when _PROC_FS_AVAILABLE is true -- see load_model_runtime_adapter.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # comm (field 2) can itself contain spaces or parens, so split on the
        # last ')' before indexing fields positionally -- a naive .split()
        # would silently misalign every field after it.
        after_comm = stat.rsplit(")", 1)[1].split()
        utime_ticks = int(after_comm[11])  # field 14 overall
        stime_ticks = int(after_comm[12])  # field 15 overall
        clk_tck = os.sysconf("SC_CLK_TCK")
        cpu_s = (utime_ticks + stime_ticks) / clk_tck

        peak_rss_mb = 0.0
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                peak_rss_mb = float(line.split()[1]) / 1024.0  # kB -> MB
                break
        return cpu_s, peak_rss_mb
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return 0.0, 0.0


def _wait_model_runtime_ready(base_url: str, proc: subprocess.Popen) -> None:
    """Poll GET /health until 200, mirroring e2e/testing/vllm-sr-cli's
    ServeProcess.wait_ready: exponential backoff 0.25s -> 2.0s, 180s total
    deadline, and check proc.poll() every iteration to fail fast with the
    captured log instead of spinning for 180s against an already-dead process.
    """
    deadline = time.monotonic() + 180.0
    interval = 0.25
    last_status = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"model-runtime exited with {proc.returncode} before becoming ready; "
                f"check its log output above"
            )
        try:
            request = urllib_request.Request(base_url + "/health")
            with urllib_request.urlopen(request, timeout=5) as response:
                if response.status == HTTP_OK:
                    return
                last_status = response.status
        except (urllib_error.URLError, ConnectionError, TimeoutError) as exc:
            last_status = str(exc)
        time.sleep(interval)
        interval = min(interval * 2, 2.0)
    raise SystemExit(f"model-runtime not ready within 180s: {last_status}")


def load_model_runtime_adapter(model_id: str, max_length: int):
    """Production path: spawn `vllm-srun serve` and classify over HTTP.

    candle-binding (and the Go helper that wrapped it) was deleted upstream
    in #4512; every native model now runs behind this standalone HTTP service
    instead of in-process. classify.helper_stats is refreshed from
    /proc/<pid> after every request so run_single_stream's existing
    snapshot/fold logic (unchanged since the round-2 resource-accounting fix)
    keeps working without modification -- only how the numbers get into
    helper_stats changes.

    Install:
      pip install ./src/model-runtime
    """
    srun = shutil.which("vllm-srun")
    if not srun:
        raise SystemExit(
            "--binding model-runtime requires the vllm-srun package.\n"
            "Install: pip install ./src/model-runtime"
        )

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    # Do NOT force HF_HUB_OFFLINE=1 here: the production path must be allowed
    # to download the model on first run (unlike the test fixture, which has
    # no network access at all and should fail fast instead of hanging).
    proc = subprocess.Popen(
        [
            srun,
            "serve",
            model_id,
            "--device",
            "cpu",
            "--port",
            str(port),
            # The fixed QSL repeats every row across warmup and every later
            # measurement pass, which is a guaranteed cache hit on the
            # content-hash-keyed result cache model-runtime enables by
            # default (16,384 entries) -- measuring cached answers instead
            # of real forward passes after the first pass. 0 genuinely
            # disables it (ResultCache.put() no-ops, and the lookup path
            # skips hashing entirely), not a weak sentinel.
            "--result-cache-entries",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _terminate() -> None:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # as a reader would with Ctrl-C
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    atexit.register(_terminate)
    _wait_model_runtime_ready(base_url, proc)

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)

    def classify(text: str) -> ClassifyResult:  # type: ignore[return]
        nonlocal conn
        body = json.dumps(
            {
                "model": model_id,
                "input": text,
                # Matches the HF adapter's truncation=True, max_length=max_length:
                # the default overflow policy is "reject", which would raise on
                # any row longer than max_tokens instead of truncating it.
                "options": {"max_tokens": max_length, "overflow": "truncate"},
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        t_submit = time.perf_counter_ns()
        try:
            conn.request("POST", "/v1/classify", body, headers)
            response = conn.getresponse()
            payload = response.read()
        except (http.client.HTTPException, OSError):
            # Retry once on a fresh connection -- a single transient reset
            # should not fail the whole run.
            conn.close()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            conn.request("POST", "/v1/classify", body, headers)
            response = conn.getresponse()
            payload = response.read()
        t_return = time.perf_counter_ns()
        elapsed = t_return - t_submit
        if response.status != HTTP_OK:
            raise RuntimeError(
                f"model-runtime classify failed ({response.status}): {payload[:500]!r}"
            )
        data = json.loads(payload)
        result = data["results"][0]
        if "error" in result:
            raise RuntimeError(f"model-runtime returned error: {result['error']}")
        if _PROC_FS_AVAILABLE:
            cpu_s, peak_rss_mb = _proc_cpu_and_rss(proc.pid)
            classify.helper_stats["cpu_s"] = cpu_s
            classify.helper_stats["peak_rss_mb"] = peak_rss_mb
        return {
            "output": str(result["label"]),
            # /v1/classify gives no tokenize/forward split, same honest
            # simplification the old candle helper already used.
            "tokenize_ns": 0,
            "forward_ns": elapsed,
            "e2e_ns": elapsed,
            "seq_len": int(data.get("usage", {}).get("input_tokens", 0)),
        }

    classify.close = _terminate  # type: ignore[attr-defined]
    classify.base_url = base_url  # type: ignore[attr-defined]
    if _PROC_FS_AVAILABLE:
        # Seed from a real reading taken now (server already confirmed ready
        # by _wait_model_runtime_ready above, no classify() call has run
        # yet), not hardcoded zeros. With --warmup 0, run_single_stream
        # snapshots this as helper_cpu_before immediately -- a stale 0.0
        # there would make the model's own load-time CPU (which the ready
        # child has already spent) leak into the first *measured* request's
        # delta, exactly the gap a real fixture run reproduced: 6.48s of
        # child CPU already spent before measurement, 6.511s reported on
        # the very first measured request. run_single_stream reads this to
        # fold the subprocess's real CPU/RSS cost into the reported run
        # metrics.
        initial_cpu_s, initial_peak_rss_mb = _proc_cpu_and_rss(proc.pid)
        classify.helper_stats = {  # type: ignore[attr-defined]
            "cpu_s": initial_cpu_s,
            "peak_rss_mb": initial_peak_rss_mb,
        }
    # Else: leave helper_stats unset. run_single_stream already treats a
    # missing helper_stats attribute as "no subprocess resource folding" --
    # the same path --binding hf takes, since it has no child process either.
    # On macOS/Windows we cannot read the model-runtime subprocess's own
    # CPU/RSS (no /proc), so attaching a dict of fabricated 0.0s would look
    # like a real measurement instead of an honest "not measured here"
    # signal -- the exact silent-wrong-value failure mode already found and
    # fixed for this harness once before (the round-2 resource-accounting
    # bug). cpu_s/peak_rss_mb in the report still reflect this process's own
    # cost; they just won't include the subprocess's.
    return classify


def load_adapter(binding: str, model_id: str, max_length: int):
    if binding == "hf":
        return load_hf_adapter(model_id, max_length)
    if binding == "model-runtime":
        return load_model_runtime_adapter(model_id, max_length)
    raise SystemExit(f"unknown binding: {binding}")


def ns_to_ms(ns: int) -> float:
    return ns / 1e6


def summarize(latencies_ms: list[float]) -> dict:
    return {
        "n": len(latencies_ms),
        "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
        "p50_ms": round(percentile(latencies_ms, 50), 3),
        "p90_ms": round(percentile(latencies_ms, 90), 3),
        "p95_ms": round(percentile(latencies_ms, 95), 3),
        "p99_ms": round(percentile(latencies_ms, 99), 3),
        "max_ms": round(max(latencies_ms), 3),
        "min_ms": round(min(latencies_ms), 3),
    }


def run_single_stream(
    classify: Callable[[str], ClassifyResult],
    qsl: list[dict],
    warmup_n: int,
    min_duration_s: float,
) -> tuple[list[dict], list[dict], dict]:
    """Run single-stream scoring and return (quality_records, latency_samples, meta).

    quality_records — one entry per QSL row from the first pass only.
      Used for label/output pairing in same_run_pair.py.
    latency_samples — one entry per scored request across *all* passes.
      Used for latency statistics (p50/p99/…) so that --min-duration-s
      produces representative tail-latency numbers rather than first-pass only.
    """
    warmup_n = min(warmup_n, len(qsl))
    for row in qsl[:warmup_n]:
        classify(row["text"])

    # If classify() delegates inference to a child process (e.g. the candle
    # helper), snapshot its cumulative CPU seconds now so the measured window
    # below excludes model-load and warmup cost, matching cpu_before below.
    helper_stats = getattr(classify, "helper_stats", None)
    helper_cpu_before = helper_stats["cpu_s"] if helper_stats is not None else 0.0

    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    quality_records: list[dict] = []
    latency_samples: list[dict] = []
    scored = 0
    while True:
        for row in qsl:
            result = classify(row["text"])
            e2e_ms = round(ns_to_ms(result["e2e_ns"]), 3)
            fwd_ms = round(ns_to_ms(result["forward_ns"]), 3)
            tok_ms = round(ns_to_ms(result["tokenize_ns"]), 3)
            latency_samples.append(
                {
                    "e2e_ms": e2e_ms,
                    "forward_ms": fwd_ms,
                    "tokenize_ms": tok_ms,
                }
            )
            if scored < len(qsl):
                quality_records.append(
                    {
                        "row_id": row["row_id"],
                        "qsl_index": row["qsl_index"],
                        "input_hash": row["input_hash"],
                        "label": row["label"],
                        "output": result["output"],
                        "seq_len": result["seq_len"],
                        "tokenize_ms": tok_ms,
                        "forward_ms": fwd_ms,
                        "e2e_ms": e2e_ms,
                    }
                )
            scored += 1
        elapsed = time.perf_counter() - wall_before
        if elapsed >= min_duration_s and scored >= len(qsl):
            break

    cpu_s = time.process_time() - cpu_before
    wall_s = time.perf_counter() - wall_before
    combined_peak_rss_mb = peak_rss_mb()
    includes_helper_resources = False

    if helper_stats is not None:
        # Fold in the helper's measured-window CPU delta (mirrors cpu_before
        # above) and its lifetime-peak RSS (mirrors peak_rss_mb() above,
        # which is itself a lifetime high-water mark for this process — the
        # two are directly comparable and additive since both processes are
        # concurrently resident for the run's duration).
        cpu_s += helper_stats["cpu_s"] - helper_cpu_before
        combined_peak_rss_mb += helper_stats["peak_rss_mb"]
        includes_helper_resources = True

    meta = {
        "warmup_n": warmup_n,
        "scored_queries": scored,
        "n_records": len(quality_records),
        "n_latency_samples": len(latency_samples),
        "cpu_s": round(cpu_s, 3),
        "wall_s": round(wall_s, 3),
        "peak_rss_mb": round(combined_peak_rss_mb, 1),
        "includes_helper_resources": includes_helper_resources,
        "min_duration_s": min_duration_s,
    }
    return quality_records, latency_samples, meta


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Same-run single-stream harness (#3856)"
    )
    parser.add_argument(
        "--qsl",
        type=Path,
        default=here / "exported_modality_routing_dataset" / "test.jsonl",
    )
    parser.add_argument(
        "--model",
        default="vllm-sr/Vela-1.0-Encoder-307M-Modality",
    )
    parser.add_argument(
        "--binding",
        default="hf",
        choices=["hf", "model-runtime"],
        help="hf = HuggingFace transformers; model-runtime = vllm-srun serve over HTTP",
    )
    parser.add_argument("--role", default="baseline", choices=["baseline", "candidate"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-duration-s", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "same_run_bert_singlestream.json",
    )
    args = parser.parse_args()

    qsl = load_qsl(args.qsl)
    if not qsl:
        raise SystemExit(f"empty QSL: {args.qsl}")

    classify = load_adapter(args.binding, args.model, args.max_length)
    quality_records, latency_samples, meta = run_single_stream(
        classify, qsl, args.warmup, args.min_duration_s
    )

    e2e = [s["e2e_ms"] for s in latency_samples]
    forward = [s["forward_ms"] for s in latency_samples]
    tokenize = [s["tokenize_ms"] for s in latency_samples]
    report = {
        "issue": "#3856",
        "scenario": "single-stream",
        "role": args.role,
        "model": args.model,
        "qsl": str(args.qsl),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host_fingerprint(args.binding),
        "quality": {
            "metric_contract": METRIC_CONTRACT,
            "emits": ["records[].label", "records[].output"],
            "note": (
                "Pooled accuracy is not defined in this harness. "
                "Score per-class metrics and thresholds with the #3194 contract "
                "on a separate split."
            ),
        },
        "run": {
            "binding": args.binding,
            "batch_size": 1,
            "max_length": args.max_length,
            "qsl_rows": len(qsl),
            **meta,
            "e2e": summarize(e2e),
            "forward": summarize(forward),
            "tokenize": summarize(tokenize),
        },
        "records": quality_records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        "output": str(args.output),
        "host": report["host"],
        **report["run"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
