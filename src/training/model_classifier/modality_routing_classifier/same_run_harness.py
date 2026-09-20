#!/usr/bin/env python3
"""Single-stream same-run harness for router-native candidates (#3856).

QSL → warmup (discarded) → one-in-flight classify → one JSON record per request.

Quality numbers are not defined here. This file emits per-row gold/pred labels
for the #3194 metric contract. Latency, peak RSS, and CPU seconds are the
harness outputs. Pair two runs from the same host with same_run_pair.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypedDict

LABELS = ("AR", "DIFFUSION", "BOTH")
HOST_IDENTITY_KEYS = ("cpu_model", "core_count", "ram_gb")
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
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def host_fingerprint(binding: str) -> dict:
    torch_version = None
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        torch_version = _package_version("torch")
    return {
        "cpu_model": _cpu_model(),
        "core_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "candle_version": os.environ.get("CANDLE_BINDING_VERSION"),
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
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
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
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def load_candle_adapter(model_id: str, max_length: int):
    """Production path: persistent Go helper wrapping ClassifyMmBert32KModality.

    The helper reads one JSON line per request from stdin and writes one JSON
    line per response to stdout:
      request:  {"text": "...", "max_length": 256}
      response: {"label": "AR", "seq_len": 128, "tokenize_ns": 0,
                 "forward_ns": 5000000, "helper_cpu_s": 0.482,
                 "helper_rss_mb": 412.3}

    helper_cpu_s and helper_rss_mb are the helper's OWN cumulative
    getrusage(RUSAGE_SELF) readings (CPU seconds and lifetime-peak RSS),
    reported on every response. Inference for this binding runs in that
    child process, not in this Python process, so run_single_stream folds
    these into cpu_s/peak_rss_mb — otherwise those fields would only reflect
    pipe I/O overhead in the parent and silently miss the actual model cost.

    Build the helper:
      cd candle-binding && go build -o candle-classify ./cmd/classify-helper/
    Then either:
      export CANDLE_CLASSIFY_HELPER=/path/to/candle-classify
      # or put candle-classify on PATH
    """
    import json as _json
    import shutil
    import subprocess

    helper = os.environ.get("CANDLE_CLASSIFY_HELPER") or shutil.which("candle-classify")
    if not helper:
        raise SystemExit(
            "--binding candle requires a compiled Go helper binary.\n"
            "Build:  cd candle-binding && go build -o candle-classify ./cmd/classify-helper/\n"
            "Export: CANDLE_CLASSIFY_HELPER=/path/to/candle-classify"
        )

    import atexit

    try:
        proc = subprocess.Popen(
            [helper, "--model", model_id, "--max-length", str(max_length)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"candle helper binary not found or not executable: {helper!r}"
        ) from exc

    if proc.poll() is not None:
        err = proc.stderr.read()
        raise SystemExit(
            f"candle helper exited on startup (model={model_id!r}): {err.strip()}"
        )

    def _terminate_helper() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream and not stream.closed:
                    stream.close()
            except Exception:
                pass

    atexit.register(_terminate_helper)

    def classify(text: str) -> ClassifyResult:  # type: ignore[return]
        request = _json.dumps({"text": text, "max_length": max_length}) + "\n"
        t_submit = time.perf_counter_ns()
        try:
            proc.stdin.write(request)
            proc.stdin.flush()
            line = proc.stdout.readline()
        except BrokenPipeError as exc:
            err = proc.stderr.read()
            raise RuntimeError(f"candle helper pipe broken: {err.strip()}") from exc
        t_return = time.perf_counter_ns()
        elapsed = t_return - t_submit
        if not line:
            err = proc.stderr.read()
            raise RuntimeError(f"candle helper closed stdout: {err.strip()}")
        data = _json.loads(line)
        if "error" in data:
            raise RuntimeError(f"candle helper returned error: {data['error']}")
        classify.helper_stats["cpu_s"] = float(data.get("helper_cpu_s", 0.0))
        classify.helper_stats["peak_rss_mb"] = float(data.get("helper_rss_mb", 0.0))
        return {
            "output": str(data["label"]),
            "tokenize_ns": int(data.get("tokenize_ns", 0)),
            "forward_ns": int(data.get("forward_ns", elapsed)),
            "e2e_ns": elapsed,
            "seq_len": int(data.get("seq_len", 0)),
        }

    classify.close = _terminate_helper  # type: ignore[attr-defined]
    # Cumulative getrusage(RUSAGE_SELF) readings from the helper process,
    # updated after every response. run_single_stream reads this to fold
    # the helper's real CPU/RSS cost into the reported run metrics.
    classify.helper_stats = {"cpu_s": 0.0, "peak_rss_mb": 0.0}  # type: ignore[attr-defined]
    return classify


def load_adapter(binding: str, model_id: str, max_length: int):
    if binding == "hf":
        return load_hf_adapter(model_id, max_length)
    if binding == "candle":
        return load_candle_adapter(model_id, max_length)
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
        default="llm-semantic-router/mmbert32k-modality-router-merged",
    )
    parser.add_argument(
        "--binding",
        default="hf",
        choices=["hf", "candle"],
        help="hf = HuggingFace transformers; candle = Go helper wrapping ClassifyMmBert32KModality",
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
