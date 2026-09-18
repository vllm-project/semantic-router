"""Frozen evaluation plans and adapter discovery."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from . import VERSION

BENCHMARKS = (
    (
        "mmlu-pro",
        "MMLU-Pro",
        "multiple-choice",
        "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
    ),
    (
        "gpqa-diamond",
        "GPQA Diamond",
        "multiple-choice",
        "https://huggingface.co/datasets/Idavidrein/gpqa",
    ),
    (
        "hle",
        "Humanity's Last Exam (text)",
        "judge",
        "https://huggingface.co/datasets/cais/hle",
    ),
    (
        "livecodebench",
        "LiveCodeBench",
        "code",
        "https://github.com/LiveCodeBench/LiveCodeBench",
    ),
    ("scicode", "SciCode", "code", "https://github.com/scicode-bench/SciCode"),
    ("terminal-bench-2.1", "Terminal-Bench 2.1", "agent", "https://www.tbench.ai/"),
    (
        "simpleqa-verified",
        "SimpleQA Verified",
        "judge",
        "https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified",
    ),
    ("arc-agi-2", "ARC-AGI-2", "grid", "https://github.com/arcprize/ARC-AGI-2"),
    ("tau3", "τ³", "agent", "https://github.com/sierra-research/tau2-bench"),
)
BENCHMARK_WEIGHTS = {
    "mmlu-pro": 0.10,
    "simpleqa-verified": 0.10,
    "gpqa-diamond": 0.15,
    "hle": 0.15,
    "arc-agi-2": 0.10,
    "livecodebench": 0.10,
    "scicode": 0.10,
    "terminal-bench-2.1": 0.10,
    "tau3": 0.10,
}

DEFAULT_LIMITS = {
    "concurrency": 1,
    "total_timeout_s": 180,
    "idle_timeout_s": 30,
    "max_output_tokens": 4096,
    "max_output_chars": 131072,
    "repetition_window": 128,
    "repetition_limit": 5,
    "max_cost_usd": 5.0,
    "max_run_seconds": 1800,
    "max_calls_per_case": 32,
    "case_timeout_s": 600,
}


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def catalog():
    return {
        "version": VERSION,
        "benchmarks": [
            dict(id=i, title=t, kind=k, source_url=u) for i, t, k, u in BENCHMARKS
        ],
        "profiles": [
            {"id": "smoke", "purpose": "Integration validation"},
            {"id": "quick", "purpose": "Fixed development cases"},
            {"id": "standard", "purpose": "Frozen holdout evaluation"},
        ],
        "modes": ["live", "preview"],
    }


def load_document(path):
    path = Path(path).expanduser().resolve()
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def _finite_positive(value, name, upper=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    if upper and value > upper:
        raise ValueError(f"{name} must be at most {upper}")


def plan(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    m = copy.deepcopy(manifest)
    if m.get("version") != VERSION:
        raise ValueError(f"version must be {VERSION}")
    if m.get("mode", "live") not in {"live", "preview"}:
        raise ValueError("mode must be live or preview; replay is not live evidence")
    m.setdefault("mode", "live")
    m.setdefault("profile", "quick")
    m.setdefault("seed", 20260918)
    m.setdefault("name", "sr-bench")
    m.setdefault("cost_policy", "require_priced")
    if m["cost_policy"] not in {"require_priced", "capability_only"}:
        raise ValueError("cost_policy must be require_priced or capability_only")
    if m["profile"] not in {"smoke", "quick", "standard"}:
        raise ValueError("unknown profile")
    if "dataset" in m:
        ds = m["dataset"]
        if not isinstance(ds, dict) or not ds.get("path") or not ds.get("sha256"):
            raise ValueError("dataset requires path and sha256")
        path = Path(ds["path"]).expanduser().resolve()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != ds["sha256"]:
            raise ValueError("dataset SHA256 mismatch")
        document = load_document(path)
        loaded = document if isinstance(document, list) else document.get("cases")
        if "cases" in m and m["cases"] != loaded:
            raise ValueError("inline cases do not match dataset")
        m["cases"] = loaded
        metadata_path = path.parent / "manifest.json"
        if metadata_path.is_file() and metadata_path != path:
            metadata = load_document(metadata_path)
            if metadata.get("sha256") == actual:
                if metadata.get("profile") != m["profile"]:
                    raise ValueError(
                        "Prepared dataset profile differs from requested run profile"
                    )
                if metadata.get("seed") != m["seed"]:
                    raise ValueError(
                        "Prepared dataset seed differs from requested run seed"
                    )
                m["dataset"] = {**metadata, **ds}

    cases = m.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("at least one case is required")
    ids = set()
    available = {x[0] for x in BENCHMARKS}
    for c in cases:
        if (
            not isinstance(c, dict)
            or not isinstance(c.get("id"), str)
            or not c["id"]
            or c["id"] in ids
        ):
            raise ValueError("case IDs must be unique nonempty strings")
        ids.add(c["id"])
        if (
            m["profile"] == "standard"
            and c.get("metadata", {}).get("split") != "holdout"
        ):
            raise ValueError("Standard profile requires prepared holdout cases")
        if c.get("benchmark") not in available:
            raise ValueError(f"unknown benchmark for case {c['id']}")
        if not c.get("messages") and c["benchmark"] not in {
            "tau3",
            "terminal-bench-2.1",
            "scicode",
        }:
            raise ValueError(f"messages are required for case {c['id']}")
        for msg in c.get("messages", []):
            if not isinstance(msg, dict) or msg.get("role") not in {
                "system",
                "user",
                "assistant",
                "tool",
                "developer",
            }:
                raise ValueError("invalid case message")
        if (
            m["mode"] == "live"
            and c["benchmark"] in {"mmlu-pro", "gpqa-diamond", "arc-agi-2"}
            and "answer" not in c
        ):
            raise ValueError(f"answer is required for {c['id']}")
    targets = m.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("at least one target is required")
    auxiliary = m.get("auxiliary_targets", {})
    if not isinstance(auxiliary, dict) or any(
        not isinstance(t, dict) or t.get("id") != key for key, t in auxiliary.items()
    ):
        raise ValueError(
            "auxiliary_targets must map target IDs to matching target objects"
        )
    ids = set()
    for t in targets + list(auxiliary.values()):
        if (
            not isinstance(t, dict)
            or not isinstance(t.get("id"), str)
            or not t["id"]
            or t["id"] in ids
        ):
            raise ValueError("target IDs must be unique nonempty strings")
        ids.add(t["id"])
        if t.get("kind") not in {"single", "mom"} or not t.get("model"):
            raise ValueError("target requires kind single/mom and model")
        u = urlparse(t.get("base_url", ""))
        if (
            u.scheme not in {"http", "https"}
            or not u.hostname
            or u.username
            or u.password
            or u.query
            or u.fragment
        ):
            raise ValueError(
                "target base_url must be an HTTP(S) URL without credentials/query"
            )
        if any(k in t for k in ("api_key", "authorization", "headers")):
            raise ValueError(
                "use api_key_env; raw credentials and arbitrary headers are forbidden"
            )
        if m["mode"] == "preview" and t in targets:
            if t["kind"] != "mom":
                raise ValueError("preview only supports mom targets")
            preview = urlparse(t.get("preview_url", ""))
            if (
                preview.scheme not in {"http", "https"}
                or not preview.hostname
                or preview.username
                or preview.password
                or preview.query
                or preview.fragment
            ):
                raise ValueError(
                    "preview requires an explicit management preview_url (/api/v1/routing/preview)"
                )
        if t["kind"] == "mom" and m["mode"] == "live" and not t.get("config_hash"):
            raise ValueError("live mom target requires frozen config_hash")
        if m["mode"] == "live" and m["cost_policy"] == "require_priced":
            if not t.get("prices") or (
                t["kind"] == "single" and t["model"] not in t["prices"]
            ):
                raise ValueError(
                    "require_priced needs known prices for every target; use explicit capability_only to omit cost claims"
                )
            if t["kind"] == "mom" and (
                not isinstance(t.get("max_inference_calls"), int)
                or not 1 <= t["max_inference_calls"] <= 256
            ):
                raise ValueError(
                    "Priced MoM requires a frozen max_inference_calls bound"
                )
        for price in t.get("prices", {}).values():
            if set(price) != {"input", "cached_input", "cache_write", "output"}:
                raise ValueError("prices must include all four token buckets")
            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0
                for v in price.values()
            ):
                raise ValueError(
                    "prices must be finite nonnegative USD per million tokens"
                )
    limits = {**DEFAULT_LIMITS, **m.get("limits", {})}
    unknown = set(limits) - set(DEFAULT_LIMITS)
    if unknown:
        raise ValueError(f"unknown limits: {sorted(unknown)}")
    for k, v in limits.items():
        _finite_positive(v, k)
    for k in (
        "concurrency",
        "max_output_tokens",
        "max_output_chars",
        "repetition_window",
        "repetition_limit",
        "max_calls_per_case",
    ):
        if not isinstance(limits[k], int):
            raise ValueError(f"{k} must be an integer")
    if (
        limits["concurrency"] > 32
        or limits["total_timeout_s"] > limits["max_run_seconds"]
        or limits["idle_timeout_s"] > limits["total_timeout_s"]
    ):
        raise ValueError("invalid concurrency or timeout hierarchy")
    m["limits"] = limits
    sampling = {
        "temperature": 0,
        "max_tokens": limits["max_output_tokens"],
        **m.get("sampling", {}),
    }
    if (
        sampling.get("max_tokens", 0) > limits["max_output_tokens"]
        or sampling.get("max_tokens", 0) <= 0
    ):
        raise ValueError(
            "sampling.max_tokens must be positive and within output token cap"
        )
    forbidden = {"model", "messages", "stream", "tools", "api_key"} & set(sampling)
    if forbidden:
        raise ValueError(f"sampling cannot override {sorted(forbidden)}")
    m["sampling"] = sampling
    m["case_sha256"] = digest(cases)
    weights = m.get("benchmark_weights", BENCHMARK_WEIGHTS)
    if weights != BENCHMARK_WEIGHTS:
        raise ValueError(
            "sr-bench 1.0 benchmark weights are fixed; use per-benchmark results for custom analysis"
        )
    m["benchmark_weights"] = BENCHMARK_WEIGHTS

    if m["mode"] == "live" and any(
        c["benchmark"] not in {"mmlu-pro", "gpqa-diamond", "arc-agi-2"} for c in cases
    ):
        from .external import preflight_case

        checked = set()
        for case in cases:
            b = case["benchmark"]
            if b not in {"mmlu-pro", "gpqa-diamond", "arc-agi-2"} and (
                b not in checked or b in {"hle", "simpleqa-verified"}
            ):
                preflight_case(case, m)
                checked.add(b)
    m["plan_sha256"] = digest({k: v for k, v in m.items() if k != "plan_sha256"})
    return m
