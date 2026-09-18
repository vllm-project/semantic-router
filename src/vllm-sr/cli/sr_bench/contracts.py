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
    "max_log_bytes": 8388608,
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
    from .adapters import list_adapters

    return {
        "version": VERSION,
        "benchmarks": [adapter.catalog_entry() for adapter in list_adapters()],
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
            json.loads(line) for line in path.read_text().split("\n") if line.strip()
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


GENERATION_FIELDS = {
    "n",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "length_penalty",
    "stop",
    "seed",
    "max_tokens",
    "min_tokens",
    "reasoning_effort",
    "chat_template_kwargs",
    "ignore_eos",
    "skip_special_tokens",
    "spaces_between_special_tokens",
}


def validate_request_params(params, limits, label="request_params"):
    if not isinstance(params, dict) or set(params) - GENERATION_FIELDS:
        raise ValueError(f"{label} contains unsupported generation fields")
    try:
        serialized = json.dumps(params, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc
    if len(serialized.encode()) > 65536:
        raise ValueError(f"{label} exceeds the frozen parameter size limit")
    if "n" in params and (isinstance(params["n"], bool) or params["n"] != 1):
        raise ValueError(
            f"{label}.n must be exactly 1; generation multiplicity is fixed"
        )
    for name in ("max_tokens", "min_tokens", "top_k", "seed", "n"):
        if name in params and (
            isinstance(params[name], bool) or not isinstance(params[name], int)
        ):
            raise ValueError(f"{label}.{name} must be an integer")
    if (
        "max_tokens" in params
        and not 1 <= params["max_tokens"] <= limits["max_output_tokens"]
    ):
        raise ValueError(f"{label}.max_tokens exceeds the output token cap")
    if "min_tokens" in params and not 0 <= params["min_tokens"] <= params.get(
        "max_tokens", limits["max_output_tokens"]
    ):
        raise ValueError(f"{label}.min_tokens exceeds max_tokens")
    if "top_k" in params and params["top_k"] != -1 and params["top_k"] < 1:
        raise ValueError(f"{label}.top_k must be -1 or positive")
    for name, low, high in (
        ("temperature", 0, 2),
        ("top_p", 0, 1),
        ("min_p", 0, 1),
        ("presence_penalty", -2, 2),
        ("frequency_penalty", -2, 2),
    ):
        if name in params and (
            isinstance(params[name], bool)
            or not isinstance(params[name], (int, float))
            or not low <= params[name] <= high
        ):
            raise ValueError(f"{label}.{name} is outside its valid range")
    for name in ("repetition_penalty", "length_penalty"):
        if name in params:
            _finite_positive(params[name], f"{label}.{name}")
    if "reasoning_effort" in params and params["reasoning_effort"] not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ValueError(f"{label}.reasoning_effort is unsupported")
    if "chat_template_kwargs" in params and not isinstance(
        params["chat_template_kwargs"], dict
    ):
        raise ValueError(f"{label}.chat_template_kwargs must be an object")
    if "stop" in params and not (
        isinstance(params["stop"], str)
        or isinstance(params["stop"], list)
        and all(isinstance(item, str) for item in params["stop"])
    ):
        raise ValueError(f"{label}.stop must be text or a list of text")
    for name in ("ignore_eos", "skip_special_tokens", "spaces_between_special_tokens"):
        if name in params and not isinstance(params[name], bool):
            raise ValueError(f"{label}.{name} must be boolean")


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
    from .adapters import get_adapter, list_adapters

    available = {adapter.id for adapter in list_adapters()}
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
        adapter = get_adapter(c["benchmark"])
        if not c.get("messages") and adapter.requires_messages:
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
        if m["mode"] == "live" and adapter.requires_answer and "answer" not in c:
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
        "max_log_bytes",
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
    if not isinstance(m.get("sampling", {}), dict):
        raise ValueError("sampling must be an object")
    sampling = {
        "temperature": 0,
        "max_tokens": limits["max_output_tokens"],
        **m.get("sampling", {}),
    }
    validate_request_params(sampling, limits, "sampling")
    for target in targets + list(auxiliary.values()):
        params = target.get("request_params", {})
        validate_request_params(params, limits)
        validate_request_params(
            {**sampling, **params}, limits, "effective request_params"
        )
    m["sampling"] = sampling
    m["case_sha256"] = digest(cases)
    expected_weights = {
        **BENCHMARK_WEIGHTS,
        **{c["benchmark"]: get_adapter(c["benchmark"]).weight for c in cases},
    }
    weights = m.get("benchmark_weights", expected_weights)
    if weights != expected_weights:
        raise ValueError(
            "Benchmark weights differ from the registered versioned adapters"
        )
    m["benchmark_weights"] = expected_weights
    versions = {c["benchmark"]: get_adapter(c["benchmark"]).version for c in cases}
    if m.get("adapter_versions", versions) != versions:
        raise ValueError("Frozen adapter versions differ from installed adapters")
    m["adapter_versions"] = versions
    if m["mode"] == "live":
        cache = {}
        for case in cases:
            adapter = get_adapter(case["benchmark"])
            if adapter.preflight is not None:
                adapter.preflight(case, m, cache)
    m["plan_sha256"] = digest({k: v for k, v in m.items() if k != "plan_sha256"})
    return m
