from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from cli.routing_preview import case_request_fields

from .adapters import AdapterContext, BenchmarkAdapter
from .canonical import canonical
from .contracts import DEFAULT_LIMITS, SESSION_AWARE, validate_request_params
from .grading import CACHE_OBSERVATION_RESULT
from .sources import _write_dataset
from .task_identity import native_task_identity, source_identity
from .transport import CACHE_REPORTING_STATES, CallFailure

BENCHMARK_ID: str = "cache-token-probe-v1"
PROBE_VERSION: str = "cache-token-probe-native-v1"
MAX_REPEATS: int = DEFAULT_LIMITS["max_calls_per_case"]
MAX_TARGETS: int = 2
SOURCE_REVISION: str = "a6d23eaff862a9d24dc10ccb778ad0618a839ef7"
SOURCE_SHA256: str = "796dc62f51d28f90665d79fb43f6d3a2f02d08f9a9007d7a564d6a65b1da8e7d"
SOURCE_URL: str = (
    "https://github.com/vllm-project/semantic-router/blob/"
    + SOURCE_REVISION
    + "/bench/cache_token_probe.py"
)
STABLE_PREFIX: str = (
    (
        "You are validating prefix cache observability for a session-aware "
        "LLM router. Keep the repeated context stable so the backend has an "
        "opportunity to reuse the prompt prefix. "
    )
    * 80
).strip()
STABLE_PREFIX_SHA256: str = hashlib.sha256(STABLE_PREFIX.encode()).hexdigest()
DEFAULT_THRESHOLDS: dict[str, float | str] = {
    "min_success_rate": 1.0,
    "min_cached_token_reporting": "missing",
    "min_cached_token_field_rate": 0.0,
    "min_cached_prompt_ratio": 0.0,
}


def messages(*, repeat_index: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a concise benchmark assistant."},
        {
            "role": "user",
            "content": (
                f"{STABLE_PREFIX}\n\n"
                f"Probe turn {repeat_index}: answer with one short sentence."
            ),
        },
    ]


def _spec(
    *, repeats: int, max_tokens: int, temperature: float, timeout: float
) -> dict[str, Any]:
    if type(repeats) is not int or not 1 <= repeats <= MAX_REPEATS:
        raise ValueError(
            f"Cache probe repeats must be an integer from 1 to {MAX_REPEATS}"
        )
    validate_request_params(
        params={"max_tokens": max_tokens, "temperature": temperature},
        limits={"max_output_tokens": max_tokens},
    )
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("Cache probe timeout must be finite and positive")
    return {
        "version": PROBE_VERSION,
        "probe_kind": "repeated-prefix-cache-token-probe",
        "repeats": repeats,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "stable_prefix_chars": len(STABLE_PREFIX),
        "stable_prefix_sha256": STABLE_PREFIX_SHA256,
        "native_prefix_sha256": STABLE_PREFIX_SHA256[:16],
        "unique_suffix_pattern": "probe_turn_index",
        "request_settings": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        },
    }


def _case_spec(case: dict[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata")
    spec = metadata.get("cache_probe") if isinstance(metadata, dict) else None
    if not isinstance(spec, dict) or not isinstance(spec.get("request_settings"), dict):
        raise ValueError("Cache probe requires its frozen native workload")
    settings = spec["request_settings"]
    expected = _spec(
        repeats=spec.get("repeats"),
        max_tokens=settings.get("max_tokens"),
        temperature=settings.get("temperature"),
        timeout=settings.get("timeout"),
    )
    if canonical(spec) != canonical(expected):
        raise ValueError("Cache probe differs from its frozen native workload")
    return spec


def preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    spec = _case_spec(case)
    if manifest["profile"] == "standard":
        raise ValueError("Cache observations do not establish a Standard holdout")
    if (
        case_request_fields(case)
        or "answer" in case
        or case["id"] != BENCHMARK_ID + "/repeated-prefix"
        or case["messages"] != messages(repeat_index=0)
    ):
        raise ValueError("Cache probe messages differ from the native workload")
    limits = manifest["limits"]
    if limits["concurrency"] != 1 or spec["repeats"] > limits["max_calls_per_case"]:
        raise ValueError(
            "Cache probe requires sequential repeats within its call budget"
        )
    settings = spec["request_settings"]
    if (
        manifest["output_policy"] != "bounded"
        or limits["total_timeout_s"] != settings["timeout"]
    ):
        raise ValueError("Cache probe limits differ from its native request settings")
    selected = {
        cell["target_id"]
        for cell in manifest.get("execution_cells", [])
        if cell["case_id"] == case["id"]
    }
    targets = [
        target
        for target in manifest["targets"]
        if "execution_cells" not in manifest or target["id"] in selected
    ]
    if len(targets) > MAX_TARGETS or (
        len(targets) == MAX_TARGETS
        and (
            [target["kind"] for target in targets] != ["mom", "single"]
            or targets[0]["base_url"].rstrip("/") == targets[1]["base_url"].rstrip("/")
        )
    ):
        raise ValueError(
            "Cache probe supports one arm or Router then a distinct backend"
        )
    for target in targets:
        if target.get("session_mode") != SESSION_AWARE:
            raise ValueError("Cache probe requires session_aware targets")
        effective = {**manifest["sampling"], **target.get("request_params", {})}
        if effective != {key: settings[key] for key in ("max_tokens", "temperature")}:
            raise ValueError(
                "Cache probe generation differs from its native request settings"
            )
    options = manifest.setdefault("benchmark_options", {})
    if not isinstance(options, dict):
        raise ValueError("benchmark_options must be an object")
    config = options.setdefault(BENCHMARK_ID, {})
    if not isinstance(config, dict) or set(config) - {"probe_version", "thresholds"}:
        raise ValueError("Cache probe options support probe_version and thresholds")
    if config.setdefault("probe_version", PROBE_VERSION) != PROBE_VERSION:
        raise ValueError("Cache probe version differs from the installed adapter")
    thresholds = config.get("thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) - DEFAULT_THRESHOLDS.keys():
        raise ValueError("Cache probe thresholds contain an unsupported field")
    thresholds = {**DEFAULT_THRESHOLDS, **thresholds}
    for name, value in thresholds.items():
        if name == "min_cached_token_reporting":
            if value not in CACHE_REPORTING_STATES:
                raise ValueError("Cache probe reporting threshold is unsupported")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                "Cache probe numeric thresholds must be between zero and one"
            )
    config["thresholds"] = thresholds


def execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    repeats = _case_spec(case)["repeats"]
    for index in range(repeats):
        response = context.call(messages=messages(repeat_index=index))
        if not response.get("output_complete", True):
            raise CallFailure(
                "Cache probe output limit reached; repeats are incomplete"
            )
    return {
        "answer": None,
        "correct": None,
        "score": None,
        "details": {
            "cache_observation": {
                "version": CACHE_OBSERVATION_RESULT,
                "planned_repeats": repeats,
                "completed_repeats": repeats,
            }
        },
    }


def adapter() -> BenchmarkAdapter:
    return BenchmarkAdapter(
        id=BENCHMARK_ID,
        title="Repeated-prefix cache observation",
        kind="cache-observation",
        source_url=SOURCE_URL,
        version=PROBE_VERSION,
        execute=execute,
        preflight=preflight,
        result_kind=CACHE_OBSERVATION_RESULT,
    )


def prepare(
    *,
    store: Path,
    repeats: int = 6,
    max_tokens: int = 16,
    temperature: float = 0.0,
    timeout: float = 120.0,
    profile: str = "quick",
) -> dict[str, Any]:
    if profile not in {"smoke", "quick"}:
        raise ValueError(
            "Cache probe preparation supports smoke/quick diagnostics only"
        )
    spec = _spec(
        repeats=repeats, max_tokens=max_tokens, temperature=temperature, timeout=timeout
    )
    source = {
        "url": SOURCE_URL,
        "revision": SOURCE_REVISION,
        "partition": "repeated-prefix",
        "files": [
            {
                "name": "cache_probe.py",
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        ],
        "normalizer": PROBE_VERSION,
        "revision_verification": "declared-native-revision-with-bundled-recipe-digest",
        "selection": spec,
    }
    case = {
        "id": BENCHMARK_ID + "/repeated-prefix",
        "benchmark": BENCHMARK_ID,
        "messages": messages(repeat_index=0),
        "metadata": {
            "split": "dev",
            "stratum": "repeated-prefix",
            "cache_probe": spec,
            "task_identity": native_task_identity(
                benchmark=BENCHMARK_ID,
                row={"id": "repeated-prefix"},
                source=source_identity(source=source, partition=source["partition"]),
            ),
        },
    }
    return _write_dataset(
        root=store,
        cases=[case],
        profile=profile,
        seed=0,
        sources={BENCHMARK_ID: source},
        custom=True,
        selection="cache-probe-native-order-v1",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a repeated-prefix cache probe"
    )
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument(
        "--repeats",
        type=int,
        default=6,
        help=f"Native repeats per sequence (1-{MAX_REPEATS}; default: 6)",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--profile", choices=("smoke", "quick"), default="quick")
    args = parser.parse_args()
    try:
        prepared = prepare(
            store=args.store,
            repeats=args.repeats,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            profile=args.profile,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(prepared, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
