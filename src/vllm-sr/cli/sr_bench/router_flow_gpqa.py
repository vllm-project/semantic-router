from __future__ import annotations

import hashlib
import re
from typing import Any

from .adapters import AdapterContext, BenchmarkAdapter

BENCHMARK: str = "router-flow-gpqa-evalscope"
PROTOCOL: str = "evalscope-1.8.1-sr-final-v1"
EVALSCOPE_VERSION: str = "1.8.1"
EVALSCOPE_REVISION: str = "fce1d21391dc2d7b45c9cf0edb9b9e40d526aed3"
NATIVE_BENCHMARK: str = "gpqa_diamond"
NATIVE_SOURCE: str = "https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond"
SELECTION: str = "native-ordered-prefix"
SEED: int = 42
SMOKE_LIMIT: int = 20
CHOICE_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")

# This version freezes the maintained row; its exporter rejects protocol drift.
# https://github.com/vllm-project/semantic-router/blob/c36c2bf5b55a90c95102d65abe1f04c846e44533/bench/router_flow/real_eval/evalscope_suite.yaml#L1-L35
NATIVE_GENERATION: dict[str, int] = {
    "temperature": 0,
    "max_tokens": 32768,
    "batch_size": 2,
    "timeout": 900,
    "retries": 2,
    "retry_interval": 15,
}
SAMPLING: dict[str, int] = {
    key: NATIVE_GENERATION[key] for key in ("temperature", "max_tokens")
}
LIMITATIONS: tuple[str, ...] = (
    "This is an EvalScope-derived Router Flow GPQA smoke diagnostic, not native benchmark parity or a published sr-bench GPQA score.",
    "Native prompts and the EvalScope 1.8.1 grader apply only to completed final answers; sr-bench retains streaming, final-only extraction, no retries and output-limit failures.",
    "Accuracy uses all planned cases, including failures and undispatched cases. The native source is an operator-supplied snapshot; this smoke prefix is not an unseen holdout.",
)


def validate_case(case: dict[str, Any]) -> None:
    metadata = case.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Router Flow GPQA requires native case metadata")
    choices = metadata.get("choices")
    question = metadata.get("question")
    index = metadata.get("source_index")
    if (
        case.get("benchmark") != BENCHMARK
        or metadata.get("protocol") != PROTOCOL
        or not isinstance(question, str)
        or not question
        or type(index) is not int
        or not 0 <= index < SMOKE_LIMIT
        or isinstance(metadata.get("native_sample_id"), bool)
        or not isinstance(metadata.get("native_sample_id"), (str, int))
        or not str(metadata["native_sample_id"])
        or not isinstance(choices, list)
        or len(choices) != len(CHOICE_LETTERS)
        or any(not isinstance(choice, str) for choice in choices)
        or case.get("answer") not in CHOICE_LETTERS
        or metadata.get("question_sha256")
        != hashlib.sha256(question.encode()).hexdigest()
        or not isinstance(metadata.get("source_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", metadata["source_sha256"])
    ):
        raise ValueError("Router Flow GPQA native case metadata is invalid")
    messages = case.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or set(messages[0]) != {"role", "content"}
        or messages[0]["role"] != "user"
        or not isinstance(messages[0]["content"], str)
        or not messages[0]["content"]
    ):
        raise ValueError("Router Flow GPQA requires one native text prompt")


def _preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    validate_case(case)
    options = manifest.get("benchmark_options", {}).get(BENCHMARK, {})
    if not isinstance(options, dict):
        raise ValueError("Router Flow GPQA requires frozen benchmark options")
    source = options.get("source", {})
    selected = manifest["cases"]
    count = options.get("limit")
    if (
        options.get("protocol") != PROTOCOL
        or options.get("source_revision") != EVALSCOPE_REVISION
        or options.get("native_generation") != NATIVE_GENERATION
        or options.get("selection") != SELECTION
        or not isinstance(options.get("suite_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", options["suite_sha256"])
        or type(count) is not int
        or not 1 <= count <= SMOKE_LIMIT
        or len(selected) != count
        or options.get("custom_subset") is not (count < SMOKE_LIMIT)
        or not isinstance(source, dict)
        or not isinstance(source.get("revision"), str)
        or not source["revision"].strip()
        or source.get("sha256") != case["metadata"]["source_sha256"]
        or source.get("url") != NATIVE_SOURCE
        or manifest["profile"] != "smoke"
        or manifest["seed"] != SEED
        or manifest["output_policy"] != "bounded"
        or manifest["sampling"] != SAMPLING
        or any(
            row.get("benchmark") != BENCHMARK
            or row.get("metadata", {}).get("source_index") != index
            for index, row in enumerate(selected)
        )
        or any(item not in manifest.get("limitations", []) for item in LIMITATIONS)
    ):
        raise ValueError("Router Flow GPQA requires its frozen smoke protocol")
    limits = manifest["limits"]
    if (
        limits["max_calls_per_case"] != 1
        or limits["concurrency"] != NATIVE_GENERATION["batch_size"]
        or limits["total_timeout_s"] != NATIVE_GENERATION["timeout"]
        or limits["case_timeout_s"] < NATIVE_GENERATION["timeout"]
        or limits["max_output_tokens"] < SAMPLING["max_tokens"]
        or any(
            {**SAMPLING, **target.get("request_params", {})} != SAMPLING
            for target in manifest["targets"]
        )
    ):
        raise ValueError(
            "Router Flow GPQA generation or limits conflict with its protocol"
        )
    from .external import preflight_case  # noqa: PLC0415 - registry/contract cycle

    preflight_case(case, manifest, cache)


def _execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    from .external import execute_case  # noqa: PLC0415 - registry/contract cycle

    return execute_case(case, context)


ADAPTER: BenchmarkAdapter = BenchmarkAdapter(
    id=BENCHMARK,
    title="Router Flow GPQA (EvalScope-derived smoke)",
    kind="multiple-choice",
    source_url=NATIVE_SOURCE,
    version=PROTOCOL,
    execute=_execute,
    preflight=_preflight,
    requires_answer=True,
)
