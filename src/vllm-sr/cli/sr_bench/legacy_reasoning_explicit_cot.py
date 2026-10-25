from __future__ import annotations

import hashlib
from typing import Any

from .adapters import AdapterContext, BenchmarkAdapter
from .canonical import digest
from .legacy_mmlu import normalize_records
from .legacy_reasoning import (
    CONFIGURATIONS,
    REQUIREMENTS,
    SOURCE_REVISION,
    SOURCE_URL,
    case_identity,
    inspect_runtime,
)
from .legacy_reasoning import validate_case as validate_plain_case
from .legacy_reasoning_modes import FIELD, native_dataset

PROTOCOL: str = "bench-reasoning-explicit-cot-v1"
SUFFIX: str = "-explicit-cot"
BENCHMARKS: dict[str, tuple[str, str]] = {
    **CONFIGURATIONS,
    "mmlu-pro": ("MMLU-Pro", "multiple-choice"),
}


def configuration(benchmark: str) -> str:
    if not benchmark.startswith("legacy-") or not benchmark.endswith(SUFFIX):
        raise ValueError("Unknown explicit-CoT diagnostic")
    name = benchmark.removeprefix("legacy-").removesuffix(SUFFIX)
    if name not in BENCHMARKS:
        raise ValueError("Unknown explicit-CoT diagnostic")
    return name


def mmlu_record(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "category": question["category"],
        "options": question["options"],
        "answer": question["correct_answer"],
    }


def validate_case(case: dict[str, Any], config: dict[str, Any]) -> None:
    name = configuration(case["benchmark"])
    if (
        not isinstance(config, dict)
        or set(config)
        != {
            "protocol",
            "source_revision",
            "requirements_sha256",
            "source",
            "selection",
            "plain_options",
        }
        or config["protocol"] != PROTOCOL
        or config["source_revision"] != SOURCE_REVISION
        or config["requirements_sha256"]
        != hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
        or not isinstance(config["selection"], dict)
        or native_dataset(config["selection"].get("dataset")) != name
    ):
        raise ValueError("Explicit-CoT requires its pinned native source and protocol")
    payload = case.get("answer")
    if not isinstance(payload, dict) or set(payload) != {"question", "plain_case"}:
        raise ValueError(
            "Explicit-CoT requires a private native question and plain case"
        )
    question, plain = payload["question"], payload["plain_case"]
    if (
        not isinstance(question, dict)
        or set(question)
        != {
            "question_id",
            "category",
            "question",
            "options",
            "correct_answer",
            "cot_content",
            "metadata",
        }
        or not isinstance(plain, dict)
        or plain.get("benchmark") != "legacy-" + name
        or (
            question["cot_content"] is not None
            and not isinstance(question["cot_content"], str)
        )
        or (
            question["metadata"] is not None
            and not isinstance(question["metadata"], dict)
        )
    ):
        raise ValueError("Invalid native explicit-CoT question")
    if name == "mmlu-pro":
        if (
            config["plain_options"] != {}
            or normalize_records([mmlu_record(question)])[0] != plain
        ):
            raise ValueError("Explicit-CoT differs from its MMLU plain case")
    else:
        validate_plain_case(case=plain, config=config["plain_options"])
        if plain["answer"] != question:
            raise ValueError("Explicit-CoT differs from its native plain question")
        if any(
            config[key] != config["plain_options"][key]
            for key in ("source", "selection", "source_revision", "requirements_sha256")
        ):
            raise ValueError("Explicit-CoT and plain source identities differ")
    metadata = {
        "protocol": PROTOCOL,
        "native_dataset": name,
        "native_question_id": question["question_id"],
        "native_question_sha256": digest(question),
        "source_sha256": digest(config["source"]),
        "plain_case_sha256": digest(plain),
        "prompt_sha256": digest(case.get("messages")),
        "stratum": question["category"],
    }
    if case.get("metadata") != metadata or case["id"] != case_identity(
        benchmark=case["benchmark"],
        source_sha256=metadata["source_sha256"],
        selection=config["selection"],
        native_id=question["question_id"],
    ):
        raise ValueError("Explicit-CoT differs from its frozen native identity")


def _preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    if FIELD not in manifest:
        raise ValueError("Explicit-CoT requires a frozen reasoning condition manifest")
    config = manifest.get("benchmark_options", {}).get(case["benchmark"], {})
    validate_case(case=case, config=config)
    section = manifest[FIELD]
    if any(section[key] != config[key] for key in ("source", "selection")):
        raise ValueError("Explicit-CoT source differs from its condition selection")
    inspect_runtime(cache=cache)


def _execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    from .external import _harness  # noqa: PLC0415 - adapter registry cycle

    return _harness(case=case, context=context)


ADAPTERS: tuple[BenchmarkAdapter, ...] = tuple(
    BenchmarkAdapter(
        id="legacy-" + name + SUFFIX,
        title=title + " (legacy explicit-CoT diagnostic)",
        kind=kind,
        source_url=SOURCE_URL,
        version=PROTOCOL,
        execute=_execute,
        preflight=_preflight,
        requires_answer=True,
    )
    for name, (title, kind) in BENCHMARKS.items()
)
