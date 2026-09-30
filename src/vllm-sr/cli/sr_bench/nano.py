"""EXPERIMENTAL sr-bench-nano scope: frozen five-benchmark id lists and run protocol.

Temporary and unversioned relative to sr-bench 1.0. Nano fixes only questions,
graders and the run protocol; each operator brings their own targets and keeps
their own results. Nothing here names a model, endpoint or dataset text.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from functools import cache
from pathlib import Path

from .canonical import digest
from .grading import MCQ_GRADER_VERSION

IDS_VERSION = "sr-bench-nano-ids-v1"
IDS_FILE = Path(__file__).with_name("nano_ids_v1.json")
PROFILES = {"nano": "dev", "nano-holdout": "holdout"}
SEED = 20260918
BENCHMARKS = ("gpqa-diamond", "hle", "livecodebench", "mmlu-pro", "simpleqa-verified")
WEIGHTS = dict.fromkeys(BENCHMARKS, 0.2)
# (offset, count) windows in sr-bench's stratified-hash-v1 order of each population.
WINDOWS = {
    "mmlu-pro": {"nano": (0, 150), "nano-holdout": (500, 500)},
    "simpleqa-verified": {"nano": (0, 100), "nano-holdout": (100, 300)},
    "gpqa-diamond": {"nano": (0, 60), "nano-holdout": (60, 138)},
    "livecodebench": {"nano": (0, 60), "nano-holdout": (60, 150)},
    "hle": {"nano": (0, 60), "nano-holdout": (60, 150)},
}
# LiveCodeBench v5+v6 additions at sr-bench's pinned release_v6 revision.
LCB_FILES = ("test5.jsonl", "test6.jsonl")
# No output cap: max_output_tokens only bounds an explicit per-target max_tokens.
# max_output_chars is a memory guard far above any realistic answer.
LIMITS = {
    "total_timeout_s": 1800,
    "idle_timeout_s": 1800,
    "case_timeout_s": 2400,
    "max_run_seconds": 604800,
    "max_output_tokens": 1048576,
    "max_output_chars": 8388608,
}
ROUND_CAPS = frozenset(2**k for k in range(8, 21)) | {1000, 2000, 4000, 8000, 16000}
TIMEOUT_ERRORS = (
    "total request deadline exceeded",
    "cancelled or total request deadline exceeded",
    "ReadTimeout",
    "Run cancelled or wall-time budget exhausted",
)

HLE_GRADER = "sr-bench-nano-hle-exact-v1"
HLE_SYSTEM = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
HLE_MAX_WORDS, HLE_MAX_CHARS = 4, 40
_HLE_NUMERIC = re.compile(r"^-?\d+(\.\d+)?$|^-?\d+/\d+$")
_HLE_TEXT = re.compile(r"^[A-Za-z][A-Za-z0-9 .'\-]*[A-Za-z0-9]$|^[A-Za-z]$")
_EXACT_LINE = re.compile(r"exact\s+answer\s*[:\uff1a]\s*(.+)", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")

SIMPLEQA_GRADER = "sr-bench-nano-simpleqa-official-v1"
# openai/simple-evals@652c89d0ca9df547706735883097e9537d40dc47 simpleqa_eval.py (MIT).
SIMPLEQA_TEMPLATE_SHA256 = (
    "063c04ea798f91c4706c050b66a721afc77a458b12d8f0d72286d4b0eee8a3ee"
)
SIMPLEQA_TEMPLATE_SOURCE = (
    "https://github.com/openai/simple-evals/blob/"
    "652c89d0ca9df547706735883097e9537d40dc47/simpleqa_eval.py"
)
_VERDICTS = {"A": "correct", "B": "incorrect", "C": "not_attempted"}


def is_nano(manifest):
    return manifest.get("profile") in PROFILES


def hle_answer_kind(answer):
    """Classify HLE references that are deterministically checkable, else None."""
    text = str(answer).strip()
    if _HLE_NUMERIC.match(text):
        return "numeric"
    if (
        _HLE_TEXT.match(text)
        and len(text.split()) <= HLE_MAX_WORDS
        and len(text) <= HLE_MAX_CHARS
    ):
        return "text"
    return None


def render_hle(cases):
    """Keep exact-match, judge-free HLE items and add HLE's exact-answer prompt."""
    kept = []
    for case in cases:
        if case["metadata"].get("answer_type") != "exactMatch":
            continue
        kind = hle_answer_kind(case["answer"])
        if kind is None:
            continue
        case["messages"] = [
            {"role": "system", "content": HLE_SYSTEM},
            *case["messages"],
        ]
        case["metadata"]["answer_kind"] = kind
        kept.append(case)
    return kept


def _hle_text(value):
    value = value.strip()
    for _ in range(3):
        value = value.rstrip(".").strip()
        for wrap in ("**", "__", "`", "$", '"', "'"):
            if (
                len(value) > 2 * len(wrap)
                and value.startswith(wrap)
                and value.endswith(wrap)
            ):
                value = value[len(wrap) : -len(wrap)].strip()
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value).rstrip(".").strip()
    return re.sub(r"\s+", " ", value).casefold()


def _hle_number(value):
    value = _hle_text(value).replace(",", "").replace(" ", "")
    fraction = re.fullmatch(r"(-?\d+)/(\d+)", value) or re.fullmatch(
        r"\\frac\{(-?\d+)\}\{(\d+)\}", value
    )
    try:
        if fraction:
            return Fraction(int(fraction.group(1)), int(fraction.group(2)))
        if re.fullmatch(r"-?\d+(\.\d+)?", value):
            return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def grade_hle(case, final):
    """Last 'Exact Answer:' line (else last \\boxed{}); numeric within reference precision."""
    reference = str(case["answer"]).strip()
    lines = _EXACT_LINE.findall(final or "")
    boxed = _BOXED.findall(final or "")
    answer, how = (
        (lines[-1].strip(), "exact_answer_line")
        if lines
        else ((boxed[-1].strip(), "boxed") if boxed else (None, "unparsed"))
    )
    correct = False
    if answer is not None:
        if hle_answer_kind(reference) == "numeric":
            expected, got = _hle_number(reference), _hle_number(answer)
            if expected is not None and got is not None:
                decimals = (
                    len(reference.split(".")[1])
                    if "." in reference and "/" not in reference
                    else 0
                )
                tolerance = Fraction(1, 2 * 10**decimals) if decimals else 0
                correct = abs(got - expected) <= tolerance
        else:
            correct = _hle_text(answer) == _hle_text(reference)
    return {
        "answer": final,
        "correct": correct,
        "score": float(correct),
        "details": {
            "grader_version": HLE_GRADER,
            "extracted": answer[:500] if answer else None,
            "answer_status": how,
        },
    }


@cache
def simpleqa_template():
    text = IDS_FILE.with_name("nano_simpleqa_grader.txt").read_text().strip()
    if hashlib.sha256(text.encode()).hexdigest() != SIMPLEQA_TEMPLATE_SHA256:
        raise ValueError("Installed SimpleQA grader template differs from its pin")
    return text


def simpleqa_messages(case, predicted):
    prompt = simpleqa_template().format(
        question=case["messages"][-1]["content"],
        target=case["answer"],
        predicted_answer=predicted,
    )
    return [{"role": "user", "content": prompt}]


def simpleqa_verdict(text):
    """Official letter parse; an unparsable grader reply fails closed."""
    match = re.search(r"(A|B|C)", text or "")
    if not match:
        raise ValueError("SimpleQA grader returned no A/B/C letter; response retained")
    return _VERDICTS[match.group(0)]


def _canonical_hashes(case):
    prompt = digest(case["messages"])
    if case["benchmark"] == "livecodebench":
        source = digest(case["metadata"]["source_record"])
        return {
            "prompt_sha256": prompt,
            "source_record_sha256": source,
            "content_sha256": digest({"messages": case["messages"], "source": source}),
        }
    return {
        "prompt_sha256": prompt,
        "answer_sha256": digest(case["answer"]),
        "content_sha256": digest(
            {"messages": case["messages"], "answer": case["answer"]}
        ),
    }


def task_record(case):
    return {
        "id": case["id"],
        "stratum": str(case["metadata"].get("stratum", "all")),
        **_canonical_hashes(case),
    }


def body_sha256(document):
    return digest({k: v for k, v in document.items() if k != "sha256"})


@cache
def frozen_ids():
    document = json.loads(IDS_FILE.read_text())
    if document.get("version") != IDS_VERSION or document.get("seed") != SEED:
        raise ValueError("Installed nano id list has an unexpected version or seed")
    if body_sha256(document) != document.get("sha256"):
        raise ValueError("Installed nano id list does not match its recorded sha256")
    return document


def split_tasks(benchmark, profile):
    return frozen_ids()["benchmarks"][benchmark]["splits"][profile]["tasks"]


def verify_cases(manifest):
    """Every nano case must be a frozen v1 task of its split with identical hashes."""
    profile = manifest["profile"]
    expected = {
        (benchmark, task["id"]): task
        for benchmark in BENCHMARKS
        for task in split_tasks(benchmark, profile)
    }
    for case in manifest["cases"]:
        if case.get("benchmark") not in WEIGHTS:
            raise ValueError("Nano profiles accept only the five nano benchmarks")
        task = expected.get((case["benchmark"], case["id"]))
        if task is None:
            raise ValueError(f"Case {case['id']} is not in the frozen {profile} list")
        if task_record(case) != task:
            raise ValueError(
                f"Case {case['id']} differs from its frozen {IDS_VERSION} hashes"
            )


def _simpleqa_grader(manifest):
    config = manifest.get("benchmark_options", {}).get("simpleqa-verified", {})
    judge = manifest.get("auxiliary_targets", {}).get(config.get("judge"))
    if judge is None:
        raise ValueError(
            "SimpleQA grader is not configured; nano has no default grader. "
            "Pass --grader-base-url, --grader-model and --grader-api-key-env to "
            "'vllm-sr benchmark nano manifest' (or SR_BENCH_NANO_GRADER_* env)"
        )
    if config.get("grader_version") != SIMPLEQA_GRADER:
        raise ValueError(f"Nano SimpleQA requires grader_version={SIMPLEQA_GRADER}")
    if judge.get("request_params", {}).get("temperature") != 0:
        raise ValueError("Nano SimpleQA grader requires request_params.temperature=0")
    return judge


def apply_policy(m):
    """Freeze the nano protocol before generic plan validation."""
    if m.get("output_policy", "uncapped") != "uncapped":
        raise ValueError(
            "Nano profiles impose no output cap; output_policy is uncapped"
        )
    m["output_policy"] = "uncapped"
    if m.get("cost_policy") != "capability_only":
        raise ValueError(
            "Nano output is uncapped, so dispatch cost cannot be bounded; "
            "set cost_policy: capability_only (usage and known costs are still reported)"
        )
    m["limits"] = {**LIMITS, **m.get("limits", {})}
    if m.get("seed", SEED) != SEED:
        raise ValueError(f"Nano profiles use the frozen seed {SEED}")
    verify_cases(m)
    if any(c["benchmark"] == "simpleqa-verified" for c in m["cases"]):
        _simpleqa_grader(m)
    return m


def is_timeout(message):
    return message in TIMEOUT_ERRORS


def report_section(manifest, results, calls):
    """Grader identities, output-cap evidence and the nano score per target."""
    options = manifest.get("benchmark_options", {})
    graders = {
        "mmlu-pro": MCQ_GRADER_VERSION,
        "gpqa-diamond": MCQ_GRADER_VERSION,
        "hle": HLE_GRADER,
        "livecodebench": {
            "grader": "lcb_runner sandbox",
            "source_revision": options.get("livecodebench", {}).get("source_revision"),
            "sandbox_image": options.get("livecodebench", {}).get("sandbox_image"),
        },
    }
    if any(c["benchmark"] == "simpleqa-verified" for c in manifest["cases"]):
        judge = _simpleqa_grader(manifest)
        graders["simpleqa-verified"] = {
            "grader_version": SIMPLEQA_GRADER,
            "template_sha256": SIMPLEQA_TEMPLATE_SHA256,
            "template_source": SIMPLEQA_TEMPLATE_SOURCE,
            "target_id": judge["id"],
            "model": judge["model"],
            "base_url": judge["base_url"],
            "request_params": judge.get("request_params", {}),
            "header_names": sorted(judge.get("header_env", {})),
            "stream": judge.get("stream", True),
        }
    document = frozen_ids()
    full = {
        benchmark: {task["id"] for task in split_tasks(benchmark, manifest["profile"])}
        for benchmark in BENCHMARKS
    }
    planned = {
        benchmark: {c["id"] for c in manifest["cases"] if c["benchmark"] == benchmark}
        for benchmark in BENCHMARKS
    }
    targets = []
    for target in manifest["targets"]:
        rows = [r for r in results if r["target_id"] == target["id"]]
        subject = [
            c
            for c in calls
            if c["target_id"] == target["id"] and c["role"] == "subject"
        ]
        max_tokens = target.get("request_params", {}).get(
            "max_tokens", manifest["sampling"].get("max_tokens")
        )
        suspected = sorted(
            {
                c["case_id"]
                for c in subject
                if c.get("finish_reason") == "length"
                or (c.get("usage") or {}).get("output_tokens") in ROUND_CAPS
            }
        )
        targets.append(
            {
                "id": target["id"],
                "model": target["model"],
                "stream": target.get("stream", True),
                "max_tokens_sent": max_tokens is not None,
                "max_tokens": max_tokens,
                "timeouts": sum(r["status"] == "timeout" for r in rows),
                "suspected_output_cap_cases": suspected,
                "terminal": sum(r["status"] in {"completed", "timeout"} for r in rows),
            }
        )
    return {
        "scope": "EXPERIMENTAL sr-bench-nano",
        "profile": manifest["profile"],
        "ids_version": IDS_VERSION,
        "ids_sha256": document["sha256"],
        "full_split": planned == full and "execution_cells" not in manifest,
        "weights": WEIGHTS,
        "generation_policy": "exactly one generation per task per target",
        "output_policy": "uncapped; max_tokens is sent only when a target sets it",
        "per_request_timeout_s": manifest["limits"]["total_timeout_s"],
        "graders": graders,
        "targets": targets,
    }
