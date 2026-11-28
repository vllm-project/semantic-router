"""Opt-in Router Flow projection of EvalScope's native MRCR contract."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from .adapters import AdapterContext, BenchmarkAdapter
from .dataset_io import MAX_ROWS, file_digest
from .grading import MRCR_METRIC, MRCR_SEQUENCE_RESULT, MRCR_TOKEN_BINS, ResultKind
from .sources import _write_dataset, read_records
from .transport import CallFailure

BENCHMARK_ID: str = "router-flow-mrcr"
SOURCE_URL: str = "https://huggingface.co/datasets/openai-mirror/mrcr"
NATIVE_ADAPTER_NAME: str = "openai_mrcr"
# EvalScope 1.8.1 registers this native dataset ID:
# https://github.com/modelscope/evalscope/blob/fce1d21391dc2d7b45c9cf0edb9b9e40d526aed3/evalscope/benchmarks/openai_mrcr/openai_mrcr_adapter.py#L19-L36
NATIVE_DATASET_ID: str = "openai-mirror/mrcr"
NATIVE_EVALSCOPE_VERSION: str = "1.8.1"
NATIVE_EVALSCOPE_REVISION: str = "fce1d21391dc2d7b45c9cf0edb9b9e40d526aed3"
NATIVE_SELECTION_VERSION: str = "evalscope-openai-mrcr-selection-v1"
PROTOCOL_VERSION: str = "evalscope-mrcr-sequence-ratio-v1"
RESULT_KIND: ResultKind = MRCR_SEQUENCE_RESULT
# The native adapter defaults to this tokenizer:
# https://github.com/modelscope/evalscope/blob/fce1d21391dc2d7b45c9cf0edb9b9e40d526aed3/evalscope/benchmarks/openai_mrcr/openai_mrcr_adapter.py#L126-L130
TOKENIZER: str = "o200k_base"
SMOKE_RAW_LIMIT: int = 5
SMOKE_NEEDLE_COUNT: int = 2
SMOKE_MIN_CONTEXT_TOKENS: int = 65536
SMOKE_MAX_CONTEXT_TOKENS: int = 262144
FORMAL_NEEDLE_COUNT: int = 8
FORMAL_MAX_CONTEXT_TOKENS: int = 131072
# The registered value is None; native str.lstrip(None) strips all whitespace:
# https://github.com/modelscope/evalscope/blob/fce1d21391dc2d7b45c9cf0edb9b9e40d526aed3/evalscope/benchmarks/openai_mrcr/openai_mrcr_adapter.py#L79-L82
# https://github.com/modelscope/evalscope/blob/fce1d21391dc2d7b45c9cf0edb9b9e40d526aed3/evalscope/benchmarks/openai_mrcr/openai_mrcr_adapter.py#L193-L196
PREFIX_FILTER: str | None = None


class TokenEncoding(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...


def _token_count(text: str, encoding: TokenEncoding) -> int:
    return len(encoding.encode(text))


def _context_token_count(prompt: str, encoding: TokenEncoding) -> int:
    try:
        messages = json.loads(prompt)
    except json.JSONDecodeError:
        return 0
    if not isinstance(messages, list):
        return 0
    return sum(
        _token_count(message.get("content", ""), encoding)
        for message in messages
        if isinstance(message, dict)
    )


def _messages(prompt: str) -> list[dict[str, Any]]:
    try:
        rows = json.loads(prompt)
    except json.JSONDecodeError:
        return []
    if not isinstance(rows, list):
        return []
    supported = {"system", "user", "assistant"}
    messages: list[dict[str, Any]] = []
    for message in rows:
        if not isinstance(message, dict) or message.get("role") not in supported:
            continue
        content = message.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            raise ValueError("MRCR messages must contain plain text")
        messages.append({"role": message["role"], "content": content})
    return messages


def _bin_index(total_tokens: int) -> int:
    last_index = len(MRCR_TOKEN_BINS) - 1
    for index, (lower, upper) in enumerate(MRCR_TOKEN_BINS):
        if index in {0, last_index}:
            if lower <= total_tokens <= upper:
                return index
        elif lower <= total_tokens < upper:
            return index
    return 0


def prepare_records(
    *,
    records: Iterable[dict[str, Any]],
    profile: str,
    limit: int | None,
    encoding: TokenEncoding,
    source_sha256: str,
    revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if profile not in {"smoke", "quick"}:
        raise ValueError("MRCR preparation supports smoke and quick profiles only")
    if not isinstance(source_sha256, str) or not re_full_sha256(source_sha256):
        raise ValueError("MRCR source requires a SHA-256 digest")
    if not isinstance(revision, str) or not revision:
        raise ValueError("MRCR local source requires a declared revision")
    raw_limit = SMOKE_RAW_LIMIT if profile == "smoke" else None
    if limit is not None:
        if type(limit) is not int or limit <= 0 or limit > MAX_ROWS:
            raise ValueError("MRCR limit must be a positive supported raw-row count")
        if profile == "smoke" and limit > SMOKE_RAW_LIMIT:
            raise ValueError(f"MRCR smoke limit cannot exceed {SMOKE_RAW_LIMIT}")
        raw_limit = limit
    scanned = list(records)
    if len(scanned) > MAX_ROWS:
        raise ValueError("MRCR source exceeds the supported row count")
    if raw_limit is not None:
        scanned = scanned[:raw_limit]

    needle_count = SMOKE_NEEDLE_COUNT if profile == "smoke" else FORMAL_NEEDLE_COUNT
    minimum = SMOKE_MIN_CONTEXT_TOKENS if profile == "smoke" else None
    maximum = (
        SMOKE_MAX_CONTEXT_TOKENS if profile == "smoke" else FORMAL_MAX_CONTEXT_TOKENS
    )
    cases: list[dict[str, Any]] = []
    for source_index, record in enumerate(scanned):
        if not isinstance(record, dict):
            raise ValueError("MRCR source rows must be objects")
        if record.get("n_needles") != needle_count:
            continue
        prompt = record.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("MRCR prompt must be a JSON string")
        input_tokens = _context_token_count(prompt, encoding)
        if maximum is not None and input_tokens > maximum:
            continue
        if minimum is not None and input_tokens <= minimum:
            continue
        answer = record.get("answer")
        if not isinstance(answer, str):
            raise ValueError("MRCR reference answer must be a string")
        answer_tokens = _token_count(answer, encoding)
        messages = _messages(prompt)
        if not messages:
            raise ValueError("MRCR source row has no supported chat messages")
        prefix = record.get("random_string_to_prepend")
        if prefix is not None and not isinstance(prefix, str):
            raise ValueError("MRCR prefix must be a string or null")
        total_tokens = input_tokens + answer_tokens
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "selection_version": NATIVE_SELECTION_VERSION,
            "dataset_id": NATIVE_DATASET_ID,
            "native_split": "train",
            "evalscope_version": NATIVE_EVALSCOPE_VERSION,
            "evalscope_revision": NATIVE_EVALSCOPE_REVISION,
            "source_revision": revision,
            "source_sha256": source_sha256,
            "source_index": source_index,
            "profile": "smoke" if profile == "smoke" else "formal",
            "random_string_to_prepend": prefix,
            "n_needles": needle_count,
            "desired_msg_index": record.get("desired_msg_index"),
            "total_messages": record.get("total_messages"),
            "n_chars": record.get("n_chars"),
            "raw_input_tok_cnt": input_tokens,
            "answer_tok_cnt": answer_tokens,
            "total_tok_cnt": total_tokens,
            "bin_index": _bin_index(total_tokens),
        }
        cases.append(
            {
                "id": f"mrcr-{source_sha256[:12]}-{source_index:06d}",
                "benchmark": BENCHMARK_ID,
                "messages": messages,
                "answer": answer,
                "metadata": {"mrcr": metadata},
            }
        )
    return cases, {
        "version": NATIVE_SELECTION_VERSION,
        "profile": "smoke" if profile == "smoke" else "formal",
        "raw_limit": raw_limit,
        "raw_rows_scanned": len(scanned),
        "needle_count": [needle_count],
        "min_context_tokens": minimum,
        "max_context_tokens": maximum,
        "tokenizer": TOKENIZER,
        "source_sha256": source_sha256,
        "source_revision": revision,
    }


def re_full_sha256(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def prepare(
    *,
    benchmark: str,
    profile: str,
    store: Path,
    source_path: Path | str | None,
    revision: str | None,
    seed: int,
    limit: int | None = None,
) -> dict[str, Any]:
    if benchmark != BENCHMARK_ID:
        raise ValueError("MRCR preparation received an unexpected benchmark ID")
    if source_path is None:
        raise ValueError("MRCR preparation requires an operator-supplied local source")
    candidate = Path(source_path).expanduser()
    if candidate.is_symlink():
        raise ValueError("MRCR source must not be a symlink")
    path = candidate.resolve()
    if not path.is_file():
        raise ValueError("MRCR source must be a local JSON, JSONL, or Parquet file")
    if not revision:
        raise ValueError("MRCR local source requires --revision")
    if path.suffix not in {".json", ".jsonl", ".parquet"}:
        raise ValueError("MRCR source must be JSON, JSONL, or Parquet")
    source_sha256 = file_digest(path)
    records = read_records(path)
    if file_digest(path) != source_sha256:
        raise ValueError("MRCR source changed while it was being read")
    if len(records) > MAX_ROWS:
        raise ValueError("MRCR source exceeds the supported row count")
    try:
        import tiktoken  # noqa: PLC0415 - only MRCR preparation needs this extra
    except ImportError as exc:
        raise ValueError(
            "MRCR preparation requires the optional tiktoken dependency; install vllm-sr[mrcr]"
        ) from exc
    encoding = tiktoken.get_encoding(TOKENIZER)
    cases, selection = prepare_records(
        records=records,
        profile=profile,
        limit=limit,
        encoding=encoding,
        source_sha256=source_sha256,
        revision=revision,
    )
    if not cases:
        raise ValueError("MRCR native selection produced no cases")
    source = {
        "url": "local-import",
        "dataset_id": NATIVE_DATASET_ID,
        "revision": revision,
        "files": [{"name": path.name, "sha256": source_sha256}],
        "revision_verification": "declared-revision-with-content-digest",
        "access_note": (
            "Operator-supplied source; preserve the upstream access and license records."
        ),
        "normalizer": PROTOCOL_VERSION,
        "native_evalscope_version": NATIVE_EVALSCOPE_VERSION,
        "native_evalscope_revision": NATIVE_EVALSCOPE_REVISION,
        "selection": selection,
    }
    return _write_dataset(
        root=store,
        cases=cases,
        profile=profile,
        seed=seed,
        sources={BENCHMARK_ID: source},
        custom=limit is not None,
        selection=NATIVE_SELECTION_VERSION,
    )


def native_grade(*, prediction: str, reference: str, prefix: str | None) -> float:
    if prefix:
        if not prediction.startswith(prefix):
            return 0.0
        prediction = prediction.removeprefix(prefix)
        reference = reference.removeprefix(prefix)
    return float(SequenceMatcher(None, prediction, reference).ratio())


def grade(*, case: dict[str, Any], prediction: str) -> dict[str, Any]:
    metadata = case["metadata"]["mrcr"]
    filtered = prediction.lstrip(PREFIX_FILTER)
    score = native_grade(
        prediction=filtered,
        reference=case["answer"],
        prefix=metadata["random_string_to_prepend"],
    )
    return {
        "answer": filtered,
        "correct": None,
        "score": score,
        "details": {
            "mrcr": {
                "version": RESULT_KIND,
                "native_metric": MRCR_METRIC,
                "source_index": metadata["source_index"],
                "bin_index": metadata["bin_index"],
                "raw_input_tok_cnt": metadata["raw_input_tok_cnt"],
                "total_tok_cnt": metadata["total_tok_cnt"],
            }
        },
    }


def execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    response = context.call(messages=case["messages"])
    if response.get("output_complete") is not True:
        raise CallFailure("MRCR output is incomplete; no native grade was recorded")
    prediction = response.get("final")
    if not isinstance(prediction, str):
        raise CallFailure("MRCR transport returned no final answer")
    return grade(case=case, prediction=prediction)


def preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    options = manifest.get("benchmark_options", {}).get(BENCHMARK_ID, {})
    if options.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("MRCR plan must freeze the supported sequence-ratio protocol")
    metadata = case.get("metadata", {}).get("mrcr")
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != expected
        for key, expected in {
            "protocol_version": PROTOCOL_VERSION,
            "selection_version": NATIVE_SELECTION_VERSION,
            "dataset_id": NATIVE_DATASET_ID,
            "native_split": "train",
            "evalscope_version": NATIVE_EVALSCOPE_VERSION,
            "evalscope_revision": NATIVE_EVALSCOPE_REVISION,
        }.items()
    ):
        raise ValueError("MRCR case differs from the frozen native source contract")
    source_sha256 = metadata.get("source_sha256")
    revision = metadata.get("source_revision")
    if (
        not isinstance(source_sha256, str)
        or not re_full_sha256(source_sha256)
        or not isinstance(revision, str)
        or not revision
        or type(metadata.get("source_index")) is not int
        or metadata["source_index"] < 0
        or type(metadata.get("n_needles")) is not int
        or type(metadata.get("bin_index")) is not int
        or not 0 <= metadata["bin_index"] < len(MRCR_TOKEN_BINS)
        or type(metadata.get("raw_input_tok_cnt")) is not int
        or type(metadata.get("total_tok_cnt")) is not int
        or type(metadata.get("answer_tok_cnt")) is not int
        or metadata["raw_input_tok_cnt"] < 0
        or metadata["answer_tok_cnt"] < 0
        or metadata["total_tok_cnt"] < metadata["raw_input_tok_cnt"]
        or metadata.get("profile") not in {"smoke", "formal"}
        or metadata.get("n_needles")
        != (
            SMOKE_NEEDLE_COUNT
            if metadata.get("profile") == "smoke"
            else FORMAL_NEEDLE_COUNT
        )
        or metadata["total_tok_cnt"]
        != metadata["raw_input_tok_cnt"] + metadata["answer_tok_cnt"]
        or metadata["bin_index"] != _bin_index(metadata["total_tok_cnt"])
        or (
            metadata.get("profile") == "smoke"
            and not SMOKE_MIN_CONTEXT_TOKENS
            < metadata["raw_input_tok_cnt"]
            <= SMOKE_MAX_CONTEXT_TOKENS
        )
        or (
            metadata.get("profile") == "formal"
            and metadata["raw_input_tok_cnt"] > FORMAL_MAX_CONTEXT_TOKENS
        )
        or (
            metadata.get("random_string_to_prepend") is not None
            and not isinstance(metadata.get("random_string_to_prepend"), str)
        )
    ):
        raise ValueError("MRCR case has invalid source or native selection metadata")
    identity = (source_sha256, revision, metadata.get("profile"))
    prior = cache.setdefault("mrcr_source_identity", identity)
    if prior != identity:
        raise ValueError("An MRCR plan cannot mix source revisions or selections")
    if not isinstance(case.get("answer"), str) or not isinstance(
        case.get("messages"), list
    ):
        raise ValueError("MRCR cases require native messages and reference answers")
    for message in case["messages"]:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("MRCR cases contain an invalid native chat message")


def adapter() -> BenchmarkAdapter:
    return BenchmarkAdapter(
        id=BENCHMARK_ID,
        title="Router Flow native MRCR sequence-ratio diagnostic",
        kind="native-sequence-ratio",
        source_url=SOURCE_URL,
        version=PROTOCOL_VERSION,
        execute=execute,
        preflight=preflight,
        prepare=prepare,
        requires_answer=True,
        result_kind=RESULT_KIND,
    )
