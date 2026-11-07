from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .adapters import AdapterContext, BenchmarkAdapter
from .canonical import digest
from .dataset_io import read_small

PROTOCOL: str = "bench-reasoning-plain-v1"
ENVIRONMENT: str = "legacy-reasoning"
SOURCE_REVISION: str = "8485be9dadbaba727ad0ee897dad150f103a8e6b"
SOURCE_URL: str = (
    "https://github.com/vllm-project/semantic-router/blob/"
    + SOURCE_REVISION
    + "/bench/reasoning"
)
REQUIREMENTS: Path = Path(__file__).with_name("legacy-reasoning-requirements.txt")
MAX_EXPORT_BYTES: int = 64 * 1024 * 1024
CONFIGURATIONS: dict[str, tuple[str, str]] = {
    "arc": ("ARC", "multiple-choice"),
    "arc-easy": ("ARC-Easy", "multiple-choice"),
    "arc-challenge": ("ARC-Challenge", "multiple-choice"),
    "gpqa-main": ("GPQA-Main", "multiple-choice"),
    "gpqa-extended": ("GPQA-Extended", "multiple-choice"),
    "gpqa-diamond": ("GPQA-Diamond", "multiple-choice"),
    "gsm8k": ("GSM8K", "free-form"),
    "aqua-rat": ("AQUA-RAT", "multiple-choice"),
    "drop": ("DROP", "free-form"),
    "strategyqa": ("StrategyQA", "binary-choice"),
    "sciq": ("SciQ", "multiple-choice"),
    "openbookqa": ("OpenBookQA", "multiple-choice"),
    "truthfulqa": ("TruthfulQA", "multiple-choice"),
    "commonsenseqa": ("CommonsenseQA", "multiple-choice"),
    "hellaswag": ("HellaSwag", "multiple-choice"),
}
ALIASES: dict[str, str] = {"gpqa": "gpqa-main"}


def configuration(name: str) -> str:
    name = ALIASES.get(name.lower(), name.lower())
    if name not in CONFIGURATIONS:
        raise ValueError("Unsupported legacy plain reasoning configuration")
    return name


def case_identity(
    *, benchmark: str, source_sha256: str, selection: dict[str, Any], native_id: str
) -> str:
    return benchmark + "/" + digest([source_sha256, selection, native_id])


def worker_request(
    *, operation: str, payload: dict[str, Any], root: Path, interpreter: Path
) -> dict[str, Any]:
    env: dict[str, str] = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "LANG", "SSL_CERT_FILE")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        PYTHONUNBUFFERED="1",
    )
    with tempfile.TemporaryDirectory(prefix="sr-bench-legacy-") as temporary:
        request_path = Path(temporary) / "request.json"
        output_path = Path(temporary) / "result.json"
        request_path.write_text(json.dumps({**payload, "source_root": str(root)}))
        request_path.chmod(0o600)
        result = subprocess.run(
            [
                str(interpreter),
                "-m",
                "cli.sr_bench.legacy_reasoning_worker",
                operation,
                str(request_path),
                str(output_path),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                f"Legacy reasoning {operation} failed:\n{result.stderr[-8000:]}"
            )
        return json.loads(read_small(output_path, MAX_EXPORT_BYTES))


def inspect_runtime(cache: dict[str, Any] | None = None) -> dict[str, Any]:
    from .external import _preflight_harness  # noqa: PLC0415 - adapter registry cycle
    from .setup import harness_paths  # noqa: PLC0415 - registry cycle

    root, interpreter = harness_paths(ENVIRONMENT)
    key = ENVIRONMENT + ":" + digest([str(root), str(interpreter)])
    if cache is not None and key in cache:
        return cache[key]
    _preflight_harness(
        benchmark=ENVIRONMENT, config={"source_revision": SOURCE_REVISION}
    )
    result = worker_request(
        operation="inspect", payload={}, root=root, interpreter=interpreter
    )
    if cache is not None:
        cache[key] = result
    return result


def validate_case(case: dict[str, Any], config: dict[str, Any]) -> None:
    if (
        not isinstance(config, dict)
        or config.get("protocol") != PROTOCOL
        or config.get("source_revision") != SOURCE_REVISION
        or config.get("requirements_sha256")
        != hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    ):
        raise ValueError(
            "Legacy reasoning requires its pinned native protocol and environment"
        )
    name = configuration(case["benchmark"].removeprefix("legacy-"))
    metadata, question = case.get("metadata"), case.get("answer")
    if not isinstance(metadata, dict) or not isinstance(question, dict):
        raise ValueError("Legacy reasoning requires private native question data")
    if set(question) != {
        "question_id",
        "category",
        "question",
        "options",
        "correct_answer",
        "cot_content",
        "metadata",
    }:
        raise ValueError("Legacy reasoning requires the native Question fields")
    if (
        not all(
            isinstance(question[key], str) and question[key]
            for key in ("question_id", "category", "question")
        )
        or not isinstance(question["options"], list)
        or not all(isinstance(option, str) for option in question["options"])
        or type(question["correct_answer"]) not in {str, int}
        or (
            question["cot_content"] is not None
            and not isinstance(question["cot_content"], str)
        )
        or (
            question["metadata"] is not None
            and not isinstance(question["metadata"], dict)
        )
    ):
        raise ValueError("Legacy reasoning has invalid native question data")
    source, selection = config.get("source"), config.get("selection")
    if (
        not isinstance(source, dict)
        or set(source) != {"kind", "revision_verification", "files"}
        or source.get("kind") not in {"fixture", "local-import"}
        or source.get("revision_verification")
        != "declared-revision-with-content-digest"
        or not isinstance(source.get("files"), list)
        or not source["files"]
        or not isinstance(selection, dict)
        or set(selection)
        != {"dataset", "method", "seed", "categories", "samples_per_category"}
        or selection.get("dataset") != name
        or selection.get("method") != "native-loader"
    ):
        raise ValueError(
            "Legacy reasoning requires explicit local source and native selection"
        )
    for file in source["files"]:
        if (
            not isinstance(file, dict)
            or set(file)
            != {"dataset", "configuration", "split", "revision", "sha256", "records"}
            or not all(
                isinstance(file[key], str) and file[key]
                for key in ("dataset", "split", "revision")
            )
            or (
                file["configuration"] is not None
                and not isinstance(file["configuration"], str)
            )
            or not isinstance(file["sha256"], str)
            or len(file["sha256"]) != hashlib.sha256().digest_size * 2
            or any(character not in "0123456789abcdef" for character in file["sha256"])
            or type(file["records"]) is not int
            or file["records"] <= 0
        ):
            raise ValueError(
                "Legacy reasoning requires source partitions with content digests"
            )
    if (
        type(selection["seed"]) is not int
        or not 0 <= selection["seed"] < 2**32
        or (
            selection["samples_per_category"] is not None
            and (
                type(selection["samples_per_category"]) is not int
                or selection["samples_per_category"] <= 0
            )
        )
        or (
            selection["categories"] is not None
            and (
                not isinstance(selection["categories"], list)
                or not selection["categories"]
                or not all(
                    isinstance(item, str) and item for item in selection["categories"]
                )
            )
        )
    ):
        raise ValueError("Legacy reasoning has invalid native selection parameters")
    if set(metadata) != {
        "protocol",
        "native_dataset",
        "native_question_id",
        "native_question_sha256",
        "source_sha256",
        "prompt_sha256",
        "stratum",
    }:
        raise ValueError(
            "Legacy reasoning metadata must not carry native reference records"
        )
    source_sha256 = digest(source)
    if (
        metadata.get("native_dataset") != name
        or metadata.get("native_question_id") != question["question_id"]
        or metadata.get("native_question_sha256") != digest(question)
        or metadata.get("source_sha256") != source_sha256
        or metadata.get("prompt_sha256") != digest(case.get("messages"))
        or metadata.get("protocol") != PROTOCOL
        or case["id"]
        != case_identity(
            benchmark=case["benchmark"],
            source_sha256=source_sha256,
            selection=selection,
            native_id=question["question_id"],
        )
    ):
        raise ValueError(
            "Legacy reasoning case differs from its frozen native identity"
        )


def _preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    if manifest.get("profile") == "standard":
        raise ValueError("Legacy plain diagnostics do not establish a holdout")
    config = manifest.get("benchmark_options", {}).get(case["benchmark"], {})
    validate_case(case=case, config=config)
    inspect_runtime(cache=cache)


def _execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    from .external import _harness  # noqa: PLC0415 - adapter registry cycle

    return _harness(case=case, context=context)


ADAPTERS: tuple[BenchmarkAdapter, ...] = tuple(
    BenchmarkAdapter(
        id="legacy-" + name,
        title=title + " (legacy plain diagnostic)",
        kind=kind,
        source_url=SOURCE_URL,
        version=PROTOCOL,
        execute=_execute,
        preflight=_preflight,
        requires_answer=True,
    )
    for name, (title, kind) in CONFIGURATIONS.items()
)
