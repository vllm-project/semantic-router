from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cli.routing_preview import case_request_fields

from . import agent_task_kernel
from .adapters import AdapterContext, BenchmarkAdapter
from .agent_task_kernel import TaskSpec, build_messages, score_answer, task_specs
from .canonical import canonical
from .contracts import SESSION_AWARE
from .dataset_io import MAX_ROWS
from .grading import TOKEN_COVERAGE_RESULT
from .sources import _write_dataset
from .task_identity import native_task_identity, source_identity
from .transport import CallFailure

BENCHMARK_ID: str = "agent-task-chat-v1"
GRADER_VERSION: str = "agent-task-native-substrings-v1"
SELECTION_VERSION: str = "agent-task-native-order-v1"
SOURCE_REVISION: str = "5b0634caa2631540e22af36fb122b2d29d329225"
SOURCE_URL: str = (
    "https://github.com/vllm-project/semantic-router/blob/"
    + SOURCE_REVISION
    + "/bench/agent_task_live_benchmark.py"
)
SUITES: tuple[str, ...] = ("smoke", "long-horizon", "all")


def _native_task(case: dict[str, Any]) -> TaskSpec:
    metadata = case.get("metadata")
    record = metadata.get("scripted_task") if isinstance(metadata, dict) else None
    if not isinstance(record, dict):
        raise ValueError("Agent tasks require the frozen native script")
    task = next(
        (task for task in task_specs("all") if task.name == record.get("name")), None
    )
    if task is None or canonical(record) != canonical(asdict(task)):
        raise ValueError("Agent task differs from its native task definition")
    return task


def preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    if manifest["profile"] == "standard":
        raise ValueError("Agent task diagnostics do not establish a Standard holdout")
    task = _native_task(case)
    repetition = case["metadata"].get("task_repetition")
    if type(repetition) is not int or not 0 <= repetition < MAX_ROWS:
        raise ValueError("Agent task repetition must be a bounded nonnegative integer")
    if (
        case_request_fields(case)
        or case["messages"] != build_messages(task=task, turn_index=0)
        or case.get("answer") != list(task.turns[-1].expected_terms)
        or any(turn.expected_terms for turn in task.turns[:-1])
        or not task.turns[-1].expected_terms
    ):
        raise ValueError(
            "Agent task messages or expected terms differ from the native script"
        )
    if len(task.turns) > manifest["limits"]["max_calls_per_case"]:
        raise ValueError("Agent task turns exceed the frozen call budget")
    selected = {
        cell["target_id"]
        for cell in manifest.get("execution_cells", [])
        if cell["case_id"] == case["id"]
    }
    for target in manifest["targets"]:
        if "execution_cells" in manifest and target["id"] not in selected:
            continue
        if target.get("session_mode") != SESSION_AWARE:
            raise ValueError("Agent tasks require session_aware subject targets")
    options = manifest.setdefault("benchmark_options", {})
    if not isinstance(options, dict):
        raise ValueError("benchmark_options must be an object")
    config = options.setdefault(BENCHMARK_ID, {})
    if not isinstance(config, dict) or set(config) - {"grader_version"}:
        raise ValueError("Agent tasks support Chat full-history execution only")
    if config.get("grader_version") != GRADER_VERSION:
        raise ValueError(f"Agent tasks require grader_version={GRADER_VERSION}")


def execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    task = _native_task(case)
    final = ""
    for index in range(len(task.turns)):
        response = context.call(messages=build_messages(task=task, turn_index=index))
        if not response.get("output_complete", True):
            raise CallFailure("Agent task output limit reached; grading is incomplete")
        final = response["final"]
    terms = task.turns[-1].expected_terms
    score, missing = score_answer(content=final, expected_terms=terms)
    return {
        "answer": final,
        "correct": None,
        "score": score,
        "details": {
            "grader_version": GRADER_VERSION,
            "token_coverage": {
                "expected_terms": list(terms),
                "matched_terms": [term for term in terms if term not in missing],
                "missing_terms": missing,
                "full_credit": score == 1.0,
            },
        },
    }


def adapter() -> BenchmarkAdapter:
    return BenchmarkAdapter(
        id=BENCHMARK_ID,
        title="Scripted agent-task Chat diagnostic",
        kind="agent",
        source_url=SOURCE_URL,
        version=GRADER_VERSION,
        execute=execute,
        preflight=preflight,
        requires_answer=True,
        result_kind=TOKEN_COVERAGE_RESULT,
    )


def prepare(
    *,
    store: Path,
    suite: str = "smoke",
    task_repetitions: int = 1,
    profile: str = "quick",
) -> dict[str, Any]:
    if suite not in SUITES:
        raise ValueError("Agent tasks require smoke, long-horizon or all")
    if profile not in {"smoke", "quick"}:
        raise ValueError("Agent task preparation supports smoke/quick diagnostics only")
    tasks = task_specs(suite=suite)
    if type(
        task_repetitions
    ) is not int or not 1 <= task_repetitions <= MAX_ROWS // len(tasks):
        raise ValueError(
            "Agent task repetitions must be positive and fit the case limit"
        )
    source = {
        "url": SOURCE_URL,
        "revision": SOURCE_REVISION,
        "partition": "scripted-tasks",
        "files": [
            {
                "name": "agent_task_kernel.py",
                "sha256": hashlib.sha256(
                    Path(agent_task_kernel.__file__).read_bytes()
                ).hexdigest(),
            }
        ],
        "normalizer": "agent-task-bundled-script-v1",
        "revision_verification": "declared-native-revision-with-bundled-kernel-digest",
        "selection": {"suite": suite, "task_repetitions": task_repetitions},
    }
    task_source = source_identity(source=source, partition=source["partition"])

    def cases() -> Iterator[dict[str, Any]]:
        for repetition in range(task_repetitions):
            for task in tasks:
                yield {
                    "id": f"{BENCHMARK_ID}/r{repetition:02d}/{task.name}",
                    "benchmark": BENCHMARK_ID,
                    "messages": build_messages(task=task, turn_index=0),
                    "answer": list(task.turns[-1].expected_terms),
                    "metadata": {
                        "split": "dev",
                        "stratum": task.suite,
                        "source_revision": SOURCE_REVISION,
                        "task_repetition": repetition,
                        "scripted_task": asdict(task),
                        "task_identity": native_task_identity(
                            benchmark=BENCHMARK_ID,
                            row={"id": task.name},
                            source=task_source,
                        ),
                    },
                }

    return _write_dataset(
        root=store,
        cases=cases(),
        profile=profile,
        seed=0,
        sources={BENCHMARK_ID: source},
        custom=True,
        selection=SELECTION_VERSION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a scripted agent-task diagnostic"
    )
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--suite", choices=SUITES, default="smoke")
    parser.add_argument("--task-repetitions", type=int, default=1)
    parser.add_argument("--profile", choices=("smoke", "quick"), default="quick")
    args = parser.parse_args()
    try:
        prepared = prepare(
            store=args.store,
            suite=args.suite,
            task_repetitions=args.task_repetitions,
            profile=args.profile,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(prepared, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
