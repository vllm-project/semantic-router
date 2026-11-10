"""Pinned public JevBench tasks, upstream rendering/scoring and fixed schedules."""

from __future__ import annotations

import copy
import importlib
import random
import subprocess
import sys
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

from .artifacts import file_digest
from .metrics import evaluate_response

COMMIT = "b6b8fff7e345b98c060ad26c13308860ddc67004"
THRESHOLD = 0.6059704079536342
ARMS = ("direct_kai", "direct_vega", "auto")
PUBLIC_ITEM_COUNT = 231
PUBLIC_GROUP_COUNT = 195
REVISIONS = {
    "kai": "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764",
    "vega": "7aec49ae11a18741706da549ab626b9052795fe7",
}
DATA_SHA256 = {
    "easy": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "original": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "hard": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}


def load_suite(checkout: Path, commit: str = COMMIT):
    """Import only the verified immutable upstream source; never download it."""
    root = checkout.resolve()
    if commit != COMMIT:
        raise ValueError("unsupported suite commit")
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != COMMIT:
        raise ValueError("JevBench checkout must use the pinned commit")
    pins = {}
    for path in (root / "jevbench").rglob("*.py"):
        relative = str(path.relative_to(root))
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{COMMIT}:{relative}"]
        )
        if path.read_bytes() != original:
            raise ValueError("upstream Python source differs from pinned commit")
        pins[relative] = file_digest(path)
    for name, expected in DATA_SHA256.items():
        relative = f"datasets/public/{name}.jsonl"
        if file_digest(root / relative) != expected:
            raise ValueError("public task bytes differ from pinned suite")
        pins[relative] = expected
    loaded = sys.modules.get("jevbench")
    if loaded and Path(loaded.__file__).resolve().parent != root / "jevbench":
        raise ValueError("another JevBench checkout is already imported")
    sys.path.insert(0, str(root))
    task_module = importlib.import_module("jevbench.tasks")
    adapter_module = importlib.import_module("jevbench.adapters.typesafe")
    scoring_module = importlib.import_module("jevbench.scoring")
    tasks = {}
    for name in DATA_SHA256:
        for task in task_module.load_jsonl(str(root / f"datasets/public/{name}.jsonl")):
            if task.id in tasks:
                raise ValueError("duplicate public task id")
            tasks[task.id] = task
    if (
        len(tasks) != PUBLIC_ITEM_COUNT
        or len({task.group or task.id for task in tasks.values()}) != PUBLIC_GROUP_COUNT
    ):
        raise ValueError("public suite item/source-group counts changed")
    return SimpleNamespace(
        tasks=tasks,
        adapter=adapter_module.TypeSafeAdapter,
        score=scoring_module.score_task,
        source={"commit": COMMIT, "files": pins},
    )


def request_for(suite, task, model: str) -> dict:
    # The existing single-request transport serializes canonical JSON, as in
    # the original controlled collection. The unchanged upstream adapter owns
    # state/rubric/criteria semantics; only declared evidence options are added.
    payload = copy.deepcopy(suite.adapter(model=model, key_env="").build_request(task))
    for question in payload["questions"].values():
        question["require_full_input"] = True
    payload["options"] = {"return_meta": True}
    return payload


def schedule_for(tasks: dict, passes: int, seed: int) -> list[dict]:
    schedule = []
    for index in range(passes):
        ids = sorted(tasks)
        random.Random(seed + index).shuffle(ids)
        for position, task_id in enumerate(ids):
            rotation = (index + position) % len(ARMS)
            arms = ARMS[rotation:] + ARMS[:rotation]
            for arm_position, arm in enumerate(arms):
                schedule.append(
                    {
                        "task_id": task_id,
                        "pass": index,
                        "position": len(schedule),
                        "arm": arm,
                        "arm_position": arm_position,
                    }
                )
    return schedule


def validate_schedule(schedule: list[dict], tasks: dict, passes: int) -> None:
    expected = {
        (task_id, index, arm)
        for task_id in tasks
        for index in range(passes)
        for arm in ARMS
    }
    seen = set()
    slots = set()
    prior_pass = -1
    for sequence, row in enumerate(schedule):
        key = (row["task_id"], row["pass"], row["arm"])
        slot = (row["pass"], row["position"], row["arm_position"])
        expected_slot = (
            sequence // (len(tasks) * len(ARMS)),
            sequence,
            sequence % len(ARMS),
        )
        if (
            key not in expected
            or key in seen
            or slot in slots
            or not 0 <= row["position"] < len(tasks) * len(ARMS) * passes
            or not 0 <= row["arm_position"] < len(ARMS)
            or row["pass"] < prior_pass
            or slot != expected_slot
            or row["task_id"] != schedule[sequence - sequence % len(ARMS)]["task_id"]
        ):
            raise ValueError(
                "schedule must contain each task/arm once in sequential passes"
            )
        seen.add(key)
        slots.add(slot)
        prior_pass = row["pass"]
    if seen != expected:
        raise ValueError("schedule is incomplete")


def score_response(suite, task, payload: dict, status: int, raw: dict) -> dict:
    label = str(task.expected)
    if task.question["type"] == "noul":
        label = "true" if task.expected == "yes" else "false"
    native = evaluate_response(
        {"request": payload, "labels": {"decision": {"label": label}}}, raw
    )
    question = native["questions"]["decision"]
    valid = (
        status == HTTPStatus.OK
        and not raw.get("error")
        and native["valid"]
        and question["top_probability"] is not None
    )
    if not valid:
        return {"valid": False, "correct": False, "top_probability": None}
    answer = raw["answers"]["decision"]
    # This is exactly the upstream TypeSafe adapter's native probability map;
    # its scorer, including ties and Score's modal level, remains unchanged.
    probabilities = (
        {"yes": answer["noul"], "no": 1 - answer["noul"]}
        if task.question["type"] == "noul"
        else answer["probabilities"]
    )
    scored = suite.score(probabilities, task)
    return {
        "valid": bool(scored["valid"]),
        "correct": bool(scored["valid"] and scored["correct"]),
        "top_probability": question["top_probability"],
    }
