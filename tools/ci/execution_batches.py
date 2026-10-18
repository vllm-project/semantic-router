"""Physical CI workers, independent of verification identity and presentation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

MAX_CONTRACT_MINUTES = 330
MAX_BATCH_MINUTES = 270
MAX_BATCH_CONTRACTS = 3
MAX_COMPONENT_CONTRACTS = 4

IMAGE_PRODUCERS = {
    "image-router": ("vllm-sr",),
    "image-dashboard": ("dashboard",),
    "image-operator": ("operator", "operator-bundle"),
    "image-fixtures": ("provider-mocker", "model-runtime"),
    "image-distribution": (
        "vllm-sr-cuda",
        "vllm-sr-rocm",
    ),
}
EXECUTOR_JOBS = (
    "quality",
    "generated",
    "security",
    "core",
    "storage",
    "dashboard",
    "operator",
    "local",
    "recipes",
    "platform",
    "performance",
    "package",
    "tools",
    "e2e-router",
    "e2e-fixtures",
    "e2e-dashboard",
)
ALL_DISPATCH_JOBS = ("plan", *IMAGE_PRODUCERS, *EXECUTOR_JOBS)
LANE_IMAGES = frozenset({"vllm-sr", "provider-mocker", "dashboard"})


def content_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dispatch_job(record: dict) -> str:
    executor = record["executor"]
    if executor == "e2e":
        images = set(record["images"])
        if images == {"vllm-sr"}:
            return "e2e-router"
        if images in (
            {"vllm-sr", "provider-mocker"},
            {"vllm-sr", "provider-mocker", "model-runtime"},
        ):
            return "e2e-fixtures"
        if images == {"vllm-sr", "dashboard"}:
            return "e2e-dashboard"
        raise ValueError(f"undeclared E2E image dependency lane: {sorted(images)}")
    return executor


def image_producers(images: list[str]) -> dict[str, list[str]]:
    result = {
        job: sorted(set(images) & set(owned)) for job, owned in IMAGE_PRODUCERS.items()
    }
    if set(images) != {image for selected in result.values() for image in selected}:
        raise ValueError("image has no declared producer")
    return result


def expected_dispatch_jobs(plan: dict) -> list[str]:
    return sorted(
        {
            "plan",
            *(dispatch_job(record) for record in plan["verifications"]),
            *(job for job, images in image_producers(plan["images"]).items() if images),
        }
    )


def _compatibility(record: dict) -> dict:
    common = {
        key: record[key]
        for key in ("executor", "runner", "runtime", "device", "platform")
    }
    common.update(images=sorted(record["images"]), dispatch_job=dispatch_job(record))
    if record["executor"] == "platform":
        common["platform_id"] = record["platform_id"]
    else:
        common["resource_class"] = record.get("resource_class", "standard")
    return common


def _batch(records: list[dict], shard: int) -> dict:
    common = _compatibility(records[0])
    executor = common["executor"]
    identity = content_digest([record["id"] for record in records])[:12]
    architecture = common["platform"].split("/")[-1]
    runtime = {"model-runtime": "Model Runtime"}.get(
        common["runtime"], common["runtime"]
    )
    label = f"{runtime} / {common['device'].upper()} / {architecture}"
    if executor == "e2e":
        resource = "Large" if common["resource_class"] == "model" else "Standard"
        label += f" / {resource} {shard}"
        # A fixture beyond the lane's own images forms a separate compatibility
        # group whose shards restart at 1 in the same Actions matrix.
        extra = [image for image in common["images"] if image not in LANE_IMAGES]
        if extra:
            label += " / " + " + ".join(extra)
    if executor == "platform":
        # Distinguish compatible groups with the same runtime label.
        label += f" / {identity[:6]}"
    minutes = sum(
        record.get("timeout_minutes", 90 if executor == "e2e" else 120)
        for record in records
    )
    return {
        **common,
        "id": f"{executor}-{identity}",
        "display_name": label,
        "shard": shard,
        "timeout_minutes": minutes,
        "job_timeout_minutes": minutes + 30,
        "verifications": records,
    }


def execution_batches(records: list[dict], executor: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in sorted(records, key=lambda row: row["id"]):
        if record["executor"] == executor:
            groups[json.dumps(_compatibility(record), sort_keys=True)].append(record)
    result = []
    for key in sorted(groups):
        pending, minutes, shard = [], 0, 1
        for record in groups[key]:
            budget = record.get("timeout_minutes", 90 if executor == "e2e" else 120)
            if budget <= 0 or budget > MAX_CONTRACT_MINUTES:
                raise ValueError(f"invalid worker time budget for {record['id']}")
            limit = (
                1 if record.get("resource_class") == "model" else MAX_BATCH_CONTRACTS
            )
            if pending and (
                len(pending) >= limit or minutes + budget > MAX_BATCH_MINUTES
            ):
                result.append(_batch(pending, shard))
                pending, minutes, shard = [], 0, shard + 1
            pending.append(record)
            minutes += budget
        if pending:
            result.append(_batch(pending, shard))
    return result


def _component_group(record: dict) -> tuple[str, str]:
    # The runtime installs a separate set of engine extras. Keep that dependency
    # environment apart from training/export tools sharing the lightweight worker.
    environment = "runtime" if record["target"] == "model-runtime-test" else "tools"
    return record["worker"], environment


def _component_batch(records: list[dict], workers: dict, shard: int) -> dict:
    worker = records[0]["worker"]
    spec = workers[worker]
    identity = content_digest([record["id"] for record in records])[:12]
    single = len(records) == 1
    return {
        **spec,
        "id": records[0]["id"] if single else f"components-{worker}-{identity}",
        "worker": worker,
        "shard": shard,
        "display_name": (
            records[0]["display_name"]
            if single
            else f"{spec['display_name']} / Batch {shard}"
        ),
        "job_timeout_minutes": 15 + 45 * len(records),
        "verifications": records,
    }


def component_batches(records: list[dict], workers: dict) -> list[dict]:
    """Share setup across bounded, compatible component contracts."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in sorted(records, key=lambda row: row["id"]):
        if record["executor"] == "tools":
            groups[_component_group(record)].append(record)
    result = []
    for key in sorted(groups):
        selected = groups[key]
        for index in range(0, len(selected), MAX_COMPONENT_CONTRACTS):
            result.append(
                _component_batch(
                    selected[index : index + MAX_COMPONENT_CONTRACTS],
                    workers,
                    index // MAX_COMPONENT_CONTRACTS + 1,
                )
            )
    return result


def validate_component_batch(batch: dict, workers: dict) -> None:
    records = batch.get("verifications", [])
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("component worker requires unique verification contracts")
    expected = component_batches(records, workers)
    shard = batch.get("shard")
    if (
        len(expected) != 1
        or not isinstance(shard, int)
        or shard <= 0
        or batch != _component_batch(expected[0]["verifications"], workers, shard)
    ):
        raise ValueError("component worker contains incompatible or altered metadata")


def e2e_batches(records: list[dict]) -> list[dict]:
    return execution_batches(records, "e2e")


def platform_batches(records: list[dict]) -> list[dict]:
    return execution_batches(records, "platform")


def validate_execution_batch(batch: dict, executor: str) -> None:
    records = batch.get("verifications", [])
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError("worker requires a nonempty unique verification inventory")
    for record in records:
        if record.get("executor") != executor:
            raise ValueError("worker received a verification for another executor")
        if record.get("contract_sha256") != content_digest(
            {key: value for key, value in record.items() if key != "contract_sha256"}
        ):
            raise ValueError("worker verification contract digest differs")
        if record.get("dispatch_job") != dispatch_job(record):
            raise ValueError("worker verification dispatch identity differs")
    expected = execution_batches(records, executor)
    if len(expected) != 1 or batch != _batch(
        expected[0]["verifications"], batch.get("shard", 0)
    ):
        raise ValueError("worker contains incompatible or altered batch metadata")
    if not isinstance(batch["shard"], int) or batch["shard"] <= 0:
        raise ValueError("worker shard must be a positive integer")
