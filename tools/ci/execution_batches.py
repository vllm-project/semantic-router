"""Physical CI workers, independent of verification identity and presentation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from execution_targets import TARGET_FIELDS, target_errors
from verification_catalog import load_catalog

MAX_JOB_MINUTES = 360

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
    "operator-source",
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
    if errors := target_errors(record):
        raise ValueError("; ".join(errors))
    common = {
        key: record[key]
        for key in (
            "executor",
            "runtime",
            "device",
            "setup_environment",
            "batch_policy",
            *TARGET_FIELDS,
        )
    }
    common.update(images=sorted(record["images"]), dispatch_job=dispatch_job(record))
    if record["executor"] == "platform":
        common["platform_id"] = record["platform_id"]
    else:
        common["resource_class"] = record.get("resource_class", "standard")
    return common


def batch_policy(record: dict) -> dict:
    policy = load_catalog()["batching"]["policies"].get(record.get("batch_policy"))
    if not isinstance(policy, dict) or any(
        type(policy.get(key)) is not int or policy[key] <= 0
        for key in ("max_contracts", "max_minutes", "overhead_minutes")
    ):
        raise ValueError(f"invalid batch policy for {record['id']}")
    if policy["max_minutes"] + policy["overhead_minutes"] > MAX_JOB_MINUTES:
        raise ValueError("batch policy exceeds the supported worker deadline")
    return policy


def contract_minutes(record: dict) -> int:
    policy = batch_policy(record)
    budget = record.get("timeout_minutes")
    if (
        type(budget) is not int
        or budget <= 0
        or budget + policy["overhead_minutes"] > MAX_JOB_MINUTES
    ):
        raise ValueError(f"invalid worker time budget for {record['id']}")
    return budget


def _pack(records: list[dict]) -> list[list[dict]]:
    """Policy controls packing only; each contract keeps its own deadline."""
    policy = batch_policy(records[0])
    result, pending, minutes = [], [], 0
    for record in records:
        budget = contract_minutes(record)
        if pending and (
            len(pending) >= policy["max_contracts"]
            or minutes + budget > policy["max_minutes"]
        ):
            result.append(pending)
            pending, minutes = [], 0
        pending.append(record)
        minutes += budget
    if pending:
        result.append(pending)
    return result


def _batch(records: list[dict], shard: int) -> dict:
    common = _compatibility(records[0])
    executor = common["executor"]
    identity = content_digest([common, [record["id"] for record in records]])[:12]
    architecture = common["platform"].split("/")[-1]
    runtime = {"model-runtime": "Model Runtime"}.get(
        common["runtime"], common["runtime"]
    )
    label = f"{runtime} / {common['device'].upper()} / {architecture}"
    if executor == "e2e":
        resource = "Large" if common["resource_class"] == "model" else "Standard"
        label += f" / {resource} {shard} / {identity[:6]}"
        # A fixture beyond the lane's own images forms a separate compatibility
        # group whose shards restart at 1 in the same Actions matrix.
        extra = [image for image in common["images"] if image not in LANE_IMAGES]
        if extra:
            label += " / " + " + ".join(extra)
    if executor == "platform":
        # Distinguish compatible groups with the same runtime label.
        label += f" / {identity[:6]}"
    minutes = sum(contract_minutes(record) for record in records)
    return {
        **common,
        "id": f"{executor}-{identity}",
        "display_name": label,
        "shard": shard,
        "timeout_minutes": minutes,
        "job_timeout_minutes": minutes + batch_policy(records[0])["overhead_minutes"],
        "verifications": records,
    }


def execution_batches(records: list[dict], executor: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in sorted(records, key=lambda row: row["id"]):
        if record["executor"] == executor:
            groups[json.dumps(_compatibility(record), sort_keys=True)].append(record)
    result = []
    for key in sorted(groups):
        result.extend(
            _batch(selected, shard)
            for shard, selected in enumerate(_pack(groups[key]), 1)
        )
    return result


def _component_batch(records: list[dict], workers: dict, shard: int) -> dict:
    worker = records[0]["worker"]
    spec = workers[worker]
    common = _compatibility(records[0])
    identity = content_digest([common, [record["id"] for record in records]])[:12]
    single = len(records) == 1
    return {
        **spec,
        **common,
        "id": records[0]["id"] if single else f"components-{worker}-{identity}",
        "worker": worker,
        "shard": shard,
        "display_name": (
            records[0]["display_name"]
            if single
            else f"{spec['display_name']} / Batch {shard} / {identity[:6]}"
        ),
        "job_timeout_minutes": sum(contract_minutes(record) for record in records)
        + batch_policy(records[0])["overhead_minutes"],
        "verifications": records,
    }


def component_batches(records: list[dict], workers: dict) -> list[dict]:
    """Share setup across bounded, compatible component contracts."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in sorted(records, key=lambda row: row["id"]):
        if record["executor"] == "tools":
            common = {**_compatibility(record), "worker": record["worker"]}
            groups[json.dumps(common, sort_keys=True)].append(record)
    result = []
    for key in sorted(groups):
        for shard, selected in enumerate(_pack(groups[key]), 1):
            result.append(_component_batch(selected, workers, shard))
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
