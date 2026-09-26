"""Physical CI workers, independent of verification identity and presentation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

MAX_CONTRACT_MINUTES = 330

IMAGE_PRODUCERS = {
    "image-router": ("extproc",),
    "image-local": ("vllm-sr",),
    "image-dashboard": ("dashboard",),
    "image-operator": ("operator", "operator-bundle"),
    "image-fixtures": ("provider-mocker",),
    "image-distribution": (
        "decision-runtime-cpu",
        "extproc-rocm",
        "vllm-sr-cuda",
        "vllm-sr-rocm",
        "vllm-sr-sim",
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
    "native-shared",
    "native-independent",
    "performance",
    "package",
    "tools",
    "e2e-router",
    "e2e-fixtures",
    "e2e-dashboard",
)
ALL_DISPATCH_JOBS = ("plan", *IMAGE_PRODUCERS, "native-build", *EXECUTOR_JOBS)


def content_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dispatch_job(record: dict) -> str:
    executor = record["executor"]
    if executor == "native":
        return "native-shared" if record["native"] else "native-independent"
    if executor == "e2e":
        images = set(record["images"])
        if images == {"extproc"}:
            return "e2e-router"
        if images == {"extproc", "provider-mocker"}:
            return "e2e-fixtures"
        if images == {"extproc", "dashboard"}:
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
            *(["native-build"] if plan["native"] else []),
        }
    )


def _compatibility(record: dict) -> dict:
    common = {
        key: record[key]
        for key in ("executor", "runner", "runtime", "device", "platform", "native")
    }
    common.update(images=sorted(record["images"]), dispatch_job=dispatch_job(record))
    if record["executor"] == "native":
        common.update(
            platform_id=record["platform_id"], execution=record.get("execution", {})
        )
    else:
        common["resource_class"] = record.get("resource_class", "standard")
    return common


def _batch(records: list[dict], shard: int) -> dict:
    common = _compatibility(records[0])
    executor = common["executor"]
    identity = content_digest([record["id"] for record in records])[:12]
    architecture = common["platform"].split("/")[-1]
    runtime = {"ort": "ORT", "candle": "Candle", "openvino": "OpenVINO"}.get(
        common["runtime"], common["runtime"]
    )
    label = f"{runtime} / {common['device'].upper()} / {architecture}"
    if executor == "e2e":
        resource = "Large" if common["resource_class"] == "model" else "Standard"
        label += f" / {resource} {shard}"
    elif common.get("execution"):
        label += " / QEMU"
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
        rows = groups[key]
        limit = (
            2 if executor == "native" or rows[0].get("resource_class") == "model" else 3
        )
        selected: list[dict] = []
        minutes = 0
        shard = 1
        for record in rows:
            budget = record.get("timeout_minutes", 90 if executor == "e2e" else 120)
            if budget <= 0 or budget > MAX_CONTRACT_MINUTES:
                raise ValueError(f"invalid worker time budget for {record['id']}")
            if selected and (
                len(selected) >= limit or minutes + budget > MAX_CONTRACT_MINUTES
            ):
                result.append(_batch(selected, shard))
                selected, minutes, shard = [], 0, shard + 1
            selected.append(record)
            minutes += budget
        if selected:
            result.append(_batch(selected, shard))
    return result


def e2e_batches(records: list[dict]) -> list[dict]:
    return execution_batches(records, "e2e")


def native_batches(records: list[dict]) -> list[dict]:
    return execution_batches(records, "native")


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
