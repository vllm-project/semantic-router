"""Load and validate the safety-classifier reconstruction contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT_PATH = PACKAGE_ROOT / "configs" / "reconstruction-v1.json"


class ContractError(ValueError):
    """Raised when a reconstruction contract is incomplete or inconsistent."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ContractError(f"Missing {context}.{key}")
    return mapping[key]


def load_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load the versioned JSON contract and validate its cross-field invariants."""
    contract_path = Path(path)
    with contract_path.open(encoding="utf-8") as handle:
        contract = json.load(handle)

    if contract.get("contract_version") != 1:
        raise ContractError("Only reconstruction contract_version=1 is supported")

    base_model = _require(contract, "base_model", "contract")
    for key in ("id", "revision"):
        value = _require(base_model, key, "base_model")
        if not isinstance(value, str) or not value:
            raise ContractError(f"base_model.{key} must be a non-empty string")

    datasets = _require(contract, "datasets", "contract")
    for dataset_name in ("aegis", "synthetic"):
        dataset = _require(datasets, dataset_name, "datasets")
        for key in ("id", "revision", "files"):
            _require(dataset, key, f"datasets.{dataset_name}")

    model = _require(contract, "model", "contract")
    if model.get("max_length") != 512:
        raise ContractError("reconstruction-v1 fixes model.max_length at 512")
    lora = _require(model, "lora", "model")
    if lora.get("alpha") != 2 * lora.get("rank", 0):
        raise ContractError("LoRA alpha must remain exactly twice the rank")
    if sorted(lora.get("target_modules", [])) != sorted(
        ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
    ):
        raise ContractError("Unexpected ModernBERT LoRA target module set")

    tasks = _require(contract, "tasks", "contract")
    if set(tasks) != {"level1", "level2"}:
        raise ContractError("tasks must contain exactly level1 and level2")
    if tasks["level1"].get("num_labels") != 2:
        raise ContractError("level1 must contain two labels")
    if tasks["level2"].get("num_labels") != 9:
        raise ContractError("level2 must contain nine labels")
    for task_name, task in tasks.items():
        label2id = _require(task, "label2id", f"tasks.{task_name}")
        expected_ids = list(range(task["num_labels"]))
        if sorted(label2id.values()) != expected_ids:
            raise ContractError(f"tasks.{task_name}.label2id must be contiguous")

    return contract


def canonical_contract_bytes(contract: dict[str, Any]) -> bytes:
    """Serialize a contract deterministically for manifests and release checks."""
    return json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def contract_sha256(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_contract_bytes(contract)).hexdigest()


def task_contract(contract: dict[str, Any], task_name: str) -> dict[str, Any]:
    try:
        return contract["tasks"][task_name]
    except KeyError as exc:
        raise ContractError(f"Unknown task: {task_name}") from exc


def id2label(task: dict[str, Any]) -> dict[int, str]:
    return {label_id: label for label, label_id in task["label2id"].items()}


def distributed_batch_parameters(
    contract: dict[str, Any], world_size: int | None = None
) -> tuple[int, int]:
    """Return per-device batch size and gradient accumulation for global batch."""
    training = contract["training"]
    effective_world_size = world_size or int(os.environ.get("WORLD_SIZE", "1"))
    if effective_world_size < 1:
        raise ContractError("WORLD_SIZE must be positive")
    per_device = int(training["per_device_train_batch_size"])
    global_batch = int(training["global_train_batch_size"])
    denominator = effective_world_size * per_device
    if global_batch % denominator:
        raise ContractError(
            "global_train_batch_size must be divisible by "
            "WORLD_SIZE * per_device_train_batch_size"
        )
    return per_device, global_batch // denominator
