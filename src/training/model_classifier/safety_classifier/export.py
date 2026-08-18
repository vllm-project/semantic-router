"""Merge a trained safety adapter and validate the two artifact shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONTRACT_PATH, id2label, load_contract, task_contract

ADAPTER_REQUIRED_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "label_mapping.json",
}
MERGED_REQUIRED_FILES = {"config.json", "label_mapping.json"}
RELEASE_METADATA_FILES = (
    "data_manifest.json",
    "metrics.json",
    "reconstruction_contract.json",
    "training_manifest.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_model_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (
        path / "model.safetensors.index.json"
    ).is_file()


def validate_artifact_shape(path: str | Path, artifact_type: str) -> None:
    """Fail when an adapter is presented as merged, or the reverse."""
    artifact_path = Path(path)
    if artifact_type == "adapter":
        missing = sorted(
            filename
            for filename in ADAPTER_REQUIRED_FILES
            if not (artifact_path / filename).is_file()
        )
        if _has_model_weights(artifact_path):
            raise ValueError(
                "Adapter directory unexpectedly contains full model weights"
            )
    elif artifact_type == "merged":
        missing = sorted(
            filename
            for filename in MERGED_REQUIRED_FILES
            if not (artifact_path / filename).is_file()
        )
        if not _has_model_weights(artifact_path):
            missing.append("model.safetensors[.index.json]")
        if (artifact_path / "adapter_model.safetensors").exists():
            raise ValueError("Merged directory unexpectedly contains adapter weights")
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    if missing:
        raise FileNotFoundError(
            f"{artifact_type} artifact is missing: {', '.join(missing)}"
        )


def _release_file_inventory(path: Path) -> list[dict[str, Any]]:
    inventory = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        if relative.startswith(".cache/") or relative == "artifact_manifest.json":
            continue
        inventory.append(
            {
                "path": relative,
                "bytes": file_path.stat().st_size,
                "sha256": file_sha256(file_path),
            }
        )
    return inventory


def write_artifact_manifest(
    path: str | Path, artifact_type: str, task_name: str
) -> dict[str, Any]:
    artifact_path = Path(path)
    validate_artifact_shape(artifact_path, artifact_type)
    manifest = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "task": task_name,
        "files": _release_file_inventory(artifact_path),
    }
    (artifact_path / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _copy_release_metadata(run_root: Path, target: Path) -> None:
    for filename in RELEASE_METADATA_FILES:
        source = run_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing release metadata: {source}")
        shutil.copy2(source, target / filename)


def _representative_texts(task_name: str) -> list[str]:
    if task_name == "level1":
        return [
            "Please explain how photosynthesis works.",
            "Someone is threatening another person and needs immediate help.",
        ]
    return [
        "A person is threatening physical violence.",
        "A message exposes another person's private address.",
        "A user asks for unqualified medical advice.",
    ]


def _export_runtime(torch_module: Any, *, use_cpu: bool) -> tuple[Any, Any]:
    use_accelerator = not use_cpu and torch_module.cuda.is_available()
    device = torch_module.device("cuda:0" if use_accelerator else "cpu")
    # Merge the LoRA delta into FP32 weights even when an accelerator is
    # available. Direct BF16 merging quantizes the delta and can materially
    # change logits relative to the unmerged adapter.
    return torch_module.float32, device


def merge_adapter(
    run_root: str | Path,
    merged_dir: str | Path,
    task_name: str,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    use_cpu: bool = False,
) -> dict[str, Any]:
    """Merge LoRA weights, compare logits, and write release manifests."""
    try:
        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError("Export dependencies are not installed") from exc

    contract = load_contract(contract_path)
    task = task_contract(contract, task_name)
    run_path = Path(run_root).resolve()
    adapter_dir = run_path / "adapter"
    merged_path = Path(merged_dir).resolve()
    if merged_path == adapter_dir:
        raise ValueError("Adapter and merged output directories must be different")
    if merged_path.exists() and any(merged_path.iterdir()):
        raise FileExistsError(f"Merged output directory is not empty: {merged_path}")
    merged_path.mkdir(parents=True, exist_ok=True)

    _copy_release_metadata(run_path, adapter_dir)
    validate_artifact_shape(adapter_dir, "adapter")
    base = contract["base_model"]
    dtype, device = _export_runtime(torch, use_cpu=use_cpu)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base["id"],
        revision=base["revision"],
        num_labels=int(task["num_labels"]),
        id2label=id2label(task),
        label2id=task["label2id"],
        problem_type="single_label_classification",
        torch_dtype=dtype,
    )
    # ModernBERT enables reference torch.compile opportunistically. Export
    # changes the module graph during merge_and_unload, so keep both parity
    # forwards eager and deterministic.
    base_model.config.reference_compile = False
    adapter_model = PeftModel.from_pretrained(base_model, adapter_dir)
    adapter_model.to(device).eval()
    inputs = tokenizer(
        _representative_texts(task_name),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(contract["model"]["max_length"]),
    ).to(device)
    with torch.no_grad():
        adapter_logits = adapter_model(**inputs).logits.float().cpu()

    merged_model = adapter_model.merge_and_unload(safe_merge=True)
    merged_model.config.id2label = id2label(task)
    merged_model.config.label2id = task["label2id"]
    merged_model.config.problem_type = "single_label_classification"
    merged_model.eval()
    with torch.no_grad():
        merged_logits = merged_model(**inputs).logits.float().cpu()

    validation = contract["artifact_validation"]
    torch.testing.assert_close(
        adapter_logits,
        merged_logits,
        atol=float(validation["parity_atol"]),
        rtol=float(validation["parity_rtol"]),
    )
    max_abs_difference = float((adapter_logits - merged_logits).abs().max().item())
    prediction_match = bool(
        torch.equal(adapter_logits.argmax(dim=-1), merged_logits.argmax(dim=-1))
    )
    if not prediction_match:
        raise AssertionError("Adapter and merged predictions differ")

    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    shutil.copy2(adapter_dir / "label_mapping.json", merged_path / "label_mapping.json")
    _copy_release_metadata(run_path, merged_path)
    parity = {
        "schema_version": 1,
        "task": task_name,
        "dtype": str(dtype),
        "atol": float(validation["parity_atol"]),
        "rtol": float(validation["parity_rtol"]),
        "max_abs_logit_difference": max_abs_difference,
        "prediction_match": prediction_match,
        "fixtures": len(_representative_texts(task_name)),
    }
    for path in (adapter_dir, merged_path):
        (path / "parity.json").write_text(
            json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    validate_artifact_shape(merged_path, "merged")

    write_artifact_manifest(adapter_dir, "adapter", task_name)
    write_artifact_manifest(merged_path, "merged", task_name)
    return parity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("level1", "level2"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parity = merge_adapter(
        run_root=args.run_root,
        merged_dir=args.merged_dir,
        task_name=args.task,
        contract_path=args.contract,
        use_cpu=args.cpu,
    )
    print(json.dumps(parity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
