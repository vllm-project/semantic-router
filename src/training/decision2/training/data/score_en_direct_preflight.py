"""Gold-free native BF16 zero-step source gate for the direct-LoRA pilot.

The reference and both trainable-adapter starts use this one inference path.
Only a sealed PASS receipt permits the separate matched optimizer pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from training.data.score_en_parity import _probabilities
from training.model.data import canonical, file_sha256
from training.model.decision_model import DecisionModel, collate, encode
from training.model.infer import (
    checkpoint_fingerprint,
    load_prompts,
    prompt_input_sha256,
    run_prompts,
    write_output,
)
from training.model.train import load_direct_lora_start

SOURCE_SHA = "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
ROSTER_SHA = "193404fb2ed3905cbb9e34379400a2f33a40d86aaae971940454c6fe71163bc5"
ARM_SHA = {
    "A": "6a6ef7d3f2eac2a63cdd61cd806275e67c0aa772e78b45bf0953200f2a776235",
    "B": "1c705c9a8271ce2e526b6bc91affe18d463a52bb86ce99b6b1bcb5007d467b41",
}
MAX_LENGTH = 1024
TOLERANCE = 1e-4


def _source_code() -> dict[str, str]:
    root = Path(__file__).parents[1]
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in (
            Path(__file__),
            root / "model" / "train.py",
            root / "model" / "infer.py",
            root / "model" / "decision_model.py",
            root / "model" / "lora.py",
        )
    }


def _roster(roster_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    path = roster_dir / "prompts.jsonl"
    manifest = json.loads((roster_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        file_sha256(path) != ROSTER_SHA
        or manifest.get("roster_sha256") != ROSTER_SHA
        or manifest.get("roster_rows") != 32
    ):
        raise ValueError("Frozen 32-row parity roster differs")
    return load_prompts(path), manifest, path


def predict(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if (
        args.output.exists()
        or args.output.with_name(args.output.name + ".manifest.json").exists()
    ):
        raise FileExistsError("Preflight output already exists")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 preflight needs one CUDA/ROCm GPU")
    rows, roster_manifest, roster_path = _roster(args.roster_dir)
    identity = checkpoint_fingerprint(args.checkpoint, args.source_path)
    if identity["model_sha256"] != SOURCE_SHA:
        raise ValueError("Original adapter-plus-base source identity changed")
    if (
        args.container_image_id
        != "sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1"
    ):
        raise ValueError("Container image ID differs from frozen native environment")
    if args.mode == "trainable":
        direct_args = argparse.Namespace(
            model_path=str(args.checkpoint),
            source_path=str(args.source_path),
            initial_model_sha256=SOURCE_SHA,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            head_dim=256,
        )
        model, tokenizer, loaded_identity = load_direct_lora_start(direct_args)
        if loaded_identity != identity:
            raise ValueError("Trainable source loader identity differs")
    else:
        model, tokenizer = DecisionModel.from_checkpoint(
            args.checkpoint, source_path=args.source_path, trainable_adapter=False
        )
    model = model.float().to("cuda:0").eval()
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_id is None:
        raise ValueError("Tokenizer has no pad/EOS ID")

    def forward(encoded: list[dict[str, Any]]) -> list[list[float]]:
        batch = {
            name: value.to("cuda:0") if torch.is_tensor(value) else value
            for name, value in collate(encoded, pad_id).items()
        }
        torch.cuda.synchronize()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = model(**batch)
        torch.cuda.synchronize()
        return [
            values[: len(item["keys"])].float().cpu().tolist()
            for values, item in zip(logits, encoded)
        ]

    from training.model.infer import question_to_row

    encodings = [
        encode(
            question_to_row(row, "decision", row["questions"]["decision"]),
            tokenizer,
            MAX_LENGTH,
        )
        for row in rows
    ]
    if [item["prompt_sha256"] for item in encodings] != roster_manifest[
        "input_sha256"
    ] or [item["token_ids_sha256"] for item in encodings] != roster_manifest[
        "token_ids_sha256"
    ]:
        raise ValueError("Native tokenizer/prompt encoding changed")
    code_files = _source_code()
    code_sha = hashlib.sha256(canonical(code_files).encode()).hexdigest()
    predictions, counts = run_prompts(
        rows,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        temperature=1.0,
        encode_fn=encode,
        predict_fn=forward,
        model_sha256=SOURCE_SHA,
        adapter_sha256=code_sha,
    )
    if counts != {
        "items": 32,
        "questions": 32,
        "valid_questions": 32,
        "invalid_questions": 0,
        "over_budget_questions": 0,
        "truncated_questions": 0,
    }:
        raise ValueError("Native zero-step preflight has invalid responses")
    hardware = torch.cuda.get_device_properties(0)
    manifest = {
        "schema_version": "decision2-score-en-direct-lora-zero-step/1",
        "mode": args.mode,
        "arm": args.arm,
        "model_sha256": SOURCE_SHA,
        "source_files_sha256": identity["files_sha256"],
        "initial_adapter_sha256": identity["files_sha256"][
            "checkpoint/adapter/adapter_model.safetensors"
        ],
        "initial_head_sha256": identity["files_sha256"][
            "checkpoint/decision_head.safetensors"
        ],
        "roster_sha256": ROSTER_SHA,
        "roster_manifest_sha256": file_sha256(args.roster_dir / "manifest.json"),
        "input_sha256": file_sha256(roster_path),
        "token_ids_sha256": roster_manifest["token_ids_sha256"],
        "adapter_sha256": code_sha,
        "adapter_files_sha256": code_files,
        "counts": counts,
        "max_length": MAX_LENGTH,
        "temperature": 1.0,
        "execution": "one question at a time; BF16 backbone; FP32 head; source PEFT LoRA",
        "container_image_id": args.container_image_id,
        "software": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "hardware": {
            "accelerator_name": hardware.name,
            "total_memory_bytes": hardware.total_memory,
            "visible_device_count": torch.cuda.device_count(),
        },
    }
    write_output(args.output, predictions, manifest)
    return {
        "mode": args.mode,
        "arm": args.arm,
        "predictions_sha256": file_sha256(args.output),
        "counts": counts,
    }


def _read(
    path: Path, mode: str, arm: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("predictions_sha256") != file_sha256(path)
        or manifest.get("mode") != mode
        or manifest.get("arm") != arm
        or manifest.get("model_sha256") != SOURCE_SHA
        or manifest.get("roster_sha256") != ROSTER_SHA
        or manifest.get("counts", {}).get("valid_questions") != 32
        or manifest.get("counts", {}).get("invalid_questions") != 0
        or manifest.get("counts", {}).get("truncated_questions") != 0
        or manifest.get("max_length") != MAX_LENGTH
        or manifest.get("temperature") != 1.0
    ):
        raise ValueError("A sealed zero-step prediction manifest differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 32 or any(
        row.get("model_sha256") != SOURCE_SHA
        or row.get("adapter_sha256") != manifest.get("adapter_sha256")
        or row.get("truncated_questions") != 0
        for row in rows
    ):
        raise ValueError("Incomplete zero-step prediction file")
    return rows, manifest


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("Zero-step parity receipt already exists")
    roster, roster_manifest, roster_path = _roster(args.roster_dir)
    reference, reference_manifest = _read(args.reference, "reference", None)
    results = {}
    if (
        reference_manifest.get("input_sha256") != ROSTER_SHA
        or reference_manifest.get("token_ids_sha256")
        != roster_manifest["token_ids_sha256"]
    ):
        raise ValueError("Reference tokenizer or roster differs")
    for arm, path, train_path in (
        ("A", args.arm_a, args.train_a),
        ("B", args.arm_b, args.train_b),
    ):
        candidate, manifest = _read(path, "trainable", arm)
        if file_sha256(train_path) != ARM_SHA[arm]:
            raise ValueError(f"{arm}: frozen TRAIN data differs")
        for field in (
            "input_sha256",
            "adapter_sha256",
            "adapter_files_sha256",
            "source_files_sha256",
            "token_ids_sha256",
            "execution",
            "software",
            "hardware",
            "initial_adapter_sha256",
            "initial_head_sha256",
            "container_image_id",
        ):
            if manifest.get(field) != reference_manifest.get(field):
                raise ValueError(
                    f"{arm}: native reference and trainable contracts differ: {field}"
                )
        same = 0
        drift = 0.0
        for index, (source, loaded, item) in enumerate(
            zip(reference, candidate, roster)
        ):
            if (
                source.get("id") != loaded.get("id")
                or source.get("id") != item["id"]
                or source.get("input_sha256") != loaded.get("input_sha256")
                or source.get("input_sha256") != prompt_input_sha256(item)
                or source.get("usage", {}).get("input_tokens")
                != loaded.get("usage", {}).get("input_tokens")
                or source.get("usage", {}).get("input_tokens")
                != roster_manifest["token_count"][index]
            ):
                raise ValueError(f"{arm}: native zero-step inputs differ")
            kind = item["questions"]["decision"]["type"]
            a_probs, a_winner = _probabilities(source, kind)
            b_probs, b_winner = _probabilities(loaded, kind)
            if a_probs.keys() != b_probs.keys():
                raise ValueError(f"{arm}: native option sets differ")
            same += a_winner == b_winner
            drift = max(drift, *(abs(a_probs[key] - b_probs[key]) for key in a_probs))
        results[arm] = {
            "prediction_sha256": file_sha256(path),
            "prediction_manifest_sha256": file_sha256(
                path.with_name(path.name + ".manifest.json")
            ),
            "train_sha256": ARM_SHA[arm],
            "same_argmax": same,
            "max_absolute_option_probability_drift": drift,
            "status": (
                "PASS" if same == 32 and drift <= TOLERANCE else "BLOCKED_START_PARITY"
            ),
        }
    report = {
        "schema_version": "decision2-score-en-direct-lora-start-parity/1",
        "source_model_sha256": SOURCE_SHA,
        "roster_sha256": file_sha256(roster_path),
        "reference_prediction_sha256": file_sha256(args.reference),
        "reference_manifest_sha256": file_sha256(
            args.reference.with_name(args.reference.name + ".manifest.json")
        ),
        "adapter_sha256": reference_manifest["adapter_sha256"],
        "adapter_files_sha256": reference_manifest["adapter_files_sha256"],
        "source_files_sha256": reference_manifest["source_files_sha256"],
        "initial_adapter_sha256": reference_manifest["initial_adapter_sha256"],
        "initial_head_sha256": reference_manifest["initial_head_sha256"],
        "container_image_id": reference_manifest["container_image_id"],
        "software": reference_manifest["software"],
        "hardware": reference_manifest["hardware"],
        "arms": results,
        "tolerance": TOLERANCE,
        "status": (
            "PASS"
            if all(r["status"] == "PASS" for r in results.values())
            else "BLOCKED_START_PARITY"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    predict_parser = sub.add_parser("predict")
    for name in ("checkpoint", "source-path", "roster-dir", "output"):
        predict_parser.add_argument("--" + name, required=True, type=Path)
    predict_parser.add_argument("--container-image-id", required=True)
    predict_parser.add_argument(
        "--mode", choices=("reference", "trainable"), required=True
    )
    predict_parser.add_argument("--arm", choices=("A", "B"))
    compare_parser = sub.add_parser("compare")
    for name in (
        "roster-dir",
        "reference",
        "arm-a",
        "arm-b",
        "train-a",
        "train-b",
        "output",
    ):
        compare_parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.action == "predict":
        if (args.mode == "trainable") != (args.arm in ARM_SHA):
            parser.error("Trainable mode needs A/B arm; reference mode needs no arm")
        print(json.dumps(predict(args), sort_keys=True))
    else:
        report = compare(args)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "arms": {
                        key: {
                            "same_argmax": value["same_argmax"],
                            "max_drift": value["max_absolute_option_probability_drift"],
                        }
                        for key, value in report["arms"].items()
                    },
                },
                sort_keys=True,
            )
        )
        if report["status"] != "PASS":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
