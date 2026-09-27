"""Gold-free 32-item native source/LoRA start parity for Score v7p.

This is versioned separately from the failed v6 direct-start experiment.
The v7p A/B/C TRAIN hashes come only from a sealed private admission manifest.
No selector key or benchmark label is read here.
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
    question_to_row,
    run_prompts,
    write_output,
)
from training.model.train import load_direct_lora_start

SOURCE_SHA = "d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2"
ROSTER_SHA = "193404fb2ed3905cbb9e34379400a2f33a40d86aaae971940454c6fe71163bc5"
IMAGE_ID = "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54"
MAX_LENGTH = 1024
TOLERANCE = 1e-4


def _source_code() -> dict[str, str]:
    root = Path(__file__).parents[1]
    files = (
        Path(__file__),
        root / "model/train.py",
        root / "model/infer.py",
        root / "model/decision_model.py",
        root / "model/lora.py",
    )
    return {str(path.relative_to(root)): file_sha256(path) for path in files}


def _admission(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    hashes = value.get("arm_sha256")
    if (
        value.get("schema_version") != "decision2-score-v7p-matched-arms/1"
        or value.get("status") != "CANDIDATE_PENDING_BLIND_QA_AND_ZERO_STEP"
        or not isinstance(hashes, dict)
        or set(hashes) != {"A", "B", "C"}
        or hashes["A"] != hashes["C"]
    ):
        raise ValueError("v7p matched-arm admission manifest differs")
    return hashes


def _roster(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    file = path / "prompts.jsonl"
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        file_sha256(file) != ROSTER_SHA
        or manifest.get("roster_sha256") != ROSTER_SHA
        or manifest.get("roster_rows") != 32
        or manifest.get("source_model_sha256") != SOURCE_SHA
    ):
        raise ValueError("Frozen 32-item gold-free roster differs")
    rows = load_prompts(file)
    if len(rows) != 32:
        raise ValueError("Parity roster length differs")
    return rows, manifest, file


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def predict(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.output.exists() or _manifest_path(args.output).exists():
        raise FileExistsError("Zero-step prediction output already exists")
    if args.container_image_id != IMAGE_ID:
        raise ValueError("Native runtime image differs")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native BF16 parity needs a GPU")
    roster, roster_manifest, roster_file = _roster(args.roster_dir)
    identity = checkpoint_fingerprint(args.checkpoint, args.source_path)
    if identity["model_sha256"] != SOURCE_SHA:
        raise ValueError("Pinned BEST368 source weights differ")
    train_sha = None
    if args.mode == "trainable":
        expected = _admission(args.admission_manifest)
        train_sha = file_sha256(args.train)
        if train_sha != expected[args.arm]:
            raise ValueError("Requested v7p TRAIN bytes differ")
        loader_args = argparse.Namespace(
            model_path=str(args.checkpoint),
            source_path=str(args.source_path),
            initial_model_sha256=SOURCE_SHA,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.05,
            head_dim=256,
        )
        model, tokenizer, loaded_identity = load_direct_lora_start(loader_args)
        if loaded_identity != identity:
            raise ValueError("Trainable LoRA source loader changed model identity")
    else:
        model, tokenizer = DecisionModel.from_checkpoint(
            args.checkpoint, source_path=args.source_path, trainable_adapter=False
        )
    model = model.float().to("cuda:0").eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer lacks pad and EOS tokens")
    encodings = [
        encode(
            question_to_row(item, "decision", item["questions"]["decision"]),
            tokenizer,
            MAX_LENGTH,
        )
        for item in roster
    ]
    if [row["prompt_sha256"] for row in encodings] != roster_manifest[
        "input_sha256"
    ] or [row["token_ids_sha256"] for row in encodings] != roster_manifest[
        "token_ids_sha256"
    ]:
        raise ValueError("Native roster tokenizer/prompt encoding differs")

    def forward(encoded: list[dict[str, Any]]) -> list[list[float]]:
        batch = {
            key: value.to("cuda:0") if torch.is_tensor(value) else value
            for key, value in collate(encoded, pad_id).items()
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

    code = _source_code()
    adapter_sha = hashlib.sha256(canonical(code).encode()).hexdigest()
    predictions, counts = run_prompts(
        roster,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        temperature=1.0,
        encode_fn=encode,
        predict_fn=forward,
        model_sha256=SOURCE_SHA,
        adapter_sha256=adapter_sha,
    )
    expected_counts = {
        "items": 32,
        "questions": 32,
        "valid_questions": 32,
        "invalid_questions": 0,
        "over_budget_questions": 0,
        "truncated_questions": 0,
    }
    if counts != expected_counts:
        raise ValueError("Native zero-step path had missing or invalid predictions")
    hardware = torch.cuda.get_device_properties(0)
    manifest = {
        "schema_version": "decision2-score-v7p-zero-step-prediction/1",
        "mode": args.mode,
        "arm": args.arm,
        "train_sha256": train_sha,
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
        "input_sha256": file_sha256(roster_file),
        "token_ids_sha256": roster_manifest["token_ids_sha256"],
        "adapter_sha256": adapter_sha,
        "adapter_files_sha256": code,
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
    path: Path, mode: str, arm: str | None, roster: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "decision2-score-v7p-zero-step-prediction/1"
        or manifest.get("predictions_sha256") != file_sha256(path)
        or manifest.get("mode") != mode
        or manifest.get("arm") != arm
        or manifest.get("model_sha256") != SOURCE_SHA
        or manifest.get("roster_sha256") != ROSTER_SHA
        or manifest.get("counts", {}).get("valid_questions") != 32
        or manifest.get("counts", {}).get("invalid_questions") != 0
        or manifest.get("counts", {}).get("over_budget_questions") != 0
        or manifest.get("counts", {}).get("truncated_questions") != 0
        or manifest.get("max_length") != MAX_LENGTH
        or manifest.get("temperature") != 1.0
    ):
        raise ValueError("Zero-step prediction manifest differs")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 32 or [row.get("id") for row in rows] != [
        item["id"] for item in roster
    ]:
        raise ValueError("Zero-step prediction rows differ")
    return rows, manifest


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("v7p direct-start receipt already exists")
    roster, roster_manifest, roster_file = _roster(args.roster_dir)
    arm_hashes = _admission(args.admission_manifest)
    reference, reference_manifest = _read(args.reference, "reference", None, roster)
    if (
        reference_manifest["input_sha256"] != ROSTER_SHA
        or reference_manifest["token_ids_sha256"] != roster_manifest["token_ids_sha256"]
    ):
        raise ValueError("Reference roster tokenizer differs")
    results = {}
    for arm in ("A", "B", "C"):
        path = getattr(args, "arm_" + arm.lower())
        train = getattr(args, "train_" + arm.lower())
        candidate, manifest = _read(path, "trainable", arm, roster)
        if (
            file_sha256(train) != arm_hashes[arm]
            or manifest.get("train_sha256") != arm_hashes[arm]
        ):
            raise ValueError(f"{arm}: frozen TRAIN identity differs")
        for field in (
            "input_sha256",
            "token_ids_sha256",
            "adapter_sha256",
            "adapter_files_sha256",
            "source_files_sha256",
            "initial_adapter_sha256",
            "initial_head_sha256",
            "execution",
            "software",
            "hardware",
            "container_image_id",
        ):
            if manifest.get(field) != reference_manifest.get(field):
                raise ValueError(
                    f"{arm}: native source/start runtime differs at {field}"
                )
        same, drift = 0, 0.0
        for index, (source, loaded, prompt) in enumerate(
            zip(reference, candidate, roster)
        ):
            if (
                source.get("id") != loaded.get("id")
                or source.get("input_sha256") != loaded.get("input_sha256")
                or source.get("input_sha256") != prompt_input_sha256(prompt)
                or source.get("usage", {}).get("input_tokens")
                != loaded.get("usage", {}).get("input_tokens")
                or source.get("usage", {}).get("input_tokens")
                != roster_manifest["token_count"][index]
            ):
                raise ValueError(f"{arm}: native inputs differ")
            kind = prompt["questions"]["decision"]["type"]
            left, winner_left = _probabilities(source, kind)
            right, winner_right = _probabilities(loaded, kind)
            if left.keys() != right.keys():
                raise ValueError(f"{arm}: output candidate keys differ")
            same += winner_left == winner_right
            drift = max(drift, *(abs(left[key] - right[key]) for key in left))
        results[arm] = {
            "status": (
                "PASS" if same == 32 and drift <= TOLERANCE else "BLOCKED_START_PARITY"
            ),
            "train_sha256": arm_hashes[arm],
            "prediction_sha256": file_sha256(path),
            "prediction_manifest_sha256": file_sha256(_manifest_path(path)),
            "same_argmax": same,
            "max_absolute_option_probability_drift": drift,
        }
    result = {
        "schema_version": "decision2-score-v7p-direct-lora-start/2",
        "status": (
            "PASS"
            if all(item["status"] == "PASS" for item in results.values())
            else "BLOCKED_START_PARITY"
        ),
        "source_model_sha256": SOURCE_SHA,
        "source_files_sha256": reference_manifest["source_files_sha256"],
        "roster_sha256": ROSTER_SHA,
        "roster_items": 32,
        "roster_manifest_sha256": file_sha256(args.roster_dir / "manifest.json"),
        "reference_prediction_sha256": file_sha256(args.reference),
        "reference_prediction_manifest_sha256": file_sha256(
            _manifest_path(args.reference)
        ),
        "admission_manifest_sha256": file_sha256(args.admission_manifest),
        "adapter_sha256": reference_manifest["adapter_sha256"],
        "adapter_files_sha256": reference_manifest["adapter_files_sha256"],
        "container_image_id": IMAGE_ID,
        "software": reference_manifest["software"],
        "hardware": reference_manifest["hardware"],
        "tolerance": TOLERANCE,
        "arms": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    predict_parser = sub.add_parser("predict")
    for name in (
        "checkpoint",
        "source-path",
        "roster-dir",
        "admission-manifest",
        "output",
    ):
        predict_parser.add_argument("--" + name, required=True, type=Path)
    predict_parser.add_argument("--container-image-id", required=True)
    predict_parser.add_argument(
        "--mode", choices=("reference", "trainable"), required=True
    )
    predict_parser.add_argument("--arm", choices=("A", "B", "C"))
    predict_parser.add_argument("--train", type=Path)
    compare_parser = sub.add_parser("compare")
    for name in (
        "roster-dir",
        "admission-manifest",
        "reference",
        "arm-a",
        "arm-b",
        "arm-c",
        "train-a",
        "train-b",
        "train-c",
        "output",
    ):
        compare_parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.action == "predict":
        if (args.mode == "trainable") != (
            args.arm in {"A", "B", "C"} and args.train is not None
        ):
            parser.error("Trainable mode needs arm plus TRAIN; reference needs neither")
        print(json.dumps(predict(args), sort_keys=True))
    else:
        result = compare(args)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "arms": {
                        key: {
                            "same_argmax": value["same_argmax"],
                            "max_drift": value["max_absolute_option_probability_drift"],
                        }
                        for key, value in result["arms"].items()
                    },
                },
                sort_keys=True,
            )
        )
        if result["status"] != "PASS":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
