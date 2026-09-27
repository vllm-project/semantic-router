"""CPU-only, one-shot lock for the official 4B Posttrained source ablation.

The lock is a review artifact, not permission to start a GPU. Run inside the
pinned Base-compatible image with no GPU devices and no network. Private file
paths and all data stay in the owner's private experiment directory.
"""

from __future__ import annotations

import argparse
import json
import math
import stat
from pathlib import Path

from scripts.admit_qwen35_4b_posttrained import (
    DATA_SHA256,
    IMAGE_ID,
    POSTTRAINED_REVISION,
    RUNTIME_PYTHON,
    _data,
    _private_root,
    _require_no_gpu,
    _runtime,
    _source,
    _write_once,
)
from scripts.preflight_qwen35_4b_posttrained_cpu import sha256_file
from training.model.plan import planned_updates

SCHEMA = "decision2-qwen35-4b-posttrained-full-clean-v2/1"
BASE_PROVENANCE_SHA256 = (
    "49b58711f9645af6806374d89dfce3db453460cf2921ae8777742290356ec7c8"
)
BASE_COMPLETE_SHA256 = (
    "a8543f3b8173cecfe7028444a6e3b015a98c1a6cdd80757d362cc299dd4490c9"
)
ADMISSION_LOCK_SHA256 = (
    "cb2690675dc1d6d39370acbf4445ea667b0d08ee54c369077ee8c9f55fcfb86a"
)
ADMISSION_FINAL_SHA256 = (
    "4b8a21348a8e0d957cf7f5f520d1b2ee7a81f8816453b25d8f50ee4a5aa21054"
)
CPU_AUDIT_SHA256 = "0c45417d4cc8f3e4c32edc55aec3c230e0922476500acc1fabc3b5c2151ac82b"
TRAIN_TOKENS = 4_194_465
TRAIN_MAX_TOKENS = 6_596
TRAIN_ROWS = 7_455
PLANNED_UPDATES = 466
GPU_HOUR_CAP = 3.0
CHECKPOINTS = [64, 128, 192, 256, 320, 384, 448, 466]
TRAINER_FILES = (
    "data.py",
    "decision_model.py",
    "loss.py",
    "plan.py",
    "source.py",
    "train.py",
    "lora.py",
)


def _read_sealed(path: Path, expected: str) -> dict:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError("Archived private evidence is missing or changed")
    return json.loads(path.read_text(encoding="utf-8"))


def _training_code() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1] / "training/model"
    return {name: sha256_file(root / name) for name in TRAINER_FILES}


def _base_contract(provenance: dict, complete: dict) -> dict:
    contract = provenance["contract"]
    expected = {
        "epochs": 1,
        "max_steps": None,
        "microbatch": 1,
        "accumulation": 16,
        "eval_batch": 2,
        "max_length": 8192,
        "head_dim": 256,
        "backbone_lr": 2e-5,
        "head_lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "save_every": 64,
        "seed": 20260926,
        "gradient_checkpointing": True,
        "objective": "ce_brier",
        "brier_weight": 0.5,
        "replay_fraction": 0.0,
        "replay_kl_weight": 0.0,
        "choice_source": [],
        "choice_source_weight": 1.0,
        "weighted_choice_count": 0,
        "train_mode": "lora",
        "planned_updates": PLANNED_UPDATES,
        "train_count": TRAIN_ROWS,
        "replay_pool_count": 0,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed Base control differs from fixed arm")
    if (
        contract.get("init_kind") != "base"
        or contract.get("base_revision") != "1001bb4d826a52d1f399e183466143f4da7b741b"
        or contract.get("data_sha256") != DATA_SHA256
        or any(
            contract.get("lora", {}).get(k) != v
            for k, v in {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "lr": 1e-4,
                "peft_version": "0.21.0",
            }.items()
        )
        or provenance.get("train_examples") != TRAIN_ROWS
        or provenance.get("select_examples") != 700
        or provenance.get("cal_examples_audited_only") != 700
        or provenance.get("train_tokens") != TRAIN_TOKENS
        or provenance.get("train_max_tokens") != TRAIN_MAX_TOKENS
        or complete.get("status") != "complete"
        or complete.get("step") != PLANNED_UPDATES
        or complete.get("planned_updates") != PLANNED_UPDATES
        or complete.get("calibration_status") != "untouched"
    ):
        raise ValueError("Completed Base evidence does not admit a matched arm")
    if provenance.get("code_sha256") != _training_code():
        raise ValueError("Trainer code differs from completed Base control")
    if planned_updates(TRAIN_ROWS, 0, 0.0, 1, 16, 1, None) != PLANNED_UPDATES:
        raise ValueError("Update planner differs from completed Base control")
    return expected


def _admission(lock: dict, final: dict, source: dict) -> None:
    if (
        lock.get("status") != "LOCKED_NO_GPU"
        or lock.get("source_revision") != POSTTRAINED_REVISION
        or lock.get("source_fingerprint") != source
        or lock.get("data_sha256") != DATA_SHA256
        or lock.get("image_id") != IMAGE_ID
        or final.get("status") != "PASS_ADMISSION_ONLY"
        or final.get("source_revision") != POSTTRAINED_REVISION
        or final.get("lock_sha256") != ADMISSION_LOCK_SHA256
        or final.get("zero_gate", {}).get("categorical_changes") != 0
        or final.get("reload", {}).get("categorical_changes") != 0
        or final.get("one_update", {}).get("step") != 1
        or not math.isfinite(final.get("one_update", {}).get("loss", float("nan")))
        or not math.isfinite(
            final.get("one_update", {}).get("gradient_norm", float("nan"))
        )
        or final.get("formal_label_access") is not False
        or final.get("cal_fit") is not False
    ):
        raise ValueError("Bounded Posttrained admission did not pass unchanged")


def _argv(paths: dict[str, Path], source: Path, output: Path) -> list[str]:
    # Omit --max-steps: the matched Base contract has max_steps=None.
    return [
        RUNTIME_PYTHON,
        "-m",
        "training.model.train",
        "--model-path",
        str(source),
        "--init-kind",
        "posttrained",
        "--base-revision",
        POSTTRAINED_REVISION,
        "--train",
        str(paths["train"]),
        "--select",
        str(paths["select"]),
        "--cal",
        str(paths["cal"]),
        "--output",
        str(output),
        "--epochs",
        "1",
        "--microbatch",
        "1",
        "--accumulation",
        "16",
        "--eval-batch",
        "2",
        "--max-length",
        "8192",
        "--head-dim",
        "256",
        "--backbone-lr",
        "2e-5",
        "--head-lr",
        "2e-4",
        "--weight-decay",
        "0.01",
        "--warmup-ratio",
        "0.05",
        "--save-every",
        "64",
        "--seed",
        "20260926",
        "--gradient-checkpointing",
        "--objective",
        "ce_brier",
        "--brier-weight",
        "0.5",
        "--train-mode",
        "lora",
        "--lora-rank",
        "16",
        "--lora-alpha",
        "32",
        "--lora-dropout",
        "0.05",
        "--lora-lr",
        "1e-4",
    ]


def prepare(args: argparse.Namespace) -> dict:
    _require_no_gpu()
    runtime = _runtime()
    if args.image_id != IMAGE_ID or runtime["python"] != RUNTIME_PYTHON:
        raise ValueError("Runtime differs from completed Base control")
    root = _private_root(args.output_root)
    if any(root.iterdir()):
        raise FileExistsError("Full-arm lock directory must be new and empty")
    if args.arm_output.is_symlink():
        raise ValueError("Fresh arm output must not be a symlink")
    arm_output = args.arm_output.resolve()
    if arm_output.exists() and any(arm_output.iterdir()):
        raise FileExistsError("Fresh full-arm output must be absent or empty")
    paths = {
        role: getattr(args, role).resolve(strict=True)
        for role in ("train", "select", "cal")
    }
    source_path = args.source.resolve(strict=True)
    source = _source(source_path)
    data_sha = _data(paths)  # CAL is parsed/hashed ONLY for split isolation.
    if data_sha != DATA_SHA256:
        raise ValueError("Frozen data hashes differ")
    base = _read_sealed(args.base_provenance, BASE_PROVENANCE_SHA256)
    complete = _read_sealed(args.base_complete, BASE_COMPLETE_SHA256)
    base_contract = _base_contract(base, complete)
    admission_lock = _read_sealed(args.admission_lock, ADMISSION_LOCK_SHA256)
    admission_final = _read_sealed(args.admission_final, ADMISSION_FINAL_SHA256)
    _admission(admission_lock, admission_final, source)
    audit = _read_sealed(args.cpu_audit, CPU_AUDIT_SHA256)
    if (
        audit.get("exact_input_parity") is not True
        or audit.get("panels", {}).get("train", {}).get("posttrained_tokens")
        != TRAIN_TOKENS
        or audit.get("panels", {}).get("train", {}).get("rows") != TRAIN_ROWS
    ):
        raise ValueError("Native token parity audit is absent or changed")
    argv = _argv(paths, source_path, arm_output)
    result = {
        "schema_version": SCHEMA,
        "status": "LOCKED_NO_GPU",
        "approval": "Independent review required before any GPU training",
        "image_id": IMAGE_ID,
        "runtime": runtime,
        "source_revision": POSTTRAINED_REVISION,
        "source_fingerprint": source,
        "data_sha256": data_sha,
        "private_paths": {
            **{role: str(path) for role, path in paths.items()},
            "source": str(source_path),
            "arm_output": str(arm_output),
        },
        "trainer_code_sha256": _training_code(),
        "lock_preparer_sha256": sha256_file(Path(__file__)),
        "evidence_sha256": {
            "base_provenance": BASE_PROVENANCE_SHA256,
            "base_complete": BASE_COMPLETE_SHA256,
            "admission_lock": ADMISSION_LOCK_SHA256,
            "admission_final": ADMISSION_FINAL_SHA256,
            "cpu_native_input_audit": CPU_AUDIT_SHA256,
        },
        "matched_base_contract": base_contract,
        "train_rows": TRAIN_ROWS,
        "train_input_tokens": TRAIN_TOKENS,
        "train_max_tokens": TRAIN_MAX_TOKENS,
        "planned_updates": PLANNED_UPDATES,
        "checkpoint_steps": CHECKPOINTS,
        "checkpoint_selector": [
            "SELECT family macro accuracy descending",
            "SELECT normalized Brier ascending",
            "earliest update",
        ],
        "cal_boundary": "Parsed and hashed for split isolation only; no tokenization, forward, metric, temperature, checkpoint selection, or release claim",
        "gpu_hour_cap": GPU_HOUR_CAP,
        "hard_stops": [
            "source, data, image, runtime, code or contract mismatch",
            "nonfinite loss or gradient norm",
            "OOM, missing row/type, overlength input, failed checkpoint",
            "3.0 one-GPU hours cumulative including SELECT and saves",
        ],
        "output_path": str(arm_output),
        "trainer_argv": argv,
        "selector_inputs": ["SELECT only"],
        "excluded_inputs": [
            "CAL model use",
            "typed DEV",
            "CSS pilot",
            "typed FINAL",
            "CSS15",
            "JevBench",
        ],
    }
    _write_once(root / "full-arm-lock.json", result)
    return result


def verify(args: argparse.Namespace) -> None:
    _require_no_gpu()
    if args.lock.is_symlink():
        raise ValueError("Private lock must not be a symlink")
    lock_path = args.lock.resolve(strict=True)
    _private_root(lock_path.parent)
    if stat.S_IMODE(lock_path.stat().st_mode) != 0o600:
        raise ValueError("Private lock must be a regular mode-0600 file")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != SCHEMA
        or lock.get("status") != "LOCKED_NO_GPU"
        or lock.get("image_id") != IMAGE_ID
        or lock.get("runtime") != _runtime()
        or lock.get("lock_preparer_sha256") != sha256_file(Path(__file__))
        or lock.get("trainer_code_sha256") != _training_code()
        or lock.get("planned_updates") != PLANNED_UPDATES
        or lock.get("checkpoint_steps") != CHECKPOINTS
        or lock.get("gpu_hour_cap") != GPU_HOUR_CAP
    ):
        raise ValueError("Full-arm lock/runtime/code changed")
    paths = lock["private_paths"]
    source = Path(paths["source"])
    partitions = {role: Path(paths[role]) for role in DATA_SHA256}
    if _source(source) != lock["source_fingerprint"]:
        raise ValueError("Posttrained source changed after lock")
    if _data(partitions) != lock["data_sha256"]:
        raise ValueError("Private partitions changed after lock")
    if lock["trainer_argv"] != _argv(partitions, source, Path(paths["arm_output"])):
        raise ValueError("Trainer command changed after lock")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    for role in (
        "source",
        "train",
        "select",
        "cal",
        "base_provenance",
        "base_complete",
        "admission_lock",
        "admission_final",
        "cpu_audit",
        "output_root",
        "arm_output",
        "image_id",
    ):
        prep.add_argument(
            "--" + role.replace("_", "-"),
            required=True,
            type=Path if role != "image_id" else str,
        )
    check = sub.add_parser("verify")
    check.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
        print("FULL_ARM_LOCKED_NO_GPU")
    else:
        verify(args)
        print("FULL_ARM_LOCK_VERIFIED_NO_GPU")


if __name__ == "__main__":
    main()
