"""Bounded no-update replay of the failed official-Base 9B TRAIN window.

Run inside the pinned training image with the original trainer source on
PYTHONPATH. This probe never accepts evaluation partitions or saves weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import chain
from pathlib import Path

EXPECTED = {
    "train": "fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c",
    "provenance": "21a544b82a3b98a5bcb5a123ae37d5802ecf5fa301e2ce0e03572ea8fdc6d94c",
    "metrics": "10ff2bd1934304350e68e12751f9d70eef91709569a66599192dfd7435024e79",
    "console": "59e2aecf8e2c6d684598d0cbd994c305cd38c9ed83834ba2673f5bd559f13ded",
    "state": "cbe3b4be2f9fd42db9fd7c11e631608eb10a3b4feef829ffaf8f8f509f5c6786",
}
SOURCE_HASHES = {
    "data.py": "632bd60555f63459ff08ffe82c8263ca6a96f75360bc8c06469fcb10f1e2e99c",
    "decision_model.py": "ee3db820db73011d60e99b98c3067e33d85c8f4cbae08b53ea0604869ffba0ca",
    "lora.py": "7049e35e2a7bf50dd5888902cd0b7030d9bbd448d7d7aba8875ceb89aebff231",
    "loss.py": "5b07dbe3b0fa55d95df25dcb20733b7b5e76b306ced414195ef715c5ba007b7a",
    "plan.py": "c39d706e1217dbc41c4f8e4f2e8d66b69058baa2e1c26abd8f194fea374cbfcb",
    "source.py": "ef7b30171c4befb26163ea4d3d6e5add9dc602228a476d6f0ead9d40b450c7e3",
    "train.py": "b24080d6289fc2d4fde8bd11d63f16cb9b02b1906c603a986fbe0ad564a2a83c",
}
SOURCE_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
STEP = 107
ACCUMULATION = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def scheduled_indices(lengths: list[int], seed: int) -> dict[str, list[int]]:
    from training.model.plan import epoch_batches

    batches = epoch_batches(
        lengths, [], epoch=0, seed=seed, microbatch=1, replay_fraction=0.0
    )
    if len(batches) != 7324 or any(len(batch) != 1 for batch in batches):
        raise ValueError("Frozen one-example TRAIN schedule differs")
    return {
        str(step): [batch[0][1] for batch in batches[(step - 1) * 16 : step * 16]]
        for step in (105, 106, 107)
    }


def check_inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    from training.model import train as original_train
    from training.model.data import file_sha256

    sources = Path(original_train.__file__).parent
    for name, expected in SOURCE_HASHES.items():
        if file_sha256(sources / name) != expected:
            raise ValueError(f"Original trainer source mismatch: {name}")
    paths = {
        "train": args.train,
        "provenance": args.run / "provenance.json",
        "metrics": args.run / "train-metrics.jsonl",
        "console": args.console,
        "state": args.run / "checkpoint-0000064/trainer_state.pt",
    }
    for name, path in paths.items():
        if sha256(path) != EXPECTED[name]:
            raise ValueError(f"Frozen {name} SHA-256 mismatch")
    provenance = json.loads(paths["provenance"].read_text())
    contract = provenance["contract"]
    required = {
        "data_sha256": {
            "train": EXPECTED["train"],
            "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
            "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
        },
        "train_count": 7324,
        "replay_pool_count": 0,
        "replay_fraction": 0.0,
        "microbatch": 1,
        "accumulation": 16,
        "seed": 20260926,
        "max_length": 4096,
        "train_mode": "lora",
        "init_kind": "base",
        "base_revision": SOURCE_REVISION,
        "gradient_checkpointing": False,
        "objective": "ce_brier",
        "brier_weight": 0.5,
        "planned_updates": 458,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ValueError(f"Frozen training contract mismatch: {key}")
    if provenance["code_sha256"] != SOURCE_HASHES:
        raise ValueError("Frozen trainer hashes differ from run provenance")
    ck_meta = json.loads((args.run / "checkpoint-0000064/checkpoint.json").read_text())
    if (ck_meta["step"], ck_meta["next_epoch"], ck_meta["next_batch"]) != (64, 0, 1024):
        raise ValueError("Step-64 cursor differs")
    steps = [json.loads(line) for line in paths["metrics"].read_text().splitlines()]
    done = [row["step"] for row in steps if row.get("event") == "train"]
    if done != list(range(1, 107)):
        raise ValueError("The last complete optimizer update is not exactly 106")
    return provenance, ck_meta


def audit(args: argparse.Namespace) -> dict:
    from training.model.data import load_partition
    from training.model.decision_model import encode
    from transformers import AutoTokenizer

    provenance, _ = check_inputs(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = load_partition(args.train, "train")
    items = [encode(row, tokenizer, 4096) for row in rows]
    lengths = [len(item["ids"]) for item in items]
    if len(items) != 7324 or sum(lengths) != 3579176 or max(lengths) > 4096:
        raise ValueError("TRAIN token count or cap differs from frozen arm")
    schedule = scheduled_indices(lengths, 20260926)
    observed = {
        row["step"]: row["tokens"]
        for row in (
            json.loads(line)
            for line in (args.run / "train-metrics.jsonl").read_text().splitlines()
        )
        if row.get("event") == "train" and row.get("step") in (105, 106)
    }
    for step in (105, 106):
        if sum(lengths[i] for i in schedule[str(step)]) != observed.get(step):
            raise ValueError(
                f"Reconstructed update {step} token total differs from sealed log"
            )
    short = min(range(len(lengths)), key=lambda i: (lengths[i], i))
    longest = max(range(len(lengths)), key=lambda i: (lengths[i], -i))
    selected = {
        str(i): {
            "row_sha256": hashlib.sha256(
                canonical([rows[i]["id"], rows[i]["input_sha256"]])
            ).hexdigest(),
            "task_type": items[i]["task_type"],
            "tokens": lengths[i],
            "label_valid": 0 <= items[i]["label"] < len(items[i]["keys"]),
            "token_ids_sha256": items[i]["token_ids_sha256"],
        }
        for i in sorted(set(chain.from_iterable(schedule.values())) | {short, longest})
    }
    if not all(row["label_valid"] for row in selected.values()):
        raise ValueError("A replay TRAIN label is invalid")
    result = {
        "protocol": "qwen35-9b-base-step107-no-update-v1",
        "source_revision": SOURCE_REVISION,
        "train_sha256": EXPECTED["train"],
        "checkpoint_state_sha256": EXPECTED["state"],
        "original_log_sha256": EXPECTED["console"],
        "source_sha256": SOURCE_HASHES,
        "seed": provenance["contract"]["seed"],
        "schedule": schedule,
        "short_index": short,
        "long_index": longest,
        "rows": selected,
        "limitation": "Exact TRAIN order and tokenization; step-106 parameter and RNG state were not saved.",
    }
    result["manifest_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.chmod(args.manifest, 0o600)
    return {
        "manifest_sha256": result["manifest_sha256"],
        "step107_tokens": [lengths[i] for i in schedule["107"]],
        "short_tokens": lengths[short],
        "long_tokens": lengths[longest],
    }


def replay(args: argparse.Namespace) -> dict:
    import torch
    from training.model.data import load_partition
    from training.model.decision_model import DecisionModel, collate, encode
    from training.model.loss import per_example_loss

    check_inputs(args)
    manifest = json.loads(args.manifest.read_text())
    fingerprint = manifest.pop("manifest_sha256")
    if hashlib.sha256(canonical(manifest)).hexdigest() != fingerprint:
        raise ValueError("Private TRAIN schedule manifest changed")
    manifest["manifest_sha256"] = fingerprint
    if (
        manifest["train_sha256"] != EXPECTED["train"]
        or manifest["source_sha256"] != SOURCE_HASHES
    ):
        raise ValueError("Manifest source or TRAIN identity differs")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("One BF16 ROCm device is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Probe must see exactly one accelerator")
    device = torch.device("cuda:0")
    model, tokenizer = DecisionModel.from_checkpoint(
        args.run / "checkpoint-0000064", source_path=args.model, trainable_adapter=True
    )
    model = model.float().to(device)
    model.backbone.config.use_cache = False
    model.train()
    rows = load_partition(args.train, "train")
    wanted = sorted({int(i) for i in manifest["rows"]})
    items = {i: encode(rows[i], tokenizer, 4096) for i in wanted}
    for i in wanted:
        evidence = manifest["rows"][str(i)]
        if (
            len(items[i]["ids"]) != evidence["tokens"]
            or items[i]["token_ids_sha256"] != evidence["token_ids_sha256"]
        ):
            raise ValueError("Replay tokenizer or row differs from sealed manifest")
    original = {
        name: hashlib.sha256(
            parameter.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    args.events.parent.mkdir(parents=True, exist_ok=True)
    with args.events.open("x") as stream:
        os.chmod(args.events, 0o600)

        def emit(event: dict) -> None:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        for round_id, indices in (
            (1, manifest["schedule"]["107"]),
            ("short", [manifest["short_index"]]),
            ("long", [manifest["long_index"]]),
            (2, manifest["schedule"]["107"]),
        ):
            model.zero_grad(set_to_none=True)
            for i in indices:
                i = int(i)
                item = items[i]
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in collate([item], pad).items()
                }
                event = {
                    "round": round_id,
                    "row_sha256": manifest["rows"][str(i)]["row_sha256"],
                    "task_type": item["task_type"],
                    "tokens": len(item["ids"]),
                }
                emit({**event, "phase": "forward_start"})
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**batch)
                    terms = per_example_loss(
                        logits,
                        batch["labels"],
                        batch["candidate_mask"],
                        objective="ce_brier",
                        brier_weight=0.5,
                        teacher_probs=batch["teacher_probs"],
                        replay_mask=batch["replay_mask"],
                    )
                    loss = terms["total"].sum() / (
                        ACCUMULATION if isinstance(round_id, int) else 1
                    )
                torch.cuda.synchronize(device)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                emit(
                    {
                        **event,
                        "phase": "backward_start",
                        "loss": float(loss.detach().cpu()),
                    }
                )
                loss.backward()
                torch.cuda.synchronize(device)
                emit({**event, "phase": "backward_complete"})
            grads = [
                p.grad
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise RuntimeError("Nonfinite or missing trainable gradient")
            emit(
                {
                    "round": round_id,
                    "phase": "round_complete",
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                }
            )
    after = {
        name: hashlib.sha256(
            parameter.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if original != after:
        raise RuntimeError("Probe changed trainable parameters")
    return {
        "status": "NARROW_PASS",
        "manifest_sha256": fingerprint,
        "events_sha256": sha256(args.events),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "trainable_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "replay"))
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--console", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--events", type=Path)
    args = parser.parse_args()
    if args.mode == "replay" and args.events is None:
        parser.error("replay requires --events")
    result = audit(args) if args.mode == "audit" else replay(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
