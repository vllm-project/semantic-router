"""Frozen-backbone decision heads on cached ~27B features (preregistered arm A3).

``train`` fits the preregistered grid (layer x head LR) on TRAIN features with
the native CE + 0.5 Brier loss, evaluates SELECT after every epoch with the
trainer's metric definitions and selects one head. ``readout`` fits per-type
CAL temperatures for the selected head and writes native-format, gold-free
DEV / CSS-pilot predictions for the unchanged scorers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from training.model.calibration import fit_report
from training.model.data import canonical, file_sha256
from training.model.decision_model import CandidateHead
from training.model.infer import load_prompts, normalized_answer, prompt_input_sha256
from training.model.infer import write_output
from training.model.loss import per_example_loss
from training.model.train import learning_factor, metric_summary

GRID_LAYERS = ("final", "three_quarter", "half")
GRID_LRS = (1e-4, 1e-3)
EPOCHS = 12
BATCH = 16
WARMUP = 0.05
WEIGHT_DECAY = 0.01
BRIER_WEIGHT = 0.5
SEED = 20260928
HEAD_DIM = 256
ADAPTER_VERSION = "decision2-27b-frozen-head-cached-features-v1"


class Split:
    def __init__(self, directory: Path, role: str, layer: str, device: torch.device):
        from safetensors.torch import load_file

        tensors = load_file(str(directory / f"{role}.features.safetensors"))
        self.rows = json.loads((directory / f"{role}.rows.json").read_text("utf-8"))
        self.offsets = tensors["candidate_offsets"].tolist()
        self.cand = tensors[f"cand_{layer}"].to(device)
        self.query = tensors[f"query_{layer}"].to(device)
        self.device = device

    def batch(self, indices: list[int]) -> dict[str, torch.Tensor]:
        width = max(len(self.rows[i]["keys"]) for i in indices)
        hidden = self.cand.shape[1]
        cand = torch.zeros(len(indices), width, hidden, device=self.device)
        mask = torch.zeros(len(indices), width, dtype=torch.bool, device=self.device)
        query = torch.stack([self.query[self.rows[i]["query_index"]] for i in indices])
        for slot, index in enumerate(indices):
            start, end = self.offsets[index], self.offsets[index + 1]
            cand[slot, : end - start] = self.cand[start:end]
            mask[slot, : end - start] = True
        labels = [self.rows[i]["label"] for i in indices]
        return {
            "cand": cand,
            "query": query,
            "mask": mask,
            "labels": (
                torch.tensor(labels, dtype=torch.long, device=self.device)
                if None not in labels
                else None
            ),
        }

    def live(self) -> list[int]:
        return [i for i, row in enumerate(self.rows) if row["query_index"] >= 0]


def logits_for(
    head: CandidateHead, split: Split, indices: list[int]
) -> list[list[float]]:
    out: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(indices), 64):
            chunk = indices[start : start + 64]
            batch = split.batch(chunk)
            values = head(batch["cand"], batch["query"]).float()
            for row, (index, vector) in enumerate(zip(chunk, values)):
                out.append(vector[: len(split.rows[index]["keys"])].cpu().tolist())
    return out


def select_records(split: Split, logits: list[list[float]]) -> list[dict[str, Any]]:
    """Mirror ``training.model.train.evaluate`` record semantics exactly."""
    records = []
    for row, values in zip(split.rows, logits):
        keys, target = row["keys"], row["label"]
        top = max(values)
        probabilities = [math.exp(v - top) for v in values]
        total = sum(probabilities)
        p = [value / total for value in probabilities]
        maximum = max(p)
        winners = [i for i, value in enumerate(p) if abs(value - maximum) <= 1e-8]
        chosen = winners[0] if len(winners) == 1 else None
        if row["task_type"] == "noul":
            p_true = p[keys.index("true")]
            chosen = (
                None
                if p_true == 0.5
                else keys.index("true" if p_true > 0.5 else "false")
            )
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "task_type": row["task_type"],
                "correct": chosen == target,
                "brier": sum((v - float(i == target)) ** 2 for i, v in enumerate(p))
                / 2,
                "nll": -math.log(max(p[target], 1e-12)),
            }
        )
    return records


def train(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    ledger = []
    for layer in GRID_LAYERS:
        train_split = Split(args.features, "train", layer, device)
        select_split = Split(args.features, "select", layer, device)
        hidden = train_split.cand.shape[1]
        indices = train_split.live()
        steps_per_epoch = math.ceil(len(indices) / BATCH)
        total = EPOCHS * steps_per_epoch
        for lr in GRID_LRS:
            torch.manual_seed(SEED)
            head = CandidateHead(hidden, HEAD_DIM).to(device).float()
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
            )
            name = f"{layer}-lr{lr:g}"
            (args.output / name).mkdir()
            step = 0
            for epoch in range(EPOCHS):
                head.train()
                generator = torch.Generator().manual_seed(SEED + epoch)
                order = [
                    indices[i]
                    for i in torch.randperm(len(indices), generator=generator).tolist()
                ]
                loss_sum = 0.0
                for start in range(0, len(order), BATCH):
                    batch = train_split.batch(order[start : start + BATCH])
                    for group in optimizer.param_groups:
                        group["lr"] = lr * learning_factor(step, total, WARMUP)
                    logits = head(batch["cand"], batch["query"]).masked_fill(
                        ~batch["mask"], -float("inf")
                    )
                    loss = per_example_loss(
                        logits,
                        batch["labels"],
                        batch["mask"],
                        objective="ce_brier",
                        brier_weight=BRIER_WEIGHT,
                    )["total"].mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"{name}: nonfinite loss at step {step}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(batch["labels"])
                    step += 1
                head.eval()
                metrics = metric_summary(
                    select_records(
                        select_split,
                        logits_for(
                            head, select_split, list(range(len(select_split.rows)))
                        ),
                    )
                )
                path = args.output / name / f"epoch-{epoch + 1:02d}.safetensors"
                save_file(
                    {
                        k: v.detach().float().cpu().contiguous()
                        for k, v in head.state_dict().items()
                    },
                    str(path),
                )
                entry = {
                    "config": name,
                    "layer": layer,
                    "lr": lr,
                    "epoch": epoch + 1,
                    "step": step,
                    "train_loss": loss_sum / len(order),
                    "select_correct": metrics["correct"],
                    "select_family_macro_accuracy": metrics["family_macro_accuracy"],
                    "select_family_macro_brier": metrics["family_macro_brier"],
                    "select_by_family": {
                        k: v["accuracy"] for k, v in metrics["by_family"].items()
                    },
                    "head_sha256": file_sha256(path),
                }
                ledger.append(entry)
                print(
                    json.dumps(
                        {
                            k: entry[k]
                            for k in (
                                "config",
                                "epoch",
                                "train_loss",
                                "select_correct",
                                "select_family_macro_accuracy",
                            )
                        }
                    ),
                    flush=True,
                )
        del train_split, select_split
    layer_rank = {name: rank for rank, name in enumerate(GRID_LAYERS)}
    best = min(
        ledger,
        key=lambda e: (
            -e["select_family_macro_accuracy"],
            e["select_family_macro_brier"],
            e["epoch"],
            e["lr"],
            layer_rank[e["layer"]],
        ),
    )
    result = {
        "schema_version": "decision2-27b-frozen-head-selection/1",
        "features_receipt_sha256": file_sha256(args.probe_receipt),
        "grid": {
            "layers": GRID_LAYERS,
            "lrs": GRID_LRS,
            "epochs": EPOCHS,
            "batch": BATCH,
            "warmup": WARMUP,
            "weight_decay": WEIGHT_DECAY,
            "brier_weight": BRIER_WEIGHT,
            "seed": SEED,
            "head_dim": HEAD_DIM,
        },
        "selection_rule": "SELECT family-macro accuracy desc, family-macro Brier asc, earlier epoch, lower LR, later layer",
        "selected": best,
        "ledger": ledger,
        "code_sha256": file_sha256(Path(__file__)),
    }
    (args.output / "SELECTION.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": {
                    k: best[k]
                    for k in (
                        "config",
                        "epoch",
                        "select_correct",
                        "select_family_macro_accuracy",
                        "select_family_macro_brier",
                    )
                }
            }
        )
    )


def readout(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selection = json.loads((args.selection_dir / "SELECTION.json").read_text("utf-8"))
    best = selection["selected"]
    head_path = (
        args.selection_dir / best["config"] / f"epoch-{best['epoch']:02d}.safetensors"
    )
    if file_sha256(head_path) != best["head_sha256"]:
        raise ValueError("Selected head bytes changed")
    state = load_file(str(head_path))
    head = (
        CandidateHead(state["key.weight"].shape[1], HEAD_DIM).to(device).float().eval()
    )
    head.load_state_dict(state, strict=True)
    probe = json.loads(args.probe_receipt.read_text("utf-8"))
    identity = {
        "adapter_version": ADAPTER_VERSION,
        "repo_id": probe["repo_id"],
        "revision": probe["revision"],
        "config_sha256": probe["config_sha256"],
        "probe_code_sha256": probe["code_sha256"],
        "layer": best["layer"],
        "head_sha256": best["head_sha256"],
    }
    model_sha256 = hashlib.sha256(canonical(identity).encode()).hexdigest()
    adapter_sha256 = hashlib.sha256(
        canonical(
            {
                "frozen_head.py": file_sha256(Path(__file__)),
                "probe": probe["code_sha256"],
            }
        ).encode()
    ).hexdigest()
    cal = Split(args.features, "cal", best["layer"], device)
    cal_logits = logits_for(head, cal, list(range(len(cal.rows))))
    report = fit_report(
        [
            {
                "id": r["id"],
                "task_type": r["task_type"],
                "logits": v,
                "label": r["label"],
            }
            for r, v in zip(cal.rows, cal_logits)
        ]
    )
    calibration = {
        "schema_version": "decision2-27b-frozen-head-calibration/1",
        "model_sha256": model_sha256,
        "cal_features_sha256": probe["features"]["splits"]["cal"]["features_sha256"],
        **report,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    calibration_path = args.output / "calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=1) + "\n", encoding="utf-8"
    )
    temperatures = report["temperature_by_type"]
    calibration_sha256 = file_sha256(calibration_path)
    for role, prompts_path in (("dev", args.dev), ("css_pilot", args.css_pilot)):
        split = Split(args.features, role, best["layer"], device)
        live = split.live()
        values = dict(zip(live, logits_for(head, split, live)))
        prompts = {item["id"]: item for item in load_prompts(prompts_path)}
        grouped: dict[str, dict[str, Any]] = {}
        counts = {
            "items": 0,
            "questions": 0,
            "valid_questions": 0,
            "invalid_questions": 0,
            "over_budget_questions": 0,
            "truncated_questions": 0,
        }
        for index, row in enumerate(split.rows):
            record = grouped.setdefault(
                row["prompt_id"], {"answers": {}, "errors": {}, "tokens": 0}
            )
            counts["questions"] += 1
            if row["query_index"] < 0:
                record["answers"][row["question_id"]] = {
                    "type": row["task_type"],
                    "error": "max_length_exceeded",
                }
                record["errors"][row["question_id"]] = "max_length_exceeded"
                counts["invalid_questions"] += 1
                counts["over_budget_questions"] += 1
                continue
            record["answers"][row["question_id"]] = normalized_answer(
                row["task_type"],
                row["keys"],
                values[index],
                temperatures[row["task_type"]],
            )
            record["tokens"] += row["tokens"]
            counts["valid_questions"] += 1
        predictions = []
        for prompt_id, item in prompts.items():
            record = grouped[prompt_id]
            counts["items"] += 1
            errors = record["errors"]
            predictions.append(
                {
                    "id": prompt_id,
                    "answers": record["answers"],
                    "latency_ms": None,
                    "usage": {"input_tokens": record["tokens"], "output_tokens": 0},
                    "adapter_status": (
                        "ok"
                        if not errors
                        else (
                            "invalid"
                            if len(errors) == len(item["questions"])
                            else "partial"
                        )
                    ),
                    "adapter_errors": errors,
                    "truncated_questions": 0,
                    "input_sha256": prompt_input_sha256(item),
                    "source_input_sha256": prompt_input_sha256(item),
                    "model_sha256": model_sha256,
                    "adapter_sha256": adapter_sha256,
                    "calibration_sha256": calibration_sha256,
                }
            )
        manifest = {
            "adapter_version": ADAPTER_VERSION,
            "model_id": args.model_id,
            "model_revision": f"{best['config']}-epoch{best['epoch']:02d}",
            "model_sha256": model_sha256,
            "model_identity": identity,
            "adapter_sha256": adapter_sha256,
            "input_sha256": file_sha256(prompts_path),
            "input_items": len(prompts),
            "max_length": probe["features"]["benchmark_limit"],
            "execution": "cached final/intermediate hidden states from the native one-question forward (FP32 parameters, BF16 autocast); FP32 head",
            "truncation_policy": "none; over-budget questions produce an invalid answer",
            "counts": counts,
            "calibration": {
                "file_sha256": calibration_sha256,
                "temperature_by_type": temperatures,
            },
        }
        write_output(
            args.output / f"{role.replace('_', '-')}.predictions.jsonl",
            predictions,
            manifest,
        )
        print(json.dumps({"role": role, **counts}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--probe-receipt", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    read = sub.add_parser("readout")
    read.add_argument("--features", type=Path, required=True)
    read.add_argument("--probe-receipt", type=Path, required=True)
    read.add_argument("--selection-dir", type=Path, required=True)
    read.add_argument("--dev", type=Path, required=True)
    read.add_argument("--css-pilot", type=Path, required=True)
    read.add_argument("--model-id", required=True)
    read.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.command == "train" else readout(args)


if __name__ == "__main__":
    main()
