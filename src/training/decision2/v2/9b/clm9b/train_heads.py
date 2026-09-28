"""Train, select and calibrate one frozen-feature arm.

Every arm sees the same TRAIN rows, batch size, update budget, optimizer,
schedule, SELECT checkpoints and selector. SELECT chooses the checkpoint
(family-macro accuracy, then family-macro Brier, then earliest update) with
the arm's primary readouts at temperature 1: relative Choice/Noul and the
absolute ordinal Score head. CAL then fits per-readout temperatures once.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from . import pins
from .arms import ARMS, ArmModel
from .features import TASK_TYPES, FeatureSet
from .heads import ordinal_probabilities
from .objectives import bidirectional_infonce, ordinal_loss, replay_kl

TRAIN_VERSION = "decision2-9b-frozen-head-grid-v1"
BUDGET = {
    "batch_rows": 128,
    "updates": 1200,
    "select_every": 100,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "warmup_fraction": 0.1,
    "clip_norm": 1.0,
    "brier_weight": 0.5,
    "replay_weight": 0.5,
}


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def load_teacher(path: Path, features: FeatureSet) -> tuple[torch.Tensor, torch.Tensor]:
    by_id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("valid"):
            by_id[item["id"]] = item
    width = features.candidate_index.shape[1]
    probs = torch.zeros((len(features), width))
    mask = torch.zeros(len(features), dtype=torch.bool)
    for i, row in enumerate(features.rows):
        item = by_id.get(row["id"])
        if item is None:
            continue
        if item["keys"] != row["keys"]:
            raise ValueError(f"{row['id']}: teacher option order differs")
        probs[i, : len(item["keys"])] = torch.tensor(item["probabilities"])
        mask[i] = True
    return probs.to(features.device), mask.to(features.device)


def arm_loss(model: ArmModel, batch, table, teacher=None) -> dict[str, torch.Tensor]:
    objective = model.spec["objective"]
    labels, mask = batch["labels"], batch["candidate_mask"]
    terms: dict[str, torch.Tensor] = {}
    relative = torch.zeros((), device=labels.device)
    if objective == "ce_brier":
        from training.model.loss import per_example_loss

        logits = model.relative_logits(batch)
        per = per_example_loss(
            logits,
            labels,
            mask,
            objective="ce_brier",
            brier_weight=BUDGET["brier_weight"],
        )
        relative = per["total"].mean()
    elif objective.startswith("infonce"):
        head = model.head
        state = head.encode_state(batch["query"])
        ids = batch["candidate_ids"]
        gold_ids = ids.gather(1, labels[:, None]).squeeze(1)
        pool_ids, gold_pool = torch.unique(gold_ids, return_inverse=True)
        pool = head.encode_action(table[pool_ids])
        same_as_gold = ids == gold_ids[:, None]
        distractor = mask & ~same_as_gold
        own_pool = (
            (ids[:, :, None] == pool_ids[None, None, :]) & distractor[..., None]
        ).any(1)
        hard = objective != "infonce"
        loss = bidirectional_infonce(
            state,
            pool,
            head.scale(),
            gold_pool,
            own_pool,
            head.encode_action(batch["candidates"]) if hard else None,
            distractor if hard else None,
        )
        relative = loss["total"]
        terms.update(
            infonce_forward=loss["forward"].detach(),
            infonce_backward=loss["backward"].detach(),
        )
        if objective.endswith("replay"):
            probs, has = teacher
            rows = has[batch["rows"]]
            if torch.any(rows):
                logits = model.relative_logits(batch)
                width = logits.shape[1]
                kl = replay_kl(
                    logits[rows], mask[rows], probs[batch["rows"]][rows][:, :width]
                )
                relative = relative + BUDGET["replay_weight"] * kl
                terms["replay_kl"] = kl.detach()
    terms["relative"] = relative.detach()
    score_rows = batch["task_type"] == TASK_TYPES.index("score")
    score = torch.zeros((), device=labels.device)
    if torch.any(score_rows):
        cumulative, counts, gold_level, order = model.score_cumulative(
            batch, score_rows
        )
        teacher_levels, weight = None, 0.0
        if objective.endswith("replay"):
            probs, has = teacher
            rows = batch["rows"][score_rows]
            if torch.all(has[rows]):
                width = order.shape[1]
                teacher_levels = probs[rows][:, :width].gather(1, order)
                weight = BUDGET["replay_weight"]
        ordinal = ordinal_loss(
            cumulative,
            counts,
            gold_level,
            BUDGET["brier_weight"],
            teacher_levels,
            weight,
        )
        score = ordinal["total"]
        terms.update(
            score_nll=ordinal["nll"].detach(),
            score_replay_kl=ordinal["replay_kl"].detach(),
        )
    terms["score"] = score.detach()
    return {"relative": relative, "score": score, **terms}


@torch.no_grad()
def collect(
    model: ArmModel, features: FeatureSet, batch_rows: int = 256
) -> list[dict[str, Any]]:
    """Raw logits for every row; invalid (over-length) rows are kept as invalid."""
    model.eval()
    records = []
    valid = torch.nonzero(features.valid).flatten()
    output: dict[int, dict[str, Any]] = {}
    for start in range(0, len(valid), batch_rows):
        indices = valid[start : start + batch_rows]
        batch = features.batch(indices)
        logits = model.relative_logits(batch)
        score_rows = batch["task_type"] == TASK_TYPES.index("score")
        cumulative = None
        if torch.any(score_rows):
            cumulative, counts, _, _ = model.score_cumulative(batch, score_rows)
            cumulative_rows = torch.nonzero(score_rows).flatten().tolist()
        for position, row_index in enumerate(indices.tolist()):
            k = int(batch["counts"][position])
            item = {"relative": logits[position, :k].tolist()}
            if cumulative is not None and bool(score_rows[position]):
                slot = cumulative_rows.index(position)
                item["cumulative"] = cumulative[slot, : k - 1].tolist()
            output[row_index] = item
    for index, row in enumerate(features.rows):
        record = {
            "index": index,
            "id": row["id"],
            "family": row.get("family"),
            "task_type": row["task_type"],
            "keys": row["keys"],
            "label": row["label"],
            "valid": index in output,
            **output.get(index, {}),
        }
        records.append(record)
    model.train()
    return records


def softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [v / temperature for v in values]
    top = max(scaled)
    weights = [math.exp(v - top) for v in scaled]
    total = sum(weights)
    return [w / total for w in weights]


def ordinal_from(cumulative: list[float], temperature: float) -> list[float]:
    tensor = torch.tensor([cumulative], dtype=torch.float64)
    counts = torch.tensor([len(cumulative) + 1])
    return ordinal_probabilities(tensor, counts, temperature)[0].tolist()


def option_probabilities(
    record: dict[str, Any], temperatures: dict[str, float], score_readout: str
) -> list[float]:
    """Probabilities in the row's option order for the requested readout."""
    kind = record["task_type"]
    if kind == "score" and score_readout == "absolute":
        by_level = ordinal_from(record["cumulative"], temperatures["score_absolute"])
        return [by_level[int(key)] for key in record["keys"]]
    key = "score_relative" if kind == "score" else kind
    return softmax(record["relative"], temperatures[key])


def point(record: dict[str, Any], probabilities: list[float]) -> int | None:
    if record["task_type"] == "noul":
        p_true = probabilities[record["keys"].index("true")]
        return (
            None
            if p_true == 0.5
            else record["keys"].index("true" if p_true > 0.5 else "false")
        )
    top = max(probabilities)
    winners = [i for i, value in enumerate(probabilities) if abs(value - top) <= 1e-8]
    return winners[0] if len(winners) == 1 else None


def select_records(
    records, temperatures, score_readout="absolute"
) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if not record["valid"]:
            out.append(
                {
                    "id": record["id"],
                    "family": record["family"],
                    "task_type": record["task_type"],
                    "correct": False,
                    "brier": 1.0,
                    "nll": -math.log(1e-12),
                }
            )
            continue
        probs = option_probabilities(record, temperatures, score_readout)
        chosen = point(record, probs)
        label = record["label"]
        out.append(
            {
                "id": record["id"],
                "family": record["family"],
                "task_type": record["task_type"],
                "correct": chosen == label,
                "brier": sum((p - float(i == label)) ** 2 for i, p in enumerate(probs))
                / 2,
                "nll": -math.log(max(probs[label], 1e-12)),
            }
        )
    return out


def summarize(records, temperatures, score_readout="absolute") -> dict[str, Any]:
    from training.model.train import metric_summary

    rows = select_records(records, temperatures, score_readout)
    summary = metric_summary(rows)
    summary["by_type"] = {}
    for kind in TASK_TYPES:
        subset = [row for row in rows if row["task_type"] == kind]
        if subset:
            summary["by_type"][kind] = {
                "n": len(subset),
                "correct": sum(row["correct"] for row in subset),
                "brier": sum(row["brier"] for row in subset) / len(subset),
                "nll": sum(row["nll"] for row in subset) / len(subset),
            }
    return summary


def ece10(pairs: list[tuple[float, bool]]) -> float:
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for confidence, hit in pairs:
        bins[min(9, int(confidence * 10))].append((confidence, hit))
    n = len(pairs)
    return sum(
        len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b))
        for b in bins
        if b
    )


def readout_metrics(records, temperatures, readout: str, kind: str) -> dict[str, Any]:
    subset = [r for r in records if r["valid"] and r["task_type"] == kind]
    nll = brier = correct = 0.0
    pairs = []
    for record in subset:
        probs = option_probabilities(record, temperatures, readout)
        label = record["label"]
        chosen = point(record, probs)
        correct += chosen == label
        nll += -math.log(max(probs[label], 1e-12))
        brier += sum((p - float(i == label)) ** 2 for i, p in enumerate(probs)) / 2
        pairs.append((max(probs), chosen == label))
    n = len(subset)
    return (
        {
            "n": n,
            "accuracy": correct / n,
            "nll": nll / n,
            "brier": brier / n,
            "ece_10": ece10(pairs),
        }
        if n
        else {"n": 0}
    )


def golden_temperature(objective) -> float:
    """Minimize a CAL objective over log-temperature in [0.05, 20]."""
    left, right = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = objective(math.exp(x1)), objective(math.exp(x2))
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(math.exp(x2))
    candidates = (1.0, 0.05, 20.0, math.exp((left + right) / 2))
    return min(candidates, key=lambda t: (objective(t), abs(math.log(t))))


def fit_calibration(records) -> dict[str, Any]:
    temperatures = {
        "choice": 1.0,
        "noul": 1.0,
        "score_relative": 1.0,
        "score_absolute": 1.0,
    }

    def nll_for(kind: str, readout: str, key: str):
        subset = [r for r in records if r["valid"] and r["task_type"] == kind]

        def objective(t: float) -> float:
            local = {**temperatures, key: t}
            return sum(
                -math.log(
                    max(option_probabilities(r, local, readout)[r["label"]], 1e-12)
                )
                for r in subset
            ) / len(subset)

        return objective

    for kind, readout, key in (
        ("choice", "relative", "choice"),
        ("noul", "relative", "noul"),
        ("score", "relative", "score_relative"),
        ("score", "absolute", "score_absolute"),
    ):
        temperatures[key] = golden_temperature(nll_for(kind, readout, key))
    report = {"temperature_by_readout": temperatures, "before": {}, "after": {}}
    unit = {key: 1.0 for key in temperatures}
    for kind, readout in (
        ("choice", "relative"),
        ("noul", "relative"),
        ("score", "relative"),
        ("score", "absolute"),
    ):
        name = f"{kind}_{readout}" if kind == "score" else kind
        report["before"][name] = readout_metrics(records, unit, readout, kind)
        report["after"][name] = readout_metrics(records, temperatures, readout, kind)
    return report


def head_state(model: ArmModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}


def save_head(path: Path, model: ArmModel) -> str:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in head_state(model).items()}, str(path))
    return pins.file_sha256(path)


def load_head(path: Path, arm: str, device) -> ArmModel:
    from safetensors.torch import load_file

    model = ArmModel(arm)
    model.load_state_dict(load_file(str(path)), strict=True)
    return model.to(device)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train(args) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = ARMS[args.arm]
    representation = spec["representation"]
    features = {
        split: FeatureSet(args.features / split, args.layer, representation, device)
        for split in ("train", "select", "cal")
    }
    manifest = json.loads((args.features / "manifest.json").read_text(encoding="utf-8"))
    teacher = None
    if spec["objective"].endswith("replay"):
        if args.teacher is None:
            raise SystemExit("Replay arm needs --teacher")
        teacher = load_teacher(args.teacher, features["train"])
        if not bool(teacher[1][features["train"].valid].all()):
            raise SystemExit("Teacher does not cover every valid TRAIN row")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "version": TRAIN_VERSION,
        "arm": args.arm,
        "arm_spec": spec,
        "layer": args.layer,
        "seed": args.seed,
        "budget": BUDGET,
        "source": manifest["source"],
        "features_manifest_sha256": pins.file_sha256(args.features / "manifest.json"),
        "feature_files": {
            split: manifest["inputs"][split]["files_sha256"] for split in features
        },
        "teacher_sha256": (
            pins.file_sha256(args.teacher) if teacher is not None else None
        ),
        "code_commit": args.code_commit,
        "preflight": args.preflight,
    }
    write_json(args.output / "config.json", config)
    torch.manual_seed(args.seed)
    model = ArmModel(args.arm).to(device)
    groups = [
        {"params": [p for p in model.head.parameters()], "name": "relative"},
        {"params": [p for p in model.score.parameters()], "name": "score"},
    ]
    groups = [g for g in groups if g["params"]]
    optimizer = torch.optim.AdamW(
        groups, lr=BUDGET["lr"], weight_decay=BUDGET["weight_decay"]
    )
    updates = 1 if args.preflight else BUDGET["updates"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=BUDGET["lr"],
        total_steps=max(updates, 2),
        pct_start=BUDGET["warmup_fraction"],
        anneal_strategy="cos",
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_rows = torch.nonzero(features["train"].valid).flatten().cpu()
    stream: list[int] = []
    unit = {"choice": 1.0, "noul": 1.0, "score_relative": 1.0, "score_absolute": 1.0}
    log = (args.output / "train-log.jsonl").open("x", encoding="utf-8")
    started = time.perf_counter()
    select_zero = summarize(collect(model, features["select"]), unit)
    log.write(json.dumps({"step": 0, "select": select_zero}) + "\n")
    best = {
        "step": 0,
        "macro": select_zero["family_macro_accuracy"],
        "brier": select_zero["family_macro_brier"],
        "state": head_state(model),
        "summary": select_zero,
    }
    has_arm_params = any(g["name"] == "relative" for g in groups)
    for step in range(1, updates + 1):
        while len(stream) < BUDGET["batch_rows"]:
            stream.extend(
                train_rows[
                    torch.randperm(len(train_rows), generator=generator)
                ].tolist()
            )
        indices, stream = (
            torch.tensor(stream[: BUDGET["batch_rows"]]),
            stream[BUDGET["batch_rows"] :],
        )
        batch = features["train"].batch(indices)
        terms = arm_loss(model, batch, features["train"].table, teacher)
        loss = terms["relative"] + terms["score"]
        if not torch.isfinite(loss):
            write_json(
                args.output / "FAILED.json", {"step": step, "reason": "nonfinite loss"}
            )
            raise SystemExit(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        if loss.requires_grad:
            loss.backward()
        norms = {}
        for group in groups:
            norms[group["name"]] = float(
                torch.nn.utils.clip_grad_norm_(group["params"], BUDGET["clip_norm"])
            )
        if not all(math.isfinite(v) for v in norms.values()):
            write_json(
                args.output / "FAILED.json",
                {"step": step, "reason": "nonfinite gradient"},
            )
            raise SystemExit(f"Nonfinite gradient at step {step}")
        optimizer.step()
        scheduler.step()
        event = {
            "step": step,
            "loss": float(loss),
            **{k: float(v) for k, v in terms.items() if k not in ("relative_tensor",)},
            "grad_norm": norms,
        }
        if step % BUDGET["select_every"] == 0 or step == updates:
            summary = summarize(collect(model, features["select"]), unit)
            event["select"] = summary
            key = (
                -summary["family_macro_accuracy"],
                summary["family_macro_brier"],
                step,
            )
            if key < (-best["macro"], best["brier"], best["step"]):
                best = {
                    "step": step,
                    "macro": summary["family_macro_accuracy"],
                    "brier": summary["family_macro_brier"],
                    "state": head_state(model),
                    "summary": summary,
                }
        log.write(json.dumps(event) + "\n")
        log.flush()
    log.close()
    model.load_state_dict(best["state"])
    head_sha = save_head(args.output / "head.safetensors", model)
    select_records_best = collect(model, features["select"])
    reloaded = load_head(args.output / "head.safetensors", args.arm, device)
    reload_records = collect(reloaded, features["select"])
    drift = max(
        max(
            (abs(a - b) for a, b in zip(x.get("relative", []), y.get("relative", []))),
            default=0.0,
        )
        + max(
            (
                abs(a - b)
                for a, b in zip(x.get("cumulative", []), y.get("cumulative", []))
            ),
            default=0.0,
        )
        for x, y in zip(select_records_best, reload_records)
    )
    cal_records = collect(model, features["cal"])
    calibration = fit_calibration(cal_records)
    calibration.update(
        head_sha256=head_sha,
        cal_rows_sha256=manifest["inputs"]["cal"]["files_sha256"]["rows.jsonl"],
        fit_split="cal",
    )
    write_json(args.output / "calibration.json", calibration)
    with (args.output / "select-best-logits.jsonl").open(
        "x", encoding="utf-8"
    ) as stream_out:
        for record in select_records_best:
            stream_out.write(json.dumps(record) + "\n")
    with (args.output / "cal-logits.jsonl").open("x", encoding="utf-8") as stream_out:
        for record in cal_records:
            stream_out.write(json.dumps(record) + "\n")
    parameters = {
        "relative": sum(p.numel() for p in model.head.parameters()),
        "score_absolute": sum(p.numel() for p in model.score.parameters()),
    }
    best_record = {
        "arm": args.arm,
        "layer": args.layer,
        "seed": args.seed,
        "selected_step": best["step"],
        "selection": "SELECT family-macro accuracy desc, family-macro Brier asc, earliest update",
        "select": best["summary"],
        "select_score_relative": summarize(select_records_best, unit, "relative"),
        "select_zero_step": select_zero,
        "reload_max_abs_drift": drift,
        "head_sha256": head_sha,
        "calibration_sha256": pins.file_sha256(args.output / "calibration.json"),
        "head_parameters": parameters,
        "has_trained_relative_head": has_arm_params,
        "seconds": time.perf_counter() - started,
    }
    write_json(args.output / "BEST.json", best_record)
    print(
        json.dumps(
            {
                k: best_record[k]
                for k in (
                    "arm",
                    "layer",
                    "seed",
                    "selected_step",
                    "reload_max_abs_drift",
                    "seconds",
                )
            }
            | {"macro": best["macro"], "correct": best["summary"]["correct"]}
        ),
        flush=True,
    )
    if drift != 0.0:
        raise SystemExit(f"Reload drift {drift} is nonzero")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="extraction output with train/select/cal",
    )
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--layer", type=int, required=True, choices=pins.LAYERS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--preflight", action="store_true", help="zero-step + one update + reload only"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
