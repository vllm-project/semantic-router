"""Run one preregistered 0.6B arm from a frozen JSON spec.

Families: `kai-native` continues a pinned Kai/Lex bundle through its own
all-types training API at the native 8,192 cap; `encoder` trains an official
bidirectional encoder with the marker readout in `encoder.py`. Both see the
same rights-clean v2 rows through the published System One converter, in one
fixed hash order, with SELECT at fixed milestones and the Qwen-control BEST
rule. `--preflight` runs step-0 SELECT, one update, export and a fresh reload
parity check, then stops.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

from . import encoder as enc
from . import kai8k
from .common import (
    MAX_INPUT_TOKENS,
    better,
    file_sha256,
    learning_rate,
    load_rights_clean,
    metric_summary,
    native_keys,
    native_records,
    original_probabilities,
    read_jsonl,
    schedule,
    select_record,
    write_json,
    write_jsonl,
)

SPEC_KEYS = {
    "arm",
    "family",
    "start",
    "data",
    "logical_batch",
    "max_micro_rows",
    "micro_token_budget",
    "seed",
    "optimizer",
    "loss",
    "teacher",
    "milestones",
    "include_step0_in_best",
    "gpu_hour_cap",
    "max_reserved_gib",
}
PARITY_ROWS = 32


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    if set(spec) != SPEC_KEYS:
        raise ValueError(f"Arm spec keys differ: {sorted(set(spec) ^ SPEC_KEYS)}")
    if spec["family"] not in ("kai-native", "encoder"):
        raise ValueError("Unknown family")
    return spec


def target_vector(record: dict[str, Any], ids: list[str]) -> list[float]:
    gold = record["target"]
    kind = record["question"]["type"].lower()
    if kind == "noul":
        return [1 - float(gold["probability"]), float(gold["probability"])]
    if "probabilities" in gold:
        return [float(v) for v in gold["probabilities"]]
    return [float(i == gold["choice_id"]) for i in ids]


def micro_batches(
    indices: list[int], kinds: list[str], lengths: list[int], max_rows: int, budget: int
) -> list[list[int]]:
    """Type-homogeneous, length-sorted micro-batches under a padded-token budget."""
    batches = []
    for kind in ("choice", "noul", "score"):
        group = sorted(
            (i for i in indices if kinds[i] == kind), key=lambda i: (-lengths[i], i)
        )
        current: list[int] = []
        for i in group:
            if current and (
                len(current) + 1 > max_rows
                or (len(current) + 1) * lengths[current[0]] > budget
            ):
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)
    return batches


def extra_terms(
    logits: Any,
    targets: Any,
    valid: Any,
    teacher: Any | None,
    brier_weight: float,
    kl_weight: float,
) -> tuple[Any, Any, Any]:
    import torch

    logp = torch.log_softmax(logits.float(), -1)
    p = logp.exp()
    brier = ((p - targets).square() * valid).sum(-1)
    if teacher is None or kl_weight == 0:
        kl = torch.zeros_like(brier)
    else:
        safe = torch.where(teacher > 0, teacher, torch.ones_like(teacher))
        kl = (teacher * (safe.log() - logp) * valid * (teacher > 0)).sum(-1)
    return brier_weight * brier + kl_weight * kl, brier, kl


class KaiFamily:
    def __init__(self, spec: dict[str, Any], device: str):
        self.spec = spec
        self.bundle = Path(spec["start"]["bundle"])
        self.backend = spec["start"]["backend"]
        self.native, self.identity = kai8k.load(
            self.bundle, self.backend, device=device
        )
        import decision_runtime as api

        self.api = api
        self.api.configure_training(self.native, max_input_tokens=MAX_INPUT_TOKENS)
        self.frozen = self.api.frozen_snapshot(self.native)

    def length(self, record: dict[str, Any]) -> int:
        return self.native.collator.encode(record, labeled=True)["input_tokens"]

    def parameter_groups(self) -> list[dict[str, Any]]:
        o = self.spec["optimizer"]
        return self.api.optimizer_groups(
            self.native, encoder_lr=o["encoder_lr"], head_lr=o["head_lr"]
        )

    def train_mode(self) -> None:
        self.api.set_training_mode(self.native, True)

    def loss(
        self,
        records: list[dict[str, Any]],
        teacher: list[list[float] | None],
        device: str,
    ) -> tuple[Any, dict[str, float]]:
        import torch

        batch, _ = self.native.collator(
            [dict(r) for r in records], labeled=True, device=device
        )
        loss_cfg = self.spec["loss"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.api.forward_for_training(self.native, batch)
            total, ce, rps = self.api.loss_for_training(
                self.native, logits, batch, score_rps_weight=loss_cfg["score_rps"]
            )
        q = None
        if any(t is not None for t in teacher):
            q = torch.zeros_like(batch["targets"])
            for i, t in enumerate(teacher):
                if t is not None:
                    q[i, : len(t)] = torch.tensor(t, device=q.device)
        extra, brier, kl = extra_terms(
            logits,
            batch["targets"],
            batch["valid_candidates"].float(),
            q,
            loss_cfg["brier"],
            loss_cfg["teacher_kl"],
        )
        per_row = total.float() + extra
        return per_row.sum(), {
            "ce": float(ce.detach().sum()),
            "brier": float(brier.detach().sum()),
            "kl": float(kl.detach().sum()),
        }

    def probabilities(self, records: list[dict[str, Any]]) -> list[list[float]]:
        self.api.restore_native_policy_for_export(self.native)
        try:
            return kai8k.record_probabilities(self.native, records)
        finally:
            self.api.configure_training(self.native, max_input_tokens=MAX_INPUT_TOKENS)

    def state(self) -> dict[str, Any]:
        return {
            k: v.detach().cpu().contiguous()
            for k, v in self.native.model.state_dict().items()
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.native.model.load_state_dict(state, strict=True)
        self.api.assert_frozen(self.native, self.frozen, content=True, versions=False)

    def export(self, output: Path, provenance: dict[str, Any]) -> str:
        self.api.restore_native_policy_for_export(self.native)
        self.api.export_native(self.native, output, provenance=provenance)
        self.api.configure_training(self.native, max_input_tokens=MAX_INPUT_TOKENS)
        return file_sha256(output / "MANIFEST.json")

    def reload_probabilities(
        self, output: Path, manifest: str, records: list[dict[str, Any]], device: str
    ) -> list[list[float]]:
        native, _ = kai8k.load(
            self.bundle,
            self.backend,
            native_dir=output,
            manifest_sha256=manifest,
            device=device,
        )
        return kai8k.record_probabilities(native, records)

    def parameters(self) -> int:
        return sum(p.numel() for p in self.native.model.parameters())


class EncoderFamily:
    def __init__(self, spec: dict[str, Any], device: str):
        self.spec = spec
        self.source_path = Path(spec["start"]["path"])
        self.device = device
        self.model, self.packer, self.metadata = enc.from_official(
            self.source_path,
            spec["start"]["source"],
            seed=int(spec["start"]["head_seed"]),
        )
        self.model.to(device)
        if hasattr(self.model.backbone, "gradient_checkpointing_enable"):
            self.model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.identity = self.metadata["source"]

    def length(self, record: dict[str, Any]) -> int:
        return self.packer.encode(record)["input_tokens"]

    def parameter_groups(self) -> list[dict[str, Any]]:
        o = self.spec["optimizer"]
        return [
            {
                "name": "backbone.encoder",
                "params": [
                    p for p in self.model.backbone.parameters() if p.requires_grad
                ],
                "lr": o["encoder_lr"],
            },
            {
                "name": "shared.head",
                "params": list(self.model.head.parameters()),
                "lr": o["head_lr"],
            },
        ]

    def train_mode(self) -> None:
        self.model.train()

    def loss(
        self,
        records: list[dict[str, Any]],
        teacher: list[list[float] | None],
        device: str,
    ) -> tuple[Any, dict[str, float]]:
        import torch

        encoded = [self.packer.encode(r) for r in records]
        batch = self.packer.collate(encoded, device)
        targets = torch.zeros(
            batch["valid_candidates"].shape, dtype=torch.float32, device=device
        )
        for i, (record, e) in enumerate(zip(records, encoded)):
            targets[i, : len(e["candidate_ids"])] = torch.tensor(
                target_vector(record, e["candidate_ids"]), device=device
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(batch)
        valid = batch["valid_candidates"].float()
        logp = torch.log_softmax(logits.float(), -1)
        ce = -(targets * logp * valid).sum(-1)
        loss_cfg = self.spec["loss"]
        if loss_cfg["score_rps"]:
            raise ValueError("Score RPS is not part of the encoder recipe")
        q = None
        if any(t is not None for t in teacher):
            raise ValueError("Teacher KL is not part of the encoder recipe")
        extra, brier, kl = extra_terms(
            logits, targets, valid, q, loss_cfg["brier"], loss_cfg["teacher_kl"]
        )
        return (loss_cfg["ce"] * ce + extra).sum(), {
            "ce": float(ce.detach().sum()),
            "brier": float(brier.detach().sum()),
            "kl": 0.0,
        }

    def probabilities(self, records: list[dict[str, Any]]) -> list[list[float]]:
        return enc.probabilities(self.model, self.packer, records, device=self.device)

    def state(self) -> dict[str, Any]:
        return {
            k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state, strict=True)

    def export(self, output: Path, provenance: dict[str, Any]) -> str:
        return enc.save(
            self.model, self.packer, self.metadata, self.source_path, output, provenance
        )

    def reload_probabilities(
        self, output: Path, manifest: str, records: list[dict[str, Any]], device: str
    ) -> list[list[float]]:
        model, packer, _ = enc.load(output, manifest, device=device)
        return enc.probabilities(model, packer, records, device=device)

    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def evaluate(
    family: Any, rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[list[float]]]:
    native = family.probabilities(records)
    outcomes, original = [], []
    for row, record, p in zip(rows, records, native):
        mapped = original_probabilities(row, native_keys(record), p)
        original.append(mapped)
        outcomes.append(select_record(row, mapped))
    return metric_summary(outcomes), original


def milestone_steps(spec: dict[str, Any], total: int) -> list[int]:
    if spec["milestones"] == "eighths":
        return sorted({max(1, round(total * k / 8)) for k in range(1, 9)})
    steps = sorted(set(int(s) for s in spec["milestones"]))
    if steps[-1] != total or steps[0] < 1:
        raise ValueError("Milestones must be positive and end at the final update")
    return steps


def environment() -> dict[str, Any]:
    import numpy
    import tokenizers
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": numpy.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run(
    spec_path: Path, output: Path, *, preflight: bool, device: str = "cuda:0"
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    started = time.monotonic()
    spec = load_spec(spec_path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.cuda.set_per_process_memory_fraction(
        min(
            1.0,
            spec["max_reserved_gib"]
            * 2**30
            / torch.cuda.get_device_properties(0).total_memory,
        )
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed_int = (
        int(spec["seed"].split("-")[0])
        if spec["seed"].split("-")[0].isdigit()
        else 20260928
    )
    torch.manual_seed(seed_int)

    splits = load_rights_clean(spec["data"]["parent"])
    records = {
        role: native_records(splits[role], spec["data"]["converter_bundle"])
        for role in ("train", "select")
    }
    family = (
        KaiFamily(spec, device)
        if spec["family"] == "kai-native"
        else EncoderFamily(spec, device)
    )

    lengths = {role: [family.length(r) for r in records[role]] for role in records}
    over = {role: sum(v > MAX_INPUT_TOKENS for v in lengths[role]) for role in lengths}
    if any(over.values()):
        raise ValueError(
            f"Rows exceed the complete-input cap; freeze a quarantine first: {over}"
        )
    kinds = [r["question"]["type"].lower() for r in records["train"]]

    teacher: list[list[float] | None] = [None] * len(records["train"])
    if spec["teacher"] is not None:
        path = Path(spec["teacher"]["path"])
        if file_sha256(path) != spec["teacher"]["sha256"]:
            raise ValueError("Teacher file differs from its frozen hash")
        by_id = {row["source_row_id"]: row["probabilities"] for row in read_jsonl(path)}
        teacher = [by_id[r["source_row_id"]] for r in records["train"]]
        for record, q in zip(records["train"], teacher):
            if len(q) != len(native_keys(record)) or abs(sum(q) - 1) > 1e-4:
                raise ValueError("Teacher vector does not match native candidates")

    plan = schedule(
        [r["source_row_id"] for r in records["train"]],
        spec["logical_batch"],
        spec["seed"],
    )
    total = len(plan)
    steps = milestone_steps(spec, total)
    warmup = int(total * spec["optimizer"]["warmup_ratio"])
    groups = family.parameter_groups()
    optimizer = torch.optim.AdamW(
        [{"params": g["params"], "lr": g["lr"], "name": g["name"]} for g in groups],
        weight_decay=spec["optimizer"]["weight_decay"],
        foreach=False,
        fused=False,
    )
    base = {
        g["name"]: (
            spec["optimizer"]["encoder_lr"]
            if g["name"].endswith("encoder")
            else spec["optimizer"]["head_lr"]
        )
        for g in groups
    }
    identity = {
        "spec_sha256": file_sha256(spec_path),
        "spec": spec,
        "start": family.identity,
        "loaded_parameters": family.parameters(),
        "train_rows": len(records["train"]),
        "select_rows": len(records["select"]),
        "train_tokens": sum(lengths["train"]),
        "max_train_tokens": max(lengths["train"]),
        "select_tokens": sum(lengths["select"]),
        "total_updates": total,
        "warmup_updates": warmup,
        "milestones": steps,
        "environment": environment(),
        "preflight": preflight,
    }
    write_json(output / "RUN.json", identity, exclusive=True)

    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    def milestone(step: int) -> None:
        nonlocal best
        metrics, probs = evaluate(family, splits["select"], records["select"])
        point = {
            "step": step,
            "metrics": metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        curve.append(point)
        write_jsonl(
            output / f"select-{step:04d}.jsonl",
            [
                {"id": r["id"], "probabilities": p}
                for r, p in zip(splits["select"], probs)
            ],
        )
        eligible = step > 0 or spec["include_step0_in_best"]
        if eligible and better(point, best):
            best = {"step": step, "metrics": metrics}
            save_file(family.state(), str(output / "best.safetensors"))
            write_json(
                output / "BEST.json",
                {**best, "state_sha256": file_sha256(output / "best.safetensors")},
            )
        write_json(output / "CURVE.json", curve)

    milestone(0)
    log = (output / "TRAIN_LOG.jsonl").open("x", encoding="utf-8")
    last_update = total if not preflight else 1
    for step in range(1, last_update + 1):
        if (time.monotonic() - started) / 3600 > spec["gpu_hour_cap"]:
            raise RuntimeError("GPU-hour cap reached; arm stopped")
        family.train_mode()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(
                step, total, warmup, base[group["name"]], spec["optimizer"]["lr_min"]
            )
        optimizer.zero_grad(set_to_none=True)
        rows = plan[step - 1]
        sums = {"ce": 0.0, "brier": 0.0, "kl": 0.0, "loss": 0.0}
        tokens = 0
        for micro in micro_batches(
            rows,
            kinds,
            lengths["train"],
            spec["max_micro_rows"],
            spec["micro_token_budget"],
        ):
            loss, parts = family.loss(
                [records["train"][i] for i in micro],
                [teacher[i] for i in micro],
                device,
            )
            (loss / len(rows)).backward()
            for key, value in parts.items():
                sums[key] += value
            sums["loss"] += float(loss.detach())
            tokens += sum(lengths["train"][i] for i in micro)
        params = [p for g in optimizer.param_groups for p in g["params"]]
        for p in params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"Nonfinite gradient at update {step}")
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                params, spec["optimizer"]["clip"], error_if_nonfinite=True
            )
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log.write(
            json.dumps(
                {
                    "step": step,
                    "rows": len(rows),
                    "tokens": tokens,
                    "grad_norm": norm,
                    "lr": {g["name"]: g["lr"] for g in optimizer.param_groups},
                    **{k: v / len(rows) for k, v in sums.items()},
                }
            )
            + "\n"
        )
        log.flush()
        if not math.isfinite(sums["loss"]):
            raise FloatingPointError(f"Nonfinite loss at update {step}")
        if not preflight and step in steps:
            milestone(step)
    log.close()

    if preflight:
        _, in_process = evaluate(
            family, splits["select"][:PARITY_ROWS], records["select"][:PARITY_ROWS]
        )
        manifest = family.export(
            output / "preflight-export", {"arm": spec["arm"], "preflight_step": 1}
        )
        reloaded_native = family.reload_probabilities(
            output / "preflight-export",
            manifest,
            records["select"][:PARITY_ROWS],
            device,
        )
        reloaded = [
            original_probabilities(r, native_keys(n), p)
            for r, n, p in zip(splits["select"], records["select"], reloaded_native)
        ]
        drift = max(
            abs(a - b) for x, y in zip(in_process, reloaded) for a, b in zip(x, y)
        )
        changed = sum(
            select_record(r, x)["chosen"] != select_record(r, y)["chosen"]
            for r, x, y in zip(splits["select"], in_process, reloaded)
        )
        result = {
            "status": (
                "PREFLIGHT_PASS" if drift <= 1e-5 and changed == 0 else "PREFLIGHT_FAIL"
            ),
            "zero_step_select": curve[0]["metrics"],
            "one_update_finite": True,
            "export_manifest_sha256": manifest,
            "reload_parity_rows": PARITY_ROWS,
            "reload_max_abs_drift": drift,
            "reload_category_changes": changed,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output / "PREFLIGHT.json", result)
        return result

    if best is None:
        raise RuntimeError("No eligible BEST checkpoint")
    family.restore(load_file(str(output / "best.safetensors")))
    manifest = family.export(
        output / "best-export",
        {"arm": spec["arm"], "best": best, "spec_sha256": identity["spec_sha256"]},
    )
    reference = read_jsonl(output / f"select-{best['step']:04d}.jsonl")
    reloaded_native = family.reload_probabilities(
        output / "best-export", manifest, records["select"], device
    )
    reloaded = [
        original_probabilities(r, native_keys(n), p)
        for r, n, p in zip(splits["select"], records["select"], reloaded_native)
    ]
    drift = max(
        abs(a - b)
        for ref, y in zip(reference, reloaded)
        for a, b in zip(ref["probabilities"], y)
    )
    changed = sum(
        select_record(r, ref["probabilities"])["chosen"]
        != select_record(r, y)["chosen"]
        for r, ref, y in zip(splits["select"], reference, reloaded)
    )
    result = {
        "status": (
            "COMPLETE"
            if drift <= 1e-5 and changed == 0
            else "COMPLETE_RELOAD_PARITY_FAIL"
        ),
        "best": best,
        "best_export_manifest_sha256": manifest,
        "reload_parity_rows": len(reference),
        "reload_max_abs_drift": drift,
        "reload_category_changes": changed,
        "curve_steps": [p["step"] for p in curve],
        "elapsed_seconds": time.monotonic() - started,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    write_json(output / "COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(
        json.dumps(
            run(args.spec, args.output, preflight=args.preflight), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
