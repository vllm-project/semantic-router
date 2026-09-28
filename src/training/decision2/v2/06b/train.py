"""Run one preregistered 0.6B arm from a frozen JSON spec.

Families: `kai-native` continues a pinned Kai/Lex bundle through its own
all-types training API at the native 8,192 cap; `encoder` trains an official
bidirectional encoder with the marker readout in `encoder.py`. Both see the
same rights-clean v2 rows through the published System One converter, in one
fixed hash order, with SELECT at fixed milestones and the Qwen-control BEST
rule. `--preflight` runs step-0 SELECT, the padded-versus-one-row micro-batch
parity gate (`parity.py`), one update, export and a fresh reload parity
check, then stops. `data.mixture` replaces the rights-clean TRAIN with a
template-S mixture of hash-pinned arms (`mixture.py`); SELECT and CAL stay
the rights-clean ones; a `materialized` mixture is a hash-pinned S2 build.
A teacher in `option-key` format (the canonical own-Lux file) applies only
where it saw the row's exact input. `optimizer.backbone_freeze_updates`
trains only the fresh head for that many updates, then starts the backbone
schedule.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

from . import encoder as enc
from . import kai8k
from . import mixture as mix
from . import parity
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
PADDING_PARITY_MICRO_BATCHES = 2


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    if set(spec) != SPEC_KEYS:
        raise ValueError(f"Arm spec keys differ: {sorted(set(spec) ^ SPEC_KEYS)}")
    if spec["family"] not in ("kai-native", "encoder", "qwen-causal"):
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
            ordinal_score=bool(spec["start"].get("ordinal_score", False)),
            candidate_pool=spec["start"].get("candidate_pool", "marker"),
            query_pool=spec["start"].get("query_pool", "first"),
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
                "params": list(self.model.head.parameters())
                + (
                    list(self.model.ordinal.parameters())
                    if self.model.ordinal is not None
                    else []
                ),
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
            q = torch.zeros_like(targets)
            for i, t in enumerate(teacher):
                if t is not None:
                    q[i, : len(t)] = torch.tensor(t, device=device)
        extra, brier, kl = extra_terms(
            logits, targets, valid, q, loss_cfg["brier"], loss_cfg["teacher_kl"]
        )
        weights = torch.tensor(
            [r.get("_loss_weight", 1.0) for r in records], device=device
        )
        return ((loss_cfg["ce"] * ce + extra) * weights).sum(), {
            "ce": float(ce.detach().sum()),
            "brier": float(brier.detach().sum()),
            "kl": float(kl.detach().sum()),
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


def native_from_original(
    row: dict[str, Any], probabilities: list[float]
) -> list[float]:
    """Inverse of `original_probabilities`: flattened option order -> native order."""
    if row["task_type"] != "noul":
        return list(probabilities)
    keys = [option["key"] for option in row["options"]]
    return [probabilities[keys.index("false")], probabilities[keys.index("true")]]


def option_key_teacher(
    rows: list[dict[str, Any]], path: Path
) -> tuple[list[list[float] | None], dict[str, int]]:
    """Option-key teacher rows (`id`, `input_sha256`, `teacher_probs`) -> native vectors.

    A row gets its teacher only if the teacher saw exactly its input (same
    `input_sha256`); renumbered option keys therefore drop the teacher term.
    """
    entries = {entry["id"]: entry for entry in read_jsonl(path)}
    out: list[list[float] | None] = []
    counts = {"covered": 0, "no_entry": 0, "input_changed": 0}
    for row in rows:
        entry = entries.get(row.get("teacher_source_id", row["id"]))
        if entry is None:
            counts["no_entry"] += 1
            out.append(None)
            continue
        if entry["input_sha256"] != row["input_sha256"]:
            counts["input_changed"] += 1
            out.append(None)
            continue
        keys = [option["key"] for option in row["options"]]
        if set(entry["teacher_probs"]) != set(keys):
            raise ValueError(f"{row['id']}: teacher keys differ from option keys")
        counts["covered"] += 1
        out.append(
            native_from_original(row, [float(entry["teacher_probs"][k]) for k in keys])
        )
    return out, counts


class CausalQwenFamily:
    """The official-Qwen causal endpoint/global-query model of the archived control.

    Reuses `training.model.decision_model` unchanged (renderer, collate, head,
    save/load); only the schedule, selection and optional teacher term are this
    trainer's, so causal and bidirectional arms share one trainer.
    """

    def __init__(
        self, spec: dict[str, Any], device: str, rows: dict[str, dict[str, Any]]
    ):
        import torch

        from training.model.decision_model import DecisionModel

        self.spec, self.device, self.rows = spec, device, rows
        path = Path(spec["start"]["path"])
        self.identity = enc.verify_source(path, "qwen3-0.6b-base")
        torch.manual_seed(int(spec["start"]["head_seed"]))
        self.model, self.tokenizer = DecisionModel.from_base(
            path, spec["start"]["revision"], head_dim=256
        )
        self.model.backbone.config.use_cache = False
        self.model.to(device)
        self.model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def encoded(self, record: dict[str, Any]) -> dict[str, Any]:
        from training.model.decision_model import encode

        return encode(
            self.rows[record["source_row_id"]], self.tokenizer, MAX_INPUT_TOKENS
        )

    def length(self, record: dict[str, Any]) -> int:
        return len(self.encoded(record)["ids"])

    def parameter_groups(self) -> list[dict[str, Any]]:
        o = self.spec["optimizer"]
        return [
            {
                "name": "backbone.encoder",
                "params": list(self.model.backbone.parameters()),
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

    def batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        from training.model.decision_model import collate

        items = [self.encoded(r) for r in records]
        batch = collate(items, self.tokenizer.pad_token_id)
        return {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()
        }

    def loss(
        self,
        records: list[dict[str, Any]],
        teacher: list[list[float] | None],
        device: str,
    ) -> tuple[Any, dict[str, float]]:
        import torch

        batch = self.batch(records)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**batch).float()
        valid = batch["candidate_mask"].float()
        logits = logits.masked_fill(
            ~batch["candidate_mask"], torch.finfo(torch.float32).min
        )
        targets = (
            torch.nn.functional.one_hot(batch["labels"], logits.shape[1]).float()
            * valid
        )
        q = None
        if any(t is not None for t in teacher):
            q = torch.zeros_like(targets)
            for i, (record, t) in enumerate(zip(records, teacher)):
                if t is None:
                    continue
                row = self.rows[record["source_row_id"]]
                mapped = original_probabilities(row, native_keys(record), t)
                q[i, : len(mapped)] = torch.tensor(mapped, device=device)
        logp = torch.log_softmax(logits, -1)
        ce = -(targets * logp * valid).sum(-1)
        cfg = self.spec["loss"]
        extra, brier, kl = extra_terms(
            logits, targets, valid, q, cfg["brier"], cfg["teacher_kl"]
        )
        return (cfg["ce"] * ce + extra).sum(), {
            "ce": float(ce.detach().sum()),
            "brier": float(brier.detach().sum()),
            "kl": float(kl.detach().sum()),
        }

    def _probabilities(
        self, model: Any, records: list[dict[str, Any]]
    ) -> list[list[float]]:
        import torch

        out = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(records), 8):
                chunk = records[start : start + 8]
                batch = self.batch(chunk)
                probs = model(**batch).float().softmax(-1).cpu().tolist()
                for record, p in zip(chunk, probs):
                    row = self.rows[record["source_row_id"]]
                    out.append(native_from_original(row, p[: len(row["options"])]))
        return out

    def probabilities(self, records: list[dict[str, Any]]) -> list[list[float]]:
        return self._probabilities(self.model, records)

    def state(self) -> dict[str, Any]:
        return {
            k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state, strict=True)

    def export(self, output: Path, provenance: dict[str, Any]) -> str:
        self.model.metadata = {**self.model.metadata, "dev2_06b_provenance": provenance}
        self.model.save(output, self.tokenizer)
        files = {
            item.relative_to(output).as_posix(): {
                "bytes": item.stat().st_size,
                "sha256": file_sha256(item),
            }
            for item in sorted(output.rglob("*"))
            if item.is_file()
        }
        return write_json(
            output.with_name(output.name + ".MANIFEST.json"),
            {"schema": "dev2-06b-causal-files/1", "files": files},
        )

    def reload_probabilities(
        self, output: Path, manifest: str, records: list[dict[str, Any]], device: str
    ) -> list[list[float]]:
        from training.model.decision_model import DecisionModel

        if file_sha256(output.with_name(output.name + ".MANIFEST.json")) != manifest:
            raise ValueError("Causal export manifest differs")
        model, _ = DecisionModel.from_checkpoint(output)
        return self._probabilities(model.to(device), records)

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


def padding_parity(
    family: Any,
    plan: list[list[int]],
    kinds: list[str],
    lengths: list[int],
    records: list[dict[str, Any]],
    teacher: list[list[float] | None],
    budget: int,
    device: str,
) -> dict[str, Any]:
    """Parity on the plan's first multi-row micro-batches, built as in padded arms."""
    micros: list[list[int]] = []
    for rows in plan:
        micros.extend(
            m for m in micro_batches(rows, kinds, lengths, 8, budget) if len(m) > 1
        )
        if len(micros) >= PADDING_PARITY_MICRO_BATCHES:
            break
    family.train_mode()
    checks = [
        {
            "kind": kinds[m[0]],
            "lengths": [lengths[i] for i in m],
            **parity.micro_batch_parity(
                family, [records[i] for i in m], [teacher[i] for i in m], device
            ),
        }
        for m in micros[:PADDING_PARITY_MICRO_BATCHES]
    ]
    return {
        "passed": bool(checks) and all(c["passed"] for c in checks),
        "micro_batches": checks,
    }


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
    kai8k.runtime_flags(torch)
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
    mixture_report = None
    if "mixture" in spec["data"]:
        from training.model.data import check_partition_isolation

        if spec["data"]["mixture"]["template"] == "materialized":
            splits["train"], mixture_report = mix.load_materialized(
                spec["data"]["mixture"]
            )
        else:
            splits["train"], mixture_report = mix.build(spec["data"]["mixture"])
        check_partition_isolation(splits)
        write_json(output / "MIXTURE.json", mixture_report, exclusive=True)
    records = {
        role: native_records(splits[role], spec["data"]["converter_bundle"])
        for role in ("train", "select")
    }
    if spec["family"] == "kai-native":
        family = KaiFamily(spec, device)
    elif spec["family"] == "qwen-causal":
        family = CausalQwenFamily(
            spec,
            device,
            {row["id"]: row for role in ("train", "select") for row in splits[role]},
        )
    else:
        family = EncoderFamily(spec, device)

    lengths = {role: [family.length(r) for r in records[role]] for role in records}
    over = {role: sum(v > MAX_INPUT_TOKENS for v in lengths[role]) for role in lengths}
    if any(over.values()):
        raise ValueError(
            f"Rows exceed the complete-input cap; freeze a quarantine first: {over}"
        )
    kinds = [r["question"]["type"].lower() for r in records["train"]]
    if spec["loss"].get("score_class_balance"):
        if spec["family"] != "encoder":
            raise ValueError(
                "Score class balance is implemented for the encoder family"
            )
        # Bin each Score gold by its relative level (low/middle/high) across level counts.
        bins = {}
        for i, (row, kind) in enumerate(zip(splits["train"], kinds)):
            if kind == "score":
                bins[i] = round(2 * row["label"] / (len(row["options"]) - 1))
        counts = {b: list(bins.values()).count(b) for b in set(bins.values())}
        for i, b in bins.items():
            records["train"][i]["_loss_weight"] = len(bins) / (len(counts) * counts[b])

    teacher: list[list[float] | None] = [None] * len(records["train"])
    teacher_coverage = None
    if spec["teacher"] is not None:
        path = Path(spec["teacher"]["path"])
        if file_sha256(path) != spec["teacher"]["sha256"]:
            raise ValueError("Teacher file differs from its frozen hash")
        if spec["teacher"].get("format", "native") == "option-key":
            teacher, teacher_coverage = option_key_teacher(splits["train"], path)
            if spec["teacher"].get("coverage") != "subset" and teacher_coverage[
                "covered"
            ] != len(teacher):
                raise ValueError("Teacher does not cover every TRAIN row")
        else:
            by_id = {
                row["source_row_id"]: row["probabilities"] for row in read_jsonl(path)
            }
            keys = [
                r.get("teacher_source_id", r["source_row_id"]) for r in records["train"]
            ]
            if spec["teacher"].get("coverage") == "subset":
                teacher = [by_id.get(key) for key in keys]
            else:
                teacher = [by_id[key] for key in keys]
        for record, q in zip(records["train"], teacher):
            if q is None:
                continue
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
    freeze = int(spec["optimizer"].get("backbone_freeze_updates", 0))
    if freeze and (spec["family"] == "kai-native" or not 0 < freeze < total - 1):
        raise ValueError("Backbone freeze is for fresh-head families, within the run")
    backbone_warmup = int(
        (total - freeze)
        * spec["optimizer"].get(
            "backbone_warmup_ratio", spec["optimizer"]["warmup_ratio"]
        )
    )
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
        "backbone_freeze_updates": freeze,
        "backbone_warmup_updates": backbone_warmup if freeze else None,
        "teacher_rows": sum(q is not None for q in teacher),
        "teacher_coverage": teacher_coverage,
        "mixture": mixture_report,
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
    padding = None
    if preflight:
        padding = padding_parity(
            family,
            plan,
            kinds,
            lengths["train"],
            records["train"],
            teacher,
            spec["micro_token_budget"],
            device,
        )
        write_json(output / "PADDING_PARITY.json", padding)
    backbone = [
        p
        for g in optimizer.param_groups
        if g["name"].endswith("encoder")
        for p in g["params"]
    ]
    if spec["start"].get("compute_dtype", "bfloat16") not in ("bfloat16", "float32"):
        raise ValueError("compute_dtype must be bfloat16 (autocast) or float32")
    compute = (
        parity.fp32_compute
        if spec["start"].get("compute_dtype") == "float32"
        else contextlib.nullcontext
    )
    log = (output / "TRAIN_LOG.jsonl").open("x", encoding="utf-8")
    last_update = total if not preflight else 1
    for step in range(1, last_update + 1):
        if (time.monotonic() - started) / 3600 > spec["gpu_hour_cap"]:
            raise RuntimeError("GPU-hour cap reached; arm stopped")
        family.train_mode()
        frozen = step <= freeze
        if step in (1, freeze + 1):
            for p in backbone:
                p.requires_grad_(not frozen)
        for group in optimizer.param_groups:
            if freeze and group["name"].endswith("encoder"):
                group["lr"] = (
                    0.0
                    if frozen
                    else learning_rate(
                        step - freeze,
                        total - freeze,
                        backbone_warmup,
                        base[group["name"]],
                        spec["optimizer"]["lr_min"],
                    )
                )
                continue
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
            with compute():
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
                    "frozen_backbone": frozen,
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
            stop = spec["start"].get("collapse_stop")
            if (
                stop
                and step >= stop["step"]
                and curve[-1]["metrics"]["correct"] < stop["min_select"]
            ):
                write_json(
                    output / "STOPPED.json",
                    {
                        "reason": "preregistered collapse stop",
                        "step": step,
                        "select": curve[-1]["metrics"]["correct"],
                        "rule": stop,
                    },
                )
                raise RuntimeError("Preregistered collapse stop")
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
                "PREFLIGHT_PASS"
                if drift <= 1e-5 and changed == 0 and padding["passed"]
                else "PREFLIGHT_FAIL"
            ),
            "padding_parity_passed": padding["passed"],
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
