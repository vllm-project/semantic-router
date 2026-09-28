"""Native System One predictions for 0.6B arms (no chat, no truncation).

`benchmark` answers gold-free typed/CSS/public prompt files and writes receipts
the existing typed, transfer and public JevBench scorers accept. `records`
writes probabilities for rights-clean SELECT/CAL/TRAIN rows (gold is never
read by the model). Kai/Lex bundles run at their 8,192 native cap; exports of
the encoder family load from their own file manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from inference.run import digest as payload_digest
from inference.run import load_prompts

from . import encoder as enc
from . import kai8k
from .common import (
    MAX_INPUT_TOKENS,
    file_sha256,
    import_bundle,
    load_rights_clean,
    metric_summary,
    native_keys,
    native_records,
    original_probabilities,
    select_record,
    write_json,
    write_jsonl,
)

ADAPTER_VERSION = "dev2-06b-native-8k-v1"


class Runner:
    """One loaded model; `answers` maps a System One request to typed answers."""

    def __init__(self, args: argparse.Namespace):
        self.family = args.family
        self.device = "cuda:0"
        if args.family == "kai-native":
            self.native, identity = kai8k.load(
                args.bundle,
                args.backend,
                native_dir=args.native_dir,
                manifest_sha256=args.manifest_sha256,
            )
            self.config_sha = identity["native_manifest_sha256"]
        else:
            import torch

            kai8k.runtime_flags(torch)
            import_bundle(args.bundle)
            self.model, self.packer, self.metadata = enc.load(
                args.native_dir, args.manifest_sha256, device=self.device
            )
            self.config_sha = args.manifest_sha256

    def probabilities(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Native rows -> per-row prediction dicts in Kai `predict` shape."""
        if self.family == "kai-native":
            return kai8k.predict_rows(self.native, rows)
        encoded = []
        for row in rows:
            try:
                encoded.append(self.packer.encode(row))
            except ValueError as exc:
                if "no implicit truncation" in str(exc):
                    raise kai8k.ContextOverflow(str(exc)) from exc
                raise
        probs = enc.probabilities(self.model, self.packer, rows, device=self.device)
        out = []
        for row, e, p in zip(rows, encoded, probs):
            item = {
                "id": row["id"],
                "question_id": row["question"].get("id"),
                "candidate_ids": e["candidate_ids"],
                "input_tokens": e["input_tokens"],
                "state_tokens_original": 0,
                "state_tokens_kept": 0,
                "probabilities": p,
            }
            if e["kind"] == "score":
                item["score"] = float(sum(i * v for i, v in enumerate(p)))
            out.append(item)
        return out

    def answers(
        self, state: Any, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        rows = kai8k.request_rows(state, questions)
        predictions = self.probabilities(rows)
        return (
            {
                row["question"]["id"]: kai8k.answer(row, p)
                for row, p in zip(rows, predictions)
            },
            sum(p["input_tokens"] for p in predictions),
        )


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    rows = load_prompts(args.input)
    if args.max_items:
        rows = rows[: args.max_items]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    receipts, overflow = [], 0
    started = time.monotonic()
    for row in rows:
        payload = {"state": row["state"], "questions": row["questions"]}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            answers, tokens = runner.answers(row["state"], row["questions"])
            invalid = None
        except kai8k.ContextOverflow:
            overflow += 1
            invalid = "context_overflow"
            tokens = None
            answers = {
                qid: {"type": q["type"], "error": invalid}
                for qid, q in row["questions"].items()
            }
        torch.cuda.synchronize()
        receipts.append(
            {
                "id": row["id"],
                "answers": answers,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "usage": (
                    {"input_tokens": tokens, "output_tokens": 0}
                    if tokens is not None
                    else None
                ),
                "model": args.model_id,
                "backend": args.backend_label,
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "revision_attested": True,
                "model_config_sha256": runner.config_sha,
                "adapter_version": ADAPTER_VERSION,
                "max_input_tokens": MAX_INPUT_TOKENS,
                "source_input_sha256": payload_digest(payload),
                "invalid_reason": invalid,
            }
        )
    sha = write_jsonl(args.output, receipts)
    report = {
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "items": len(rows),
        "context_overflow": overflow,
        "output_sha256": sha,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_config_sha256": runner.config_sha,
        "adapter_version": ADAPTER_VERSION,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(args.output.with_name(args.output.name + ".manifest.json"), report)
    return report


def records(args: argparse.Namespace) -> dict[str, Any]:
    splits = load_rights_clean(args.data_parent)
    rows = splits[args.role]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native = native_records(rows, args.bundle)
    runner = Runner(args)
    prepared = []
    for index, record in enumerate(native):
        item = {
            k: v for k, v in record.items() if k not in ("target", "hard_target_id")
        }
        item["id"] = f"record:{index}"
        prepared.append(item)
    started = time.monotonic()
    predictions = runner.probabilities(prepared)
    out, outcomes = [], []
    for row, record, prediction in zip(rows, native, predictions):
        p = prediction["probabilities"]
        if not all(math.isfinite(v) for v in p):
            raise FloatingPointError(row["id"])
        out.append(
            {
                "source_row_id": row["id"],
                "native_ids": native_keys(record),
                "probabilities": p,
            }
        )
        if args.role != "train":
            outcomes.append(
                select_record(row, original_probabilities(row, native_keys(record), p))
            )
    sha = write_jsonl(args.output, out)
    report = {
        "role": args.role,
        "rows": len(rows),
        "output_sha256": sha,
        "model_config_sha256": runner.config_sha,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": metric_summary(outcomes) if outcomes else None,
    }
    write_json(args.output.with_name(args.output.name + ".report.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("benchmark", "records"):
        p = sub.add_parser(name)
        p.add_argument("--family", choices=("kai-native", "encoder"), required=True)
        p.add_argument(
            "--bundle",
            type=Path,
            required=True,
            help="Pinned Kai/Lex bundle (converter and runtime)",
        )
        p.add_argument("--backend", choices=("kai", "lex"), default="kai")
        p.add_argument("--native-dir", type=Path)
        p.add_argument("--manifest-sha256")
        p.add_argument("--output", type=Path, required=True)
    sub.choices["benchmark"].add_argument("--input", type=Path, required=True)
    sub.choices["benchmark"].add_argument("--model-id", required=True)
    sub.choices["benchmark"].add_argument("--model-revision", required=True)
    sub.choices["benchmark"].add_argument("--backend-label", required=True)
    sub.choices["benchmark"].add_argument("--max-items", type=int)
    sub.choices["records"].add_argument("--data-parent", type=Path, required=True)
    sub.choices["records"].add_argument(
        "--role", choices=("train", "select", "cal"), required=True
    )
    args = parser.parse_args()
    if args.family == "encoder" and (
        args.native_dir is None or args.manifest_sha256 is None
    ):
        parser.error("encoder exports need --native-dir and --manifest-sha256")
    result = benchmark(args) if args.command == "benchmark" else records(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
