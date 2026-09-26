"""Verify two frozen public receipts and make an exposed-panel rank input.

This reads only file hashes, safetensors headers, and aggregate score reports.
It runs no model or scorer and never parses prompt or target values.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

from jev_arena import arena, public_rank
from small_decision_competitors import sha_file

ARMS = {"rlcd_a": "a", "rlcd_b": "b"}
PUBLIC_ITEMS = 231


def verify_hash(path: Path, expected: str) -> None:
    if sha_file(path) != expected:
        raise ValueError(f"hash differs: {path}")


def tensor_count(path: Path) -> int:
    with path.open("rb") as model:
        header_len = struct.unpack("<Q", model.read(8))[0]
        if not 0 < header_len < 100_000_000:
            raise ValueError(f"invalid safetensors header: {path}")
        header = json.loads(model.read(header_len))
    return sum(
        math.prod(tensor["shape"])
        for name, tensor in header.items()
        if name != "__metadata__"
    )


def verify_inputs(
    roster_path: Path, run_dir: Path, panel_dir: Path, scorer_path: Path
) -> dict:
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    expected_panel = roster["expected_panel"]
    if len(roster["models"]) != 2 or set(roster["expected_receipts"]) != set(ARMS):
        raise ValueError("this diagnostic requires exactly its two pinned arms")
    for name, expected in (
        ("prompts.jsonl", expected_panel["prompts_sha256"]),
        ("targets.jsonl", expected_panel["targets_sha256"]),
        ("manifest.json", expected_panel["panel_manifest_sha256"]),
    ):
        verify_hash(panel_dir / name, expected)
    verify_hash(scorer_path, expected_panel["scorer_sha256"])
    verify_hash(Path(public_rank.__file__), expected_panel["ranker_sha256"])
    verify_hash(Path(arena.__file__), expected_panel["arena_sha256"])

    for entry in roster["models"]:
        key = entry["key"]
        if key not in ARMS:
            raise ValueError(f"unknown arm: {key}")
        arm = ARMS[key]
        receipt = roster["expected_receipts"][key]
        pred_path = run_dir / "predictions" / f"{arm}-public.jsonl"
        report_path = run_dir / "scores" / entry["report"]
        pred_manifest_path = pred_path.with_suffix(pred_path.suffix + ".manifest.json")
        verify_hash(pred_path, receipt["predictions_sha256"])
        verify_hash(report_path, receipt["report_sha256"])
        pred_manifest = json.loads(pred_manifest_path.read_text(encoding="utf-8"))
        if (
            pred_manifest["arm"] != arm
            or pred_manifest["revision"] != entry["revision"]
            or pred_manifest["collector_sha256"] != expected_panel["collector_sha256"]
            or pred_manifest["prompts_sha256"] != expected_panel["prompts_sha256"]
            or pred_manifest["predictions_sha256"] != receipt["predictions_sha256"]
            or not receipt["model_hashes"].items()
            <= pred_manifest["model_hashes"].items()
            or pred_manifest["n"] != PUBLIC_ITEMS
        ):
            raise ValueError(f"native prediction identity differs: {key}")
        verify_hash(
            run_dir / "preflight" / f"{arm}-public.jsonl",
            pred_manifest["preflight_sha256"],
        )
        model_dir = run_dir / "models" / arm
        for name, digest in receipt["model_hashes"].items():
            verify_hash(model_dir / name, digest)
        count = tensor_count(model_dir / "model.safetensors")
        if arm == "b":
            count += tensor_count(model_dir / "decision_head.safetensors")
        if count != receipt["actual_parameters"] or entry["size_b"] != count / 1e9:
            raise ValueError(f"actual parameter count differs: {key}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report["score_version"] != expected_panel["score_version"]
            or report["model_id"] != entry["model_id"]
            or report["model_revision"] != entry["revision"]
            or report["prompts_sha256"] != expected_panel["prompts_sha256"]
            or report["targets_sha256"] != expected_panel["targets_sha256"]
            or report["panel_manifest_sha256"]
            != expected_panel["panel_manifest_sha256"]
            or report["items"] != PUBLIC_ITEMS
        ):
            raise ValueError(f"aggregate report identity differs: {key}")
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_inputs(args.roster, args.run_dir, args.panel_dir, args.scorer)
    result = public_rank.rank(args.roster, args.run_dir / "scores")
    if result["scope"].find("not official") < 0:
        raise ValueError("public-only rank scope was lost")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    print(
        json.dumps(
            {
                "rank_sha256": sha_file(args.output),
                "roster_sha256": sha_file(args.roster),
                "scores": [
                    {
                        key: row[key]
                        for key in (
                            "key",
                            "rank",
                            "score",
                            "size_b",
                            "pareto_frontier",
                            "valid",
                        )
                    }
                    for row in result["models"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
