"""Own-Lux targets on the XL rows that lack them (M3b prereg §3): run facts and coverage.

    python3 -m v2.data.m3.luxxl provenance --wave lux-xl-w1 --prompts lux-xl-w1.prompts.jsonl \\
        --guard lux-xl-w1.guard.json --collector-log lux-xl-w1.jsonl.log --teach-line teach.json \\
        --image-id sha256:... --launcher-sha256 ... --queue-sha256 ... --teach-mirror <sha> \\
        --triton-cache-files N --triton-cache-sha256 H --out provenance.json
    python3 -m v2.data.m3.luxxl coverage --manifest mx-xl.manifest.json --missing-dir DIR \\
        --wave lux-xl-w1=lux-xl-w1.prompts.jsonl ... --out coverage.json

``provenance`` checks one finished node-B wave before conversion: the target guard passed on
exactly this prompt file, the Milestone 2 launcher exited 0 on it, the collector summary
reports the pinned Lux1 identity, an attested revision, the validated runtime and every
prompt collected in one pass, and the image is the one of own-Lux waves 1-4. It writes the
path-free JSON that ``v2.data.m2.targets --provenance`` copies into the report (``per_row``
into every attestation line). ``coverage`` counts, per XL recipe, the rows that gain an
own-Lux target with each wave and the rows no wave covers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from v2.data.m2.common import read_jsonl
from v2.data.m2.targets import file_sha256

LUX_ID = "llm-semantic-router/Decision-1.0-Lux-9B"
LUX_REVISION = "bd45a30aee8c84032791c245c70f86dee5389cc8"
ADAPTER = "native-published-v2-overbudget-invalid-v1"
IMAGE_TAG = "decision20-lux-runtime:latest"
IMAGE_ID = "sha256:ce895822fc48bb6864911d4488a3946f3a18fd3dd2ec90c8a0a49b259145f2fb"
CONV1D_FALLBACK = (
    "`causal_conv1d_fn` is falling back to its reference PyTorch implementation"
)
IMAGE_DISCLOSURE = (
    "Image sha256:ce895822 (node B decision20-lux-runtime) has no causal-conv1d kernel; "
    "Transformers computes the same convolution with its reference PyTorch fallback. The "
    "eval track found that this image changes 43 of 8,778 Lux1 argmax answers against the "
    "kernel-equipped image (note 2026-09-28 23:40 UTC+8). Own-Lux RP-v2 waves 1-4 used this "
    "image, so the XL targets stay consistent with every earlier own-Lux target."
)
SUMMARY_KEYS = (
    "adapter_version",
    "backend",
    "collected_now",
    "input_items",
    "model_config_sha256",
    "model_id",
    "over_budget_rows_now",
    "previously_completed",
    "revision",
    "revision_attested",
    "runtime_differences",
    "runtime_matches_validated",
)


def _ids(path: Path) -> list[str]:
    return [row["id"] for row in read_jsonl(path)]


def collector_summary(log_text: str) -> dict[str, Any]:
    lines = [line for line in log_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("collector log is empty")
    summary = json.loads(lines[-1])
    if not isinstance(summary, dict):
        raise ValueError("collector log does not end with a JSON summary")
    return summary


def check_run(
    summary: dict[str, Any], teach: dict[str, Any], prompts: int, wave: str
) -> list[str]:
    expected = {
        "backend": "lux",
        "model_id": LUX_ID,
        "revision": LUX_REVISION,
        "adapter_version": ADAPTER,
        "revision_attested": True,
        "runtime_matches_validated": True,
        "runtime_differences": {},
        "previously_completed": 0,
        "input_items": prompts,
        "collected_now": prompts,
    }
    problems = [key for key, value in expected.items() if summary.get(key) != value]
    if teach.get("teacher") != "lux" or teach.get("rc") != 0:
        problems.append("launcher_rc")
    if not str(teach.get("input", "")).endswith(f"/{wave}.prompts.jsonl"):
        problems.append("launcher_input")
    return problems


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    guard = json.loads(args.guard.read_text(encoding="utf-8"))
    prompts_sha256 = file_sha256(args.prompts)
    prompts = len(_ids(args.prompts))
    if guard.get("pass") is not True or guard.get("prompts_sha256") != prompts_sha256:
        raise ValueError("target guard did not pass on this prompt file")
    if guard.get("prompts") != prompts:
        raise ValueError("guard receipt counts a different prompt file")
    log_text = args.collector_log.read_text(encoding="utf-8")
    summary = collector_summary(log_text)
    teach = json.loads(args.teach_line.read_text(encoding="utf-8"))
    problems = check_run(summary, teach, prompts, args.wave)
    if args.image_id != IMAGE_ID:
        problems.append("image_id")
    if problems:
        raise ValueError(
            f"{args.wave}: teacher run not on the M2 own-Lux runtime: {problems}"
        )
    guard = dict(guard, protected_files=len(guard.get("protected_files", {})))
    per_row = {"node": "node B", "gpu": 7, "image_id": IMAGE_ID}
    return {
        "schema": "decision2-m3b-luxxl-provenance/1",
        "wave": args.wave,
        "prompts": prompts,
        "prompts_sha256": prompts_sha256,
        "guard": guard,
        "teacher_run": {
            **per_row,
            "launcher": "Milestone 2 teach.sh (docker, --network none, GPU7)",
            "launcher_sha256": args.launcher_sha256,
            "queue_sha256": args.queue_sha256,
            "collector": "inference.run --backend lux --over-budget-invalid (native, no chat API)",
            "mirror_commit": args.teach_mirror,
            "image": IMAGE_TAG,
            "causal_conv1d_kernel": CONV1D_FALLBACK not in log_text,
            "triton_cache": {
                "role": "node-B Lux autotune cache of own-Lux waves 1-4",
                "files": args.triton_cache_files,
                "tree_sha256": args.triton_cache_sha256,
            },
            "rc": teach["rc"],
            "wall_s": teach["wall_s"],
            "end_utc": teach["end_utc"],
            "gpu_hours": round(teach["wall_s"] / 3600, 4),
            "collector_summary": {key: summary.get(key) for key in SUMMARY_KEYS},
        },
        "image_disclosure": IMAGE_DISCLOSURE,
        "per_row": per_row,
    }


def coverage(
    manifest: dict[str, Any], missing_dir: Path, waves: list[tuple[str, Path]]
) -> dict[str, Any]:
    wave_ids = []
    for name, path in waves:
        ids = _ids(path)
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: duplicate prompt id")
        wave_ids.append((name, set(ids)))
    covered = set().union(*(ids for _, ids in wave_ids))
    out: dict[str, Any] = {
        "waves": {name: len(ids) for name, ids in wave_ids},
        "recipes": {},
    }
    for recipe, spec in sorted(manifest["recipes"].items()):
        lux = spec["coverage"]["lux1"]
        missing = set(_ids(missing_dir / f"{recipe}.lux1.missing.jsonl"))
        if len(missing) != lux["rows_without"]:
            raise ValueError(f"{recipe}: missing list differs from the manifest")
        with_targets = lux["rows_with_targets"]
        steps = []
        for name, ids in wave_ids:
            gained = len(missing & ids)
            with_targets += gained
            steps.append(
                {
                    "wave": name,
                    "gained": gained,
                    "rows_with_targets": with_targets,
                    "share": round(with_targets / spec["rows"], 4),
                }
            )
        out["recipes"][recipe] = {
            "rows": spec["rows"],
            "sha256": spec["sha256"],
            "rows_with_targets_before": lux["rows_with_targets"],
            "after_wave": steps,
            "left_without_target": len(missing - covered),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("provenance")
    p.add_argument("--wave", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--guard", type=Path, required=True)
    p.add_argument("--collector-log", type=Path, required=True)
    p.add_argument("--teach-line", type=Path, required=True)
    p.add_argument("--image-id", required=True)
    p.add_argument("--launcher-sha256", required=True)
    p.add_argument("--queue-sha256", required=True)
    p.add_argument("--teach-mirror", required=True)
    p.add_argument("--triton-cache-files", type=int, required=True)
    p.add_argument("--triton-cache-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--missing-dir", type=Path, required=True)
    c.add_argument("--wave", action="append", required=True, help="NAME=PROMPTS")
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "provenance":
        result = provenance(args)
    else:
        waves = []
        for spec in args.wave:
            name, _, path = spec.partition("=")
            waves.append((name, Path(path)))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = coverage(manifest, args.missing_dir, waves)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
    print(json.dumps(result if args.command == "coverage" else {"wave": args.wave}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
