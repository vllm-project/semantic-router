"""Own-Lux targets on the XL rows that lack them (M3b prereg §3): run facts and coverage.

    python3 -m v2.data.m3.luxxl provenance --wave lux-xl-w1 --prompts lux-xl-w1.prompts.jsonl \\
        --guard lux-xl-w1.guard.json --collector-log lux-xl-w1.jsonl.log --teach-line teach.json \\
        --image-id sha256:... --launcher-sha256 ... --queue-sha256 ... --teach-mirror <sha> \\
        --triton-cache-files N --triton-cache-sha256 H --out provenance.json
    python3 -m v2.data.m3.luxxl coverage --manifest mx-xl.manifest.json --missing-dir DIR \\
        --wave lux-xl-w1=lux-xl-w1.prompts.jsonl ... --out coverage.json
    python3 -m v2.data.m3.luxxl control-ids --missing-dir DIR --out control-only.missing.jsonl
    python3 -m v2.data.m3.luxxl repeat-prompts --prompts lux-xl-h-w1.prompts.jsonl \\
        --out lux-xl-h-w1-r256.prompts.jsonl
    python3 -m v2.data.m3.luxxl repeat --wave lux-xl-h-w1 --wave-prompts lux-xl-h-w1.prompts.jsonl \\
        --wave-output lux-xl-h-w1.jsonl --prompts lux-xl-h-w1-r256.prompts.jsonl \\
        --guard lux-xl-h-w1-r256.guard.json --output lux-xl-h-w1-r256.jsonl \\
        --collector-log lux-xl-h-w1-r256.jsonl.log --teach-line teach-r256.json --out repeat.json

``provenance`` checks one finished node-B wave before conversion: the target guard passed on
exactly this prompt file, the Milestone 2 launcher exited 0 on it, the collector summary
reports the pinned Lux1 identity, an attested revision, the validated runtime and every
prompt collected in one pass, and the image is the one of own-Lux waves 1-4. With
``--repeat-check`` the wave's repeat receipt must pass, and with ``--triton-cache-checks``
every recorded cache state (JSON lines ``{stage, files, tree_sha256}``) must equal the cache
at conversion. It writes the path-free JSON that ``v2.data.m2.targets --provenance`` copies
into the report (``per_row`` into every attestation line). ``coverage`` counts, per XL recipe,
the rows that gain an own-Lux target with each wave and the rows no wave covers; a
``--require-full`` recipe left with any row refuses the output. ``control-ids`` lists the rows
of the four ``cx-xl-*`` controls that lack an own-Lux target and are in neither XL recipe's
missing list (M3b amendment 3 §4), sorted by id, as ``v2.data.m3.xl_prompts --missing`` input.

``repeat-prompts`` and ``repeat`` are the repeat check of wave h-w1 (prereg
``m3b-lux-h-prereg-2026-09-29.md`` §4): the first 256 wave prompts in
``sha256("m3b-lux-h-repeat:" + id)`` order, byte-identical lines sorted by id, answered again
in a fresh process and compared with the wave's answers (M3a ``compare`` and the M2
``repeat_max_abs_diff``). It passes with every prompt compared, no validity or argmax
mismatch and a maximum drift of at most 1e-3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from v2.data.m2.common import read_jsonl
from v2.data.m2.targets import file_sha256
from v2.data.m3.qualify import REPEAT_MAX_DRIFT, compare, ranked
from v2.data.replay_targets import repeat_max_abs_diff

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
REPEAT_PROMPTS = 256
REPEAT_SALT = "m3b-lux-h-repeat:"


def _ids(path: Path) -> list[str]:
    return [row["id"] for row in read_jsonl(path)]


def _guard_passed(guard_path: Path, prompts_path: Path) -> dict[str, Any]:
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    if guard.get("pass") is not True or guard.get("prompts_sha256") != file_sha256(
        prompts_path
    ):
        raise ValueError("target guard did not pass on this prompt file")
    if guard.get("prompts") != len(_ids(prompts_path)):
        raise ValueError("guard receipt counts a different prompt file")
    return guard


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
    guard = _guard_passed(args.guard, args.prompts)
    prompts_sha256 = file_sha256(args.prompts)
    prompts = len(_ids(args.prompts))
    log_text = args.collector_log.read_text(encoding="utf-8")
    summary = collector_summary(log_text)
    teach = json.loads(args.teach_line.read_text(encoding="utf-8"))
    problems = check_run(summary, teach, prompts, args.wave)
    if args.image_id != IMAGE_ID:
        problems.append("image_id")
    repeat = None
    if args.repeat_check is not None:
        repeat = json.loads(args.repeat_check.read_text(encoding="utf-8"))
        if repeat.get("pass") is not True or repeat.get("wave") != args.wave:
            problems.append("repeat_check")
    cache_checks = None
    if args.triton_cache_checks is not None:
        cache_checks = list(read_jsonl(args.triton_cache_checks))
        frozen = (args.triton_cache_files, args.triton_cache_sha256)
        if not cache_checks or any(
            (c.get("files"), c.get("tree_sha256")) != frozen for c in cache_checks
        ):
            problems.append("triton_cache_changed")
    if problems:
        raise ValueError(
            f"{args.wave}: teacher run not on the M2 own-Lux runtime: {problems}"
        )
    guard = dict(guard, protected_files=len(guard.get("protected_files", {})))
    per_row = {"node": "node B", "gpu": 7, "image_id": IMAGE_ID}
    triton_cache: dict[str, Any] = {
        "role": "node-B Lux autotune cache of own-Lux waves 1-4",
        "files": args.triton_cache_files,
        "tree_sha256": args.triton_cache_sha256,
    }
    if cache_checks is not None:
        triton_cache["checks"] = cache_checks
    out = {
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
            "triton_cache": triton_cache,
            "rc": teach["rc"],
            "wall_s": teach["wall_s"],
            "end_utc": teach["end_utc"],
            "gpu_hours": round(teach["wall_s"] / 3600, 4),
            "collector_summary": {key: summary.get(key) for key in SUMMARY_KEYS},
        },
        "image_disclosure": IMAGE_DISCLOSURE,
        "per_row": per_row,
    }
    if repeat is not None:
        out["repeat_check"] = repeat
    return out


def repeat_selection(wave_prompts: Path, n: int, salt: str) -> bytes:
    lines = {
        json.loads(line)["id"]: line
        for line in wave_prompts.read_bytes().splitlines(keepends=True)
        if line.strip()
    }
    chosen = sorted(ranked(list(lines), salt)[:n])
    if len(chosen) != n:
        raise ValueError(f"{len(lines)} wave prompts cannot give {n} repeat prompts")
    return b"".join(lines[i] for i in chosen)


def repeat(args: argparse.Namespace) -> dict[str, Any]:
    name = f"{args.wave}-r{args.n}"
    selection = repeat_selection(args.wave_prompts, args.n, args.salt)
    if args.prompts.read_bytes() != selection:
        raise ValueError("repeat prompts are not the preregistered selection")
    guard = _guard_passed(args.guard, args.prompts)
    ids = sorted(_ids(args.prompts))
    log_text = args.collector_log.read_text(encoding="utf-8")
    summary = collector_summary(log_text)
    teach = json.loads(args.teach_line.read_text(encoding="utf-8"))
    problems = check_run(summary, teach, len(ids), name)
    if problems:
        raise ValueError(
            f"{name}: repeat run not on the M2 own-Lux runtime: {problems}"
        )
    again = list(read_jsonl(args.output))
    if sorted(r["id"] for r in again) != ids:
        raise ValueError("repeat output ids differ from its prompts")
    wanted = set(ids)
    first = [r for r in read_jsonl(args.wave_output) if r["id"] in wanted]
    if len(first) != len(wanted):
        raise ValueError("wave output lacks repeat prompts")
    stats = compare(
        {r["id"]: r["answers"] for r in first},
        {r["id"]: r["answers"] for r in again},
        ids,
    )
    passed = (
        stats["questions"] == len(ids)
        and stats["missing_prompts"] == 0
        and stats["validity_mismatches"] == 0
        and stats["argmax_mismatches"] == 0
        and stats["max_drift"] <= REPEAT_MAX_DRIFT
    )
    return {
        "schema": "decision2-m3b-luxxl-repeat/1",
        "wave": args.wave,
        "selection": {
            "rule": f"first {args.n} wave prompts in sha256(salt + id) order, sorted by id",
            "salt": args.salt,
            "wave_prompts_sha256": file_sha256(args.wave_prompts),
        },
        "prompts": len(ids),
        "prompts_sha256": file_sha256(args.prompts),
        "guard_pass": guard["pass"],
        "run": {
            "process": "fresh teach.sh lux process after the wave, same image and cache",
            "rc": teach["rc"],
            "wall_s": teach["wall_s"],
            "end_utc": teach["end_utc"],
            "gpu_hours": round(teach["wall_s"] / 3600, 4),
            "causal_conv1d_kernel": CONV1D_FALLBACK not in log_text,
            "collector_summary": {key: summary.get(key) for key in SUMMARY_KEYS},
        },
        "compare": stats,
        "m2_repeat_max_abs_diff": repeat_max_abs_diff(first, again),
        "bitwise_identical": stats["identical_answers"]
        == stats["questions"]
        == len(ids),
        "criterion": {
            "validity_mismatches": 0,
            "argmax_mismatches": 0,
            "max_drift": REPEAT_MAX_DRIFT,
        },
        "pass": passed,
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


def require_full(result: dict[str, Any], recipes: list[str]) -> None:
    for recipe in recipes:
        spec = result["recipes"].get(recipe)
        if spec is None:
            raise ValueError(f"{recipe}: not in the manifest")
        last = spec["after_wave"][-1]["rows_with_targets"] if spec["after_wave"] else 0
        if spec["left_without_target"] or last != spec["rows"]:
            raise ValueError(f"{recipe}: rows left without an own-Lux target")


CONTROLS = (
    "cx-xl-a7v1-full",
    "cx-xl-a7v1-short",
    "cx-xl-v2v1-full",
    "cx-xl-v2v1-short",
)
RECIPES = ("mx-xl-full", "mx-xl-short")


def control_only_ids(missing_dir: Path) -> list[str]:
    def ids(name: str) -> set[str]:
        return set(_ids(missing_dir / f"{name}.lux1.missing.jsonl"))

    controls = set().union(*(ids(name) for name in CONTROLS))
    return sorted(controls - set().union(*(ids(name) for name in RECIPES)))


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
    p.add_argument("--repeat-check", type=Path)
    p.add_argument("--triton-cache-checks", type=Path)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--missing-dir", type=Path, required=True)
    c.add_argument("--wave", action="append", required=True, help="NAME=PROMPTS")
    c.add_argument("--require-full", action="append", default=[], metavar="RECIPE")
    c.add_argument("--out", type=Path, required=True)
    k = sub.add_parser("control-ids")
    k.add_argument("--missing-dir", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("repeat-prompts")
    s.add_argument("--prompts", type=Path, required=True)
    s.add_argument("--n", type=int, default=REPEAT_PROMPTS)
    s.add_argument("--salt", default=REPEAT_SALT)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("repeat")
    r.add_argument("--wave", required=True)
    r.add_argument("--wave-prompts", type=Path, required=True)
    r.add_argument("--wave-output", type=Path, required=True)
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--guard", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--collector-log", type=Path, required=True)
    r.add_argument("--teach-line", type=Path, required=True)
    r.add_argument("--n", type=int, default=REPEAT_PROMPTS)
    r.add_argument("--salt", default=REPEAT_SALT)
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "control-ids":
        ids = control_only_ids(args.missing_dir)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.writelines(json.dumps({"id": i}) + "\n" for i in ids)
        print(json.dumps({"control_only_ids": len(ids)}))
        return 0
    if args.command == "repeat-prompts":
        data = repeat_selection(args.prompts, args.n, args.salt)
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        print(
            json.dumps({"prompts": args.n, "sha256": hashlib.sha256(data).hexdigest()})
        )
        return 0
    if args.command == "provenance":
        result = provenance(args)
    elif args.command == "repeat":
        result = repeat(args)
    else:
        waves = []
        for spec in args.wave:
            name, _, path = spec.partition("=")
            waves.append((name, Path(path)))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = coverage(manifest, args.missing_dir, waves)
        require_full(result, args.require_full)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
    if args.command == "repeat":
        print(
            json.dumps(
                {"pass": result["pass"], "max_drift": result["compare"]["max_drift"]}
            )
        )
        return 0 if result["pass"] else 3
    print(json.dumps(result if args.command == "coverage" else {"wave": args.wave}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
