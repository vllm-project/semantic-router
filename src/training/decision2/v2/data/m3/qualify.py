"""AutoJev-27B ROCm runtime qualification for teacher targets (M3a preregistration).

    python3 -m v2.data.m3.qualify sets --rows rp-v2.rows.jsonl --prompts rp-v2.prompts.jsonl \\
        --tokens A.tokens.jsonl ... --panel public231=P.jsonl=SHA256 ... --out-dir DIR
    python3 -m v2.data.m3.qualify compare --sets DIR/sets.json --warm-manifest W.json \\
        --run P1=p1.jsonl=p1.manifest.json ... --pair P1:P2 ... \\
        --reference eval=typed.jsonl,public.jsonl,css.jsonl --out report.json

``sets`` draws the warm-up set W and the repeat set R from RP-v2 TRAIN prompts and the
spot-check set S from the eval track's gold-free panel prompts, and writes
``warm.prompts.jsonl`` (W), ``qual.prompts.jsonl`` (R, then S) and ``sets.json``.
``compare`` applies gates G1-G4 of the preregistration and writes a count-only report
(no prompt text, answers or ids). Outputs on S are never converted into targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from v2.data.m2.common import read_jsonl

REPEAT_PER_TYPE = 128
REPEAT_LONG = 128
WARM_PER_TYPE = 16
WARM_LONG = 8
LONG_TOKENS = 4096
SPOT = {"public231": None, "typed-final": 128, "css15": 128}
TYPES = ("choice", "noul", "score")

REPEAT_MAX_DRIFT = 1e-3
SPOT_MIN_ARGMAX_AGREEMENT = 0.99
SPOT_MAX_P99_DRIFT = 0.05

IMAGE_ID = "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54"
LOADED_PARAMETERS = 26086635760
IDENTITY = {
    "model_id": "denis-pplx/autojev-27b",
    "model_revision": "6f5b557e037f5edb25c7dc92dbc6553e5a19c015",
    "revision_attested": True,
    "adapter_version": "autojev27-native-v1",
    "backend": "autojev-native-eager",
    "model_config_sha256": "bacbcbb281a53af5ef5cc6c9028601097d155bf981129f18a727219517921dcd",
}
RUN_CONSTANT = ("native_model_sha256", "runtime_source_sha256")
PENDING = "pytorch_bf16_rocm_pending_repeatability"


def ranked(ids: Iterable[str], salt: str) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode("utf-8")).hexdigest())


def _write(path: Path, data: bytes) -> str:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def draw(
    kinds: Mapping[str, str], tokens: Mapping[str, int]
) -> tuple[list[str], list[str]]:
    """Repeat set R and warm-up set W (disjoint) from TRAIN ids by type and length."""
    repeat: list[str] = []
    for kind in TYPES:
        pool = [i for i, k in kinds.items() if k == kind]
        repeat += ranked(pool, "m3a-repeat:")[:REPEAT_PER_TYPE]
    chosen = set(repeat)
    long_pool = [i for i in kinds if tokens[i] >= LONG_TOKENS and i not in chosen]
    repeat += ranked(long_pool, "m3a-repeat:")[:REPEAT_LONG]
    chosen = set(repeat)
    warm: list[str] = []
    for kind in TYPES:
        pool = [i for i, k in kinds.items() if k == kind and i not in chosen]
        warm += ranked(pool, "m3a-warm:")[:WARM_PER_TYPE]
    taken = chosen | set(warm)
    long_pool = [i for i in kinds if tokens[i] >= LONG_TOKENS and i not in taken]
    warm += ranked(long_pool, "m3a-warm:")[:WARM_LONG]
    return sorted(repeat), sorted(warm)


def sets(args: argparse.Namespace) -> int:
    kinds = {row["id"]: row["task_type"] for row in read_jsonl(args.rows)}
    tokens: dict[str, int] = {}
    for path in args.tokens:
        for row in read_jsonl(path):
            if row["id"] in kinds:
                tokens[row["id"]] = int(row["native"])
    if set(tokens) != set(kinds):
        raise ValueError(f"{len(set(kinds) - set(tokens))} RP-v2 rows lack token counts")
    repeat, warm = draw(kinds, tokens)
    lines: dict[str, bytes] = {}
    with args.prompts.open("rb") as stream:
        for line in stream:
            ident = json.loads(line)["id"]
            if ident in kinds:
                lines[ident] = line
    if set(lines) != set(kinds):
        raise ValueError("prompt file does not match the RP-v2 rows")
    spot: dict[str, list[bytes]] = {}
    panels: dict[str, dict[str, Any]] = {}
    for spec in args.panel:
        name, path, expected = spec.split("=")
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"{name}: panel prompt file differs from the sealed eval prompts")
        by_id = {json.loads(line)["id"]: line + b"\n" for line in data.splitlines() if line}
        take = SPOT[name]
        picked = sorted(by_id) if take is None else sorted(ranked(by_id, "m3a-spot:")[:take])
        spot[name] = [by_id[i] for i in picked]
        panels[name] = {"prompt_sha256": expected, "items": len(picked)}
    spot_ids = {json.loads(line)["id"] for items in spot.values() for line in items}
    if spot_ids & set(kinds) or sum(map(len, spot.values())) != len(spot_ids):
        raise ValueError("spot-check ids collide with TRAIN ids or each other")
    qual = b"".join(lines[i] for i in repeat) + b"".join(
        line for name in sorted(spot) for line in spot[name]
    )
    warm_data = b"".join(lines[i] for i in warm)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "decision2-m3a-qualification-sets/1",
        "repeat": {
            "prompts": len(repeat),
            "by_type": {k: sum(kinds[i] == k for i in repeat) for k in TYPES},
            "long": sum(tokens[i] >= LONG_TOKENS for i in repeat),
            "max_native_tokens": max(tokens[i] for i in repeat),
            "ids_sha256": hashlib.sha256("\n".join(repeat).encode()).hexdigest(),
        },
        "warm": {
            "prompts": len(warm),
            "long": sum(tokens[i] >= LONG_TOKENS for i in warm),
            "ids_sha256": hashlib.sha256("\n".join(warm).encode()).hexdigest(),
            "file_sha256": _write(args.out_dir / "warm.prompts.jsonl", warm_data),
        },
        "spot": panels,
        "repeat_ids": repeat,
        "spot_ids": sorted(spot_ids),
        "qual_file_sha256": _write(args.out_dir / "qual.prompts.jsonl", qual),
    }
    _write(
        args.out_dir / "sets.json",
        (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_ids")}, sort_keys=True))
    return 0


def vector(answer: Any) -> tuple[str, dict[str, float]] | None:
    """A native answer as (type, distribution over option keys); None when invalid."""
    if not isinstance(answer, dict) or "error" in answer:
        return None
    kind = answer.get("type")
    if kind == "noul":
        p = float(answer["noul"])
        return kind, {"false": 1.0 - p, "true": p}
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None
    return kind, {str(key): float(value) for key, value in probs.items()}


def top(dist: Mapping[str, float]) -> str:
    return max(sorted(dist), key=lambda key: dist[key])


def load_answers(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in read_jsonl(path):
            if record["id"] in out:
                raise ValueError(f"{path}: duplicate id")
            out[record["id"]] = record["answers"]
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def compare(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    ids: Iterable[str],
) -> dict[str, Any]:
    """Per-question agreement of two runs on the given prompt ids."""
    questions = identical = validity = argmax = missing = 0
    drifts: list[float] = []
    for ident in ids:
        a, b = left.get(ident), right.get(ident)
        if a is None or b is None or set(a) != set(b):
            missing += 1
            continue
        for key in sorted(a):
            questions += 1
            identical += a[key] == b[key]
            va, vb = vector(a[key]), vector(b[key])
            if va is None or vb is None:
                validity += (va is None) != (vb is None)
                continue
            if va[0] != vb[0] or set(va[1]) != set(vb[1]):
                validity += 1
                continue
            drifts.append(max(abs(va[1][k] - vb[1][k]) for k in va[1]))
            argmax += top(va[1]) != top(vb[1])
    return {
        "questions": questions,
        "missing_prompts": missing,
        "identical_answers": identical,
        "validity_mismatches": validity,
        "argmax_mismatches": argmax,
        "compared_distributions": len(drifts),
        "max_drift": max(drifts, default=0.0),
        "p99_drift": _quantile(drifts, 0.99),
        "p50_drift": _quantile(drifts, 0.50),
        "mean_drift": sum(drifts) / len(drifts) if drifts else 0.0,
        "over_1e-6": sum(d > 1e-6 for d in drifts),
        "over_1e-3": sum(d > 1e-3 for d in drifts),
        "over_1e-2": sum(d > 1e-2 for d in drifts),
    }


def identity_check(
    receipts: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any], rows: int
) -> dict[str, Any]:
    """G1 for one process: pinned identity on every receipt plus the launcher manifest."""
    problems: dict[str, int] = {}
    constants: dict[str, set[str]] = {key: set() for key in RUN_CONSTANT}
    seen = 0
    for record in receipts:
        seen += 1
        for key, value in IDENTITY.items():
            if record.get(key) != value:
                problems[key] = problems.get(key, 0) + 1
        if record.get("runtime_qualification") != PENDING:
            problems["runtime_qualification"] = problems.get("runtime_qualification", 0) + 1
        for key in RUN_CONSTANT:
            constants[key].add(str(record.get(key)))
    summary = manifest.get("collector") or {}
    checks = {
        "rows": seen == rows,
        "exit_code": manifest.get("exit_code") == 0,
        "image_id": manifest.get("image_id") == IMAGE_ID,
        "fla_path": manifest.get("fla_reference_fallback") is False,
        "loaded_parameters": summary.get("loaded_parameters") == LOADED_PARAMETERS,
        "receipt_identity": not problems,
        "constant_package_hashes": all(len(v) == 1 for v in constants.values()),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "receipt_problems": problems,
        "package_hashes": {k: sorted(v) for k, v in constants.items()},
    }


def compare_command(args: argparse.Namespace) -> int:
    manifest = json.loads(args.sets.read_text(encoding="utf-8"))
    repeat_ids, spot_ids = manifest["repeat_ids"], manifest["spot_ids"]
    all_ids = repeat_ids + spot_ids
    warm = json.loads(args.warm_manifest.read_text(encoding="utf-8"))
    frozen = warm["autotune_after"]
    runs: dict[str, dict[str, Any]] = {}
    g1: dict[str, Any] = {}
    g3: dict[str, Any] = {}
    hashes: dict[str, set[str]] = {key: set() for key in RUN_CONSTANT}
    for spec in args.run:
        name, output, manifest_path = spec.split("=")
        run_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        receipts = list(read_jsonl(Path(output)))
        g1[name] = identity_check(receipts, run_manifest, len(all_ids))
        for key in RUN_CONSTANT:
            hashes[key].update(g1[name]["package_hashes"][key])
        g3[name] = (
            run_manifest.get("autotune_before") == frozen
            and run_manifest.get("autotune_after") == frozen
        )
        runs[name] = {r["id"]: r["answers"] for r in receipts}
    pairs = {}
    for spec in args.pair:
        a, b = spec.split(":")
        pairs[spec] = compare(runs[a], runs[b], all_ids)
        pairs[spec]["pass"] = (
            pairs[spec]["missing_prompts"] == 0
            and pairs[spec]["validity_mismatches"] == 0
            and pairs[spec]["argmax_mismatches"] == 0
            and pairs[spec]["max_drift"] <= REPEAT_MAX_DRIFT
        )
    name, _, paths = args.reference.partition("=")
    reference = load_answers(Path(p) for p in paths.split(","))
    spot = {}
    for run in args.spot_run:
        stats = compare(runs[run], reference, spot_ids)
        answered = stats["compared_distributions"]
        stats["argmax_agreement"] = (
            (answered - stats["argmax_mismatches"]) / answered if answered else 0.0
        )
        stats["pass"] = (
            stats["missing_prompts"] == 0
            and stats["validity_mismatches"] == 0
            and stats["argmax_agreement"] >= SPOT_MIN_ARGMAX_AGREEMENT
            and stats["p99_drift"] <= SPOT_MAX_P99_DRIFT
        )
        spot[f"{run}:{name}"] = stats
    gates = {
        "G1_identity": all(v["pass"] for v in g1.values())
        and all(len(v) == 1 for v in hashes.values()),
        "G2_repeat": bool(pairs) and all(v["pass"] for v in pairs.values()),
        "G3_autotune_frozen": bool(g3) and all(g3.values()),
        "G4_spot_check": bool(spot) and all(v["pass"] for v in spot.values()),
    }
    report = {
        "schema": "decision2-m3a-autojev-qualification/1",
        "tolerances": {
            "repeat_max_drift": REPEAT_MAX_DRIFT,
            "spot_min_argmax_agreement": SPOT_MIN_ARGMAX_AGREEMENT,
            "spot_max_p99_drift": SPOT_MAX_P99_DRIFT,
        },
        "sets_sha256": hashlib.sha256(args.sets.read_bytes()).hexdigest(),
        "identity": IDENTITY,
        "image_id": IMAGE_ID,
        "package_hashes": {k: sorted(v) for k, v in hashes.items()},
        "autotune_frozen": frozen,
        "g1": g1,
        "g3": g3,
        "pairs": pairs,
        "spot": spot,
        "gates": gates,
        "pass": all(gates.values()),
    }
    _write(args.out, (json.dumps(report, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"gates": gates, "pass": report["pass"]}, sort_keys=True))
    return 0 if report["pass"] else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sets")
    s.add_argument("--rows", type=Path, required=True)
    s.add_argument("--prompts", type=Path, required=True)
    s.add_argument("--tokens", type=Path, action="append", required=True)
    s.add_argument("--panel", action="append", required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--sets", type=Path, required=True)
    c.add_argument("--warm-manifest", type=Path, required=True)
    c.add_argument("--run", action="append", required=True)
    c.add_argument("--pair", action="append", required=True)
    c.add_argument("--reference", required=True)
    c.add_argument("--spot-run", action="append", required=True)
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return sets(args) if args.command == "sets" else compare_command(args)


if __name__ == "__main__":
    sys.exit(main())
