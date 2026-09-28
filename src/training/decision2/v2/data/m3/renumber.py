"""Positional option-key versions (pk1) of A0, A0s, A0p, RP-v1q and R2 (M3a item 3).

    python3 -m v2.data.m3.renumber rows --file A0=a0.jsonl --file A0s=... --file A0p=... \\
        --file RP-v1q=... --r2 kai=kai.replay.jsonl ... --out-dir DIR
    python3 -m v2.data.m3.renumber lux --out-dir DIR --canonical A0-train.canonical.jsonl \\
        --output lux.jsonl --model-id ID --revision REV
    python3 -m v2.data.m3.renumber r2 --out-dir DIR \\
        --tier kai=kai.replay.jsonl=kai.jsonl=MODEL_ID=REVISION ...

``rows`` applies `v2.common.option_keys` (A7 rule 7d) to every file and writes
``DIR/<name>/train.jsonl`` with a count-only receipt, plus the re-derivation prompt files:
``lux-a0.prompts.jsonl`` (every renumbered A0 row and CONTROL unchanged rows) and
``r2.prompts.jsonl`` (every renumbered RP-v1q row and unchanged control rows present in
every R2 tier). ``lux`` and ``r2`` rebuild the teacher files: unchanged rows keep their
targets; renumbered rows get the teacher's answer on the renumbered prompt, checked by
`v2.data.replay_targets.convert` (identity, attested revision, validated runtime, prompt
digest), never the old target moved with its key. Control rows measure drift against the
original targets. Rows the teacher cannot answer natively get no target (counted).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from training.model.data import canonical, validate_row
from v2.common.option_keys import AUDIT_KEY, RULE, renumber_rows
from v2.data.build_a0_variants import native_prompt
from v2.data.m2.common import read_jsonl

CONTROL_A0 = 64
CONTROL_R2 = 42


def ranked(ids, salt: str) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode()).hexdigest())


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, value: Any) -> str:
    return _write(path, (json.dumps(value, indent=1, sort_keys=True) + "\n").encode())


def _lines(rows) -> bytes:
    return "".join(canonical(r) + "\n" for r in rows).encode("utf-8")


def renumbered_ids(rows) -> list[str]:
    return sorted(r["id"] for r in rows if AUDIT_KEY in r["audit_metadata"])


def rows_command(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    versions: dict[str, list[dict]] = {}
    for spec in args.file:
        name, _, path = spec.partition("=")
        raw = Path(path).read_bytes()
        lines = raw.splitlines()
        rows = [json.loads(line) for line in lines]
        new, summary = renumber_rows(rows)
        for row in new:
            validate_row(row, "train")
        identical = sum(canonical(a) == canonical(b) for a, b in zip(rows, new))
        receipt = {
            "schema": "decision2-m3a-pk1/1",
            "rule": RULE,
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "input_canonical_lines": all(
                canonical(r).encode("utf-8") == line for r, line in zip(rows, lines)
            ),
            "unchanged_rows_byte_identical": identical
            == summary["rows"] - summary["renumbered"],
            **summary,
        }
        receipt["output_sha256"] = _write(
            args.out_dir / name / "train.jsonl", _lines(new)
        )
        _json(args.out_dir / name / "receipt.json", receipt)
        versions[name] = new
        out[name] = {k: receipt[k] for k in ("rows", "renumbered", "output_sha256")}
    a0 = versions["A0"]
    changed = renumbered_ids(a0)
    others = [r["id"] for r in a0 if r["id"] not in set(changed)]
    control = sorted(ranked(others, "m3a-pk1-control:")[: args.control_a0])
    by_id = {r["id"]: r for r in a0}
    lux_ids = sorted(changed + control)
    out["lux-a0.prompts.jsonl"] = _write(
        args.out_dir / "lux-a0.prompts.jsonl",
        "".join(
            json.dumps(native_prompt(by_id[i]), ensure_ascii=False) + "\n"
            for i in lux_ids
        ).encode("utf-8"),
    )
    rp = versions["RP-v1q"]
    rp_changed = renumbered_ids(rp)
    tiers = {}
    for spec in args.r2:
        tier, _, path = spec.partition("=")
        tiers[tier] = {r["id"] for r in read_jsonl(Path(path))}
    shared = set.intersection(*tiers.values()) if tiers else {r["id"] for r in rp}
    if not set(rp_changed) <= shared:
        raise ValueError("a renumbered RP-v1q row is missing from an R2 tier")
    rp_others = [
        r["id"] for r in rp if r["id"] in shared and r["id"] not in set(rp_changed)
    ]
    rp_control = sorted(ranked(rp_others, "m3a-pk1-control:")[: args.control_r2])
    rp_by_id = {r["id"]: r for r in rp}
    r2_ids = sorted(rp_changed + rp_control)
    out["r2.prompts.jsonl"] = _write(
        args.out_dir / "r2.prompts.jsonl",
        "".join(
            json.dumps(native_prompt(rp_by_id[i]), ensure_ascii=False) + "\n"
            for i in r2_ids
        ).encode("utf-8"),
    )
    _json(
        args.out_dir / "rederive.json",
        {
            "lux_a0": {"renumbered": changed, "control": control},
            "r2": {"renumbered": rp_changed, "control": rp_control},
        },
    )
    return out


def _drift(new: dict[str, float], old: dict[str, float]) -> float:
    return max(abs(new[k] - old[k]) for k in new)


def _top(dist: dict[str, float]) -> str:
    return max(sorted(dist), key=lambda k: dist[k])


def control_stats(pairs: list[tuple[dict, dict]]) -> dict[str, Any]:
    drifts = [_drift(a, b) for a, b in pairs]
    return {
        "rows": len(pairs),
        "identical": sum(a == b for a, b in pairs),
        "argmax_agree": sum(_top(a) == _top(b) for a, b in pairs),
        "max_abs_diff": max(drifts, default=0.0),
    }


def rederived(
    rows: list[dict], receipts_path: Path, teacher: str, model_id: str, revision: str
) -> tuple[dict[str, dict], dict[str, Any]]:
    from v2.data.replay_targets import convert

    receipts = list(read_jsonl(receipts_path))
    wanted = {r["id"] for r in receipts}
    chosen = [r for r in rows if r["id"] in wanted]
    replay, report = convert(
        chosen,
        [native_prompt(r) for r in chosen],
        receipts,
        teacher=teacher,
        model_id=model_id,
        revision=revision,
    )
    report["teacher_output_sha256"] = hashlib.sha256(
        receipts_path.read_bytes()
    ).hexdigest()
    return {r["id"]: r for r in replay}, report


def lux_command(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads((args.out_dir / "rederive.json").read_text())["lux_a0"]
    a0 = list(read_jsonl(args.out_dir / "A0" / "train.jsonl"))
    old = {r["id"]: r for r in read_jsonl(args.canonical)}
    new, report = rederived(a0, args.output, "lux", args.model_id, args.revision)
    control = [
        (new[i]["teacher_probs"], old[i]["teacher_probs"])
        for i in plan["control"]
        if i in new
    ]
    changed = set(plan["renumbered"])
    rows, missing = [], 0
    for row in sorted(a0, key=lambda r: r["id"]):
        if row["id"] in changed:
            if row["id"] not in new:
                missing += 1
                continue
            probs = new[row["id"]]["teacher_probs"]
        else:
            if old[row["id"]]["input_sha256"] != row["input_sha256"]:
                raise ValueError(
                    f"{row['id']}: unchanged row with a changed input hash"
                )
            probs = old[row["id"]]["teacher_probs"]
        rows.append(
            {
                "id": row["id"],
                "input_sha256": row["input_sha256"],
                "teacher_probs": probs,
            }
        )
    content = _write(args.out_dir / "lux1" / "A0-train.canonical.jsonl", _lines(rows))
    summary = {
        "schema": "decision2-m3a-pk1-lux-canonical/1",
        "previous_sha256": hashlib.sha256(args.canonical.read_bytes()).hexdigest(),
        "a0_pk1_sha256": hashlib.sha256(
            (args.out_dir / "A0" / "train.jsonl").read_bytes()
        ).hexdigest(),
        "rows": len(rows),
        "rederived": len(changed) - missing,
        "rederive_missing": missing,
        "control": control_stats(control),
        "rederive_report": report,
        "content_sha256": content,
    }
    _json(args.out_dir / "lux1" / "A0-train.canonical.report.json", summary)
    return {
        k: summary[k]
        for k in ("rows", "rederived", "rederive_missing", "control", "content_sha256")
    }


def r2_command(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads((args.out_dir / "rederive.json").read_text())["r2"]
    rp = list(read_jsonl(args.out_dir / "RP-v1q" / "train.jsonl"))
    changed = set(plan["renumbered"])
    out = {}
    for spec in args.tier:
        tier, old_path, output, model_id, revision = spec.split("=")
        old_rows = list(read_jsonl(Path(old_path)))
        old = {r["id"]: r for r in old_rows}
        new, report = rederived(rp, Path(output), tier, model_id, revision)
        control = [
            (new[i]["teacher_probs"], old[i]["teacher_probs"])
            for i in plan["control"]
            if i in new and i in old
        ]
        rows, missing = [], 0
        for row in old_rows:
            if row["id"] in changed:
                if row["id"] not in new:
                    missing += 1
                    continue
                rows.append(new[row["id"]])
            else:
                rows.append(row)
        for row in rows:
            validate_row(row, "train", replay=True)
        content = _write(args.out_dir / "R2" / tier / "replay.jsonl", _lines(rows))
        summary = {
            "schema": "decision2-m3a-pk1-r2/1",
            "tier": tier,
            "previous_sha256": hashlib.sha256(Path(old_path).read_bytes()).hexdigest(),
            "rows": len(rows),
            "rederived": len(changed) - missing,
            "rederive_missing": missing,
            "control": control_stats(control),
            "rederive_report": report,
            "content_sha256": content,
        }
        _json(args.out_dir / "R2" / tier / "report.json", summary)
        out[tier] = {
            k: summary[k]
            for k in (
                "rows",
                "rederived",
                "rederive_missing",
                "control",
                "content_sha256",
            )
        }
    return out


def prompts_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = sorted(read_jsonl(args.rows), key=lambda r: r["id"])
    data = "".join(
        json.dumps(native_prompt(r), ensure_ascii=False) + "\n" for r in rows
    )
    return {"prompts": len(rows), "sha256": _write(args.out, data.encode("utf-8"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--file", action="append", required=True)
    r.add_argument("--r2", action="append", default=[])
    r.add_argument("--control-a0", type=int, default=CONTROL_A0)
    r.add_argument("--control-r2", type=int, default=CONTROL_R2)
    r.add_argument("--out-dir", type=Path, required=True)
    lux = sub.add_parser("lux")
    lux.add_argument("--out-dir", type=Path, required=True)
    lux.add_argument("--canonical", type=Path, required=True)
    lux.add_argument("--output", type=Path, required=True)
    lux.add_argument("--model-id", required=True)
    lux.add_argument("--revision", required=True)
    r2 = sub.add_parser("r2")
    r2.add_argument("--out-dir", type=Path, required=True)
    r2.add_argument("--tier", action="append", required=True)
    pr = sub.add_parser("prompts")
    pr.add_argument("--rows", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    command = {
        "rows": rows_command,
        "lux": lux_command,
        "r2": r2_command,
        "prompts": prompts_command,
    }[args.command]
    print(json.dumps(command(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
