"""9B M10 ML block (amendment 4): the M15 multilingual-preserving copy block on an m9_data.py matched build.

The released 9B mixture x60 has multilingual token share s (rows whose language is not `en`). A matched build (kept
x60 rows + IB rows, nearly all English) dilutes it. This adds one copy (id suffix `~m2`, M15's) of whole kept x60
groups whose every row is non-`en`, per language in proportion to the kept multilingual tokens (M15's `upsample`,
seeded per language), adding U = (s·T − ML − I_ml) / (1 − s) native tokens so the TRAIN multilingual share equals s
(T = build tokens, ML / I_ml = its x60 / IB multilingual tokens). Each copy's teacher line is its original's targets
under the copy id. The build fails if the share is off s by more than M15's tolerance.

usage: m10_ml.py --build DIR --x60-train F --x60-ids F --ib-tokens F [F ...] --seed N --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

M15 = Path(__file__).resolve().parents[3] / "dec" / "ops" / "m15" / "m15_data.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def m15() -> Any:
    spec = importlib.util.spec_from_file_location("m15_data", M15)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(args: argparse.Namespace) -> dict[str, Any]:
    lib = m15()
    native = {}
    with args.x60_ids.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                e = json.loads(line)
                native[e["id"]] = int(e["native"])
    for path in args.ib_tokens:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    e = json.loads(line)
                    native[e["id"]] = int(e["native"])
    full_ml = full = 0
    with args.x60_train.open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            full += native[r["id"]]
            full_ml += native[r["id"]] if r["language"] != "en" else 0
    s = full_ml / full
    manifest = json.loads((args.build / "manifest.json").read_text())
    n_x60 = manifest["x60_rows_kept"]
    lines = (args.build / "train.jsonl").read_bytes().splitlines(keepends=True)
    base, T, ML, I_ml = [], 0, 0, 0
    for k, line in enumerate(lines):
        r = json.loads(line)
        tok = native[r["id"]]
        T += tok
        if k < n_x60:
            base.append(
                {
                    "id": r["id"],
                    "group_id": r["group_id"],
                    "language": r["language"],
                    "tokens": tok,
                }
            )
            ML += tok if r["language"] != "en" else 0
        elif r.get("language", "en") != "en":
            I_ml += tok
    U = max(0, round((s * T - ML - I_ml) / (1 - s)))
    copies = lib.upsample(base, U, args.seed)
    added = sum(base[j]["tokens"] for j in copies)
    share = (ML + I_ml + added) / (T + added)
    if abs(share - s) > lib.TOL:
        raise ValueError(f"built multilingual share {share:.4f} is off {s:.4f}")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "train.jsonl").open("xb") as out:
        out.writelines(lines)
        for j in copies:
            out.write(lib.copy_line(lines[j]))
    teacher = {}
    with (args.build / "teacher.jsonl").open("rb") as stream:
        teacher_lines = stream.read().splitlines(keepends=True)
    for line in teacher_lines:
        teacher[json.loads(line)["id"]] = line
    with (args.output / "teacher.jsonl").open("xb") as out:
        out.writelines(teacher_lines)
        for j in copies:
            entry = json.loads(teacher[base[j]["id"]])
            entry["id"] = base[j]["id"] + lib.ML_SUFFIX
            out.write(
                json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode()
                + b"\n"
            )
    langs: Counter = Counter()
    for j in copies:
        langs[base[j]["language"]] += base[j]["tokens"]
    report = {
        "schema": "lux9b-m10-ml/1",
        "build": str(args.build),
        "build_train_sha256": sha256(args.build / "train.jsonl"),
        "x60_ml_share": round(s, 6),
        "build_tokens": T,
        "build_ml_tokens": ML + I_ml,
        "build_ml_share": round((ML + I_ml) / T, 6),
        "target_added_tokens": U,
        "copies": {
            "rows": len(copies),
            "groups": len({base[j]["group_id"] for j in copies}),
            "tokens": added,
            "tokens_by_language": dict(langs.most_common()),
        },
        "train_native_tokens": T + added,
        "ml_share": round(share, 6),
        "rows": len(lines) + len(copies),
        "seed": args.seed,
        "train_sha256": sha256(args.output / "train.jsonl"),
        "teacher_sha256": sha256(args.output / "teacher.jsonl"),
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--x60-train", type=Path, required=True)
    parser.add_argument("--x60-ids", type=Path, required=True)
    parser.add_argument("--ib-tokens", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args(argv))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
