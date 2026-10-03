"""Arm factory 4B wave-2 TRAIN files (prereg af-prereg-2026-10-02.md), host python3 (standard library), CPU only.

  4b-LHS17IB4ML  M17's locked 4b-LHS17SD TRAIN (byte for byte), then IB4 phase 1 TRAIN (every row, file order), then
                 IB3-r2 TRAIN (every row, file order) - i.e. M17's locked 4b-LHS17IB4 TRAIN - then 4b-LHA10SDML's
                 multilingual copies (its `~m2` rows, file order); teacher = 4b-LHS17SD's, then the copies' teacher
                 rows from 4b-LHA10SDML's (file order): M17's 4b-LHS17IB4 and 4b-LHS17ML rules combined.
  4b-LHS23IB4    M17's locked 4b-LHS23SD TRAIN (byte for byte), then IB4 phase 1 TRAIN, then IB3-r2 TRAIN; teacher =
                 4b-LHS23SD's (IB rows gold only): M17 arm (a)'s rule at the S23 swap dose.

Checks: every input against its locked SHA-256; 4b-LHS17IB4ML's first rows equal M17's 4b-LHS17IB4 TRAIN; ids unique
in each output; every copy has its teacher row. Writes <out>/<ARM>/{train.jsonl,teacher-s.jsonl} and prints a report.

usage: af_prep4b.py --inputs /data/dev2/runs/af/4b/inputs/dec --out /data/dev2/runs/af/4b/data/w2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INPUTS = {
    "s17_train": (
        "m17/data/4b/4b-LHS17SD/train.jsonl",
        "14bce13ce926b354e214581e7cf4718d03f80c975b36dbce6731fbc6517e25a0",
    ),
    "s17_teacher": (
        "m17/data/4b/4b-LHS17SD/teacher-s.jsonl",
        "374f4fa68c8ae32f7b85fc8beeb988a1e4de43be544d4cfbf68193b2fe2fcea2",
    ),
    "s17ib4_train": (
        "m17/data/4b-s3/4b-LHS17IB4/train.jsonl",
        "dfed394459d77ea04b44046828bf8639175939582e852ae121c533ab9e94b2a5",
    ),
    "s23_train": (
        "m17/data/4b-s2/4b-LHS23SD/train.jsonl",
        "c629913c3570505d919f4ed48af49e4a24e50286d2991e5bec396f6fb7bec6a8",
    ),
    "s23_teacher": (
        "m17/data/4b-s2/4b-LHS23SD/teacher-s.jsonl",
        "a012a5e1ddcc4a8ce2c99658367e15e13550146bbb4833024f703a4fdc9b3f5c",
    ),
    "sdml_train": (
        "m15/data/4b/4b-LHA10SDML/train.jsonl",
        "fef6b036f33de6756dab083fd21ab462ec2975cfa63120f2452d3c9145d33dd4",
    ),
    "sdml_teacher": (
        "m15/data/4b/4b-LHA10SDML/teacher-ml.jsonl",
        "b95c5e635921b3f947e5c14627f6b21cf5ca5af913fcc48735471c79aa0793dd",
    ),
    "ib4p1": (
        "m17/inputs/ib4p1/ib4.train.jsonl",
        "6045b456d4b3032db4b76d70d792e3e9e30aad13e16a8f86fb3e99bc325806fb",
    ),
    "ib3r2": (
        "m18/inputs/ib3r2/m6/ib3/ib3.train.jsonl",
        "9d92d92a207109a585ceead7fa5e1bdb0b2c61859de8227a2a6058428f852dea",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    for name, (rel, want) in INPUTS.items():
        got = sha256(a.inputs / rel)
        if got != want:
            raise SystemExit(f"{name}: {rel} is {got}, not {want}")

    def read(name: str) -> list[bytes]:
        return (a.inputs / INPUTS[name][0]).read_bytes().splitlines(keepends=True)

    def ident(line: bytes) -> str:
        return json.loads(line)["id"]

    def is_copy(line: bytes) -> bool:
        return ident(line).endswith("~m2")

    s17, s23, ib4, ib3 = (
        read("s17_train"),
        read("s23_train"),
        read("ib4p1"),
        read("ib3r2"),
    )
    copies = [x for x in read("sdml_train") if is_copy(x)]
    copy_teacher = [x for x in read("sdml_teacher") if is_copy(x)]
    if [ident(x) for x in copies] != [ident(x) for x in copy_teacher]:
        raise SystemExit("the multilingual copies and their teacher rows differ")
    if b"".join(s17 + ib4 + ib3) != b"".join(read("s17ib4_train")):
        raise SystemExit("S17 + IB4 + IB3 is not M17's locked 4b-LHS17IB4 TRAIN")
    arms = {
        "4b-LHS17IB4ML": (
            s17,
            s17 + ib4 + ib3 + copies,
            read("s17_teacher") + copy_teacher,
        ),
        "4b-LHS23IB4": (s23, s23 + ib4 + ib3, read("s23_teacher")),
    }
    report = {
        "schema": "af-4b-w2data/1",
        "inputs": {k: {"path": rel, "sha256": s} for k, (rel, s) in INPUTS.items()},
        "arms": {},
    }
    for arm, (base, lines, teacher) in arms.items():
        ids = [ident(x) for x in lines]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{arm}: duplicate ids")
        out = a.out / arm
        out.mkdir(parents=True, exist_ok=False)
        (out / "train.jsonl").write_bytes(b"".join(lines))
        (out / "teacher-s.jsonl").write_bytes(b"".join(teacher))
        families: dict[str, int] = {}
        for x in lines[len(base) :]:
            fam = "ml-copy" if is_copy(x) else json.loads(x)["family"]
            families[fam] = families.get(fam, 0) + 1
        report["arms"][arm] = {
            "rows": len(lines),
            "base_rows": len(base),
            "added_rows": len(lines) - len(base),
            "added_families": families,
            "teacher_rows": len(teacher),
            "train_sha256": sha256(out / "train.jsonl"),
            "teacher_sha256": sha256(out / "teacher-s.jsonl"),
        }
    print(json.dumps(report))


if __name__ == "__main__":
    main()
