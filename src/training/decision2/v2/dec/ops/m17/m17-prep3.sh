#!/usr/bin/env bash
# Decoder M17 stage 2, wave 2 data on node F (prereg dec-m17-stage2-prereg-2026-10-02.md arm (a); amendment 1), CPU
# only (host python3, standard library), from an exact mirror. Idempotent.
#   4b-LHS17IB4   stage 1's locked 4b-LHS17SD TRAIN (byte for byte), then IB4 phase 1 TRAIN (every row, file order),
#                 then IB3-r2 TRAIN (every row, file order); teacher = a hard link of 4b-LHS17SD's (IB rows gold only);
#   4b-LHS17IB4X  the same without IB4's `isarc2` rows (the in-distribution family, kept separable).
# Checks: every input against its hash; ids unique across each output; then data/READY-m17s3.json, written once.
#
# usage: m17-prep3.sh <mirror-dir> <amendment-commit>
set -euo pipefail
AMEND=$2
M=/data/dev2/runs/dec/m17
D=$M/data/4b-s3
log() { echo "$(date -u +%FT%TZ) prep3 $*" | tee -a "$M/OPERATIONS.log"; }
[ -f "$M/data/READY-m17s3.json" ] && { log "READY-m17s3.json exists; nothing to do"; exit 0; }
mkdir -p "$D"
python3 - "$M" "$D" "$AMEND" << 'EOF' | tee "$D/report.json"
import hashlib, json, os, sys
from pathlib import Path
m, d, amend = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
S17 = m / "data/4b/4b-LHS17SD"
INPUTS = {
    "s17_train": (S17 / "train.jsonl", "14bce13ce926b354e214581e7cf4718d03f80c975b36dbce6731fbc6517e25a0"),
    "s17_teacher": (S17 / "teacher-s.jsonl", "374f4fa68c8ae32f7b85fc8beeb988a1e4de43be544d4cfbf68193b2fe2fcea2"),
    "ib4p1": (m / "inputs/ib4p1/ib4.train.jsonl", "6045b456d4b3032db4b76d70d792e3e9e30aad13e16a8f86fb3e99bc325806fb"),
    "ib3r2": (Path("/data/dev2/runs/dec/m18/inputs/ib3r2/m6/ib3/ib3.train.jsonl"),
              "9d92d92a207109a585ceead7fa5e1bdb0b2c61859de8227a2a6058428f852dea"),
}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
for name, (p, want) in INPUTS.items():
    if sha(p) != want:
        sys.exit(f"{name}: {p} is not {want}")
base = INPUTS["s17_train"][0].read_bytes().splitlines(keepends=True)
ib4 = INPUTS["ib4p1"][0].read_bytes().splitlines(keepends=True)
ib3 = INPUTS["ib3r2"][0].read_bytes().splitlines(keepends=True)
fam = lambda line: json.loads(line)["family"]
arms = {"4b-LHS17IB4": base + ib4 + ib3, "4b-LHS17IB4X": base + [x for x in ib4 if fam(x) != "isarc2"] + ib3}
report = {"schema": "dec-m17-s3data/1", "inputs": {k: {"path": str(p), "sha256": s} for k, (p, s) in INPUTS.items()},
          "arms": {}}
lock = {"schema": "dec-m17-ready/1", "record": "dec-m17-stage2-amendment-1-2026-10-02.md", "record_commit": amend,
        "arms": {}, "teachers": {}, "weights": {}}
for arm, lines in arms.items():
    ids = [json.loads(x)["id"] for x in lines]
    if len(set(ids)) != len(ids):
        sys.exit(f"{arm}: duplicate ids")
    out = d / arm
    out.mkdir(exist_ok=True)
    (out / "train.jsonl").write_bytes(b"".join(lines))
    if not (out / "teacher-s.jsonl").exists():
        os.link(INPUTS["s17_teacher"][0], out / "teacher-s.jsonl")
    added = lines[len(base):]
    fams = {}
    for x in added:
        fams[fam(x)] = fams.get(fam(x), 0) + 1
    report["arms"][arm] = {"rows": len(lines), "base_rows": len(base), "added_rows": len(added),
                           "added_families": fams, "train_sha256": sha(out / "train.jsonl")}
    lock["arms"][arm] = report["arms"][arm]["train_sha256"]
    lock["teachers"][arm] = sha(out / "teacher-s.jsonl")
with open(m / "data/READY-m17s3.json", "x") as f:
    json.dump(lock, f, indent=1)
print(json.dumps(report))
EOF
log "wave-2 TRAIN built: $(tr -d '\n' < "$D/report.json" | cut -c1-900)"
