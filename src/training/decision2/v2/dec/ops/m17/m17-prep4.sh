#!/usr/bin/env bash
# Decoder M17 stage 2, wave 3 data on node F (amendment 2: the released M15 4b-LHA10SDML is the base), CPU only (host
# python3, standard library), from an exact mirror. Idempotent.
#   4b-SDMLIB4  the released 4b-LHA10SDML TRAIN (M15 lock, byte for byte), then IB4 phase 1 TRAIN (every row, file
#               order), then IB3-r2 TRAIN (every row, file order); teacher = a hard link of 4b-LHA10SDML's (IB rows gold
#               only) - stage 2's arm (a) on the new base;
#   4b-LHS17ML  stage 1's locked 4b-LHS17SD TRAIN (byte for byte), then 4b-LHA10SDML's multilingual copies (its `~m2`
#               rows, file order); teacher = 4b-LHS17SD's, then those copies' teacher rows from 4b-LHA10SDML's (file
#               order) - the swap and the multilingual copies combined.
# Checks: every input against its hash; ids unique across each output; every copy has one teacher row; then
# data/READY-m17s4.json, written once.
#
# usage: m17-prep4.sh <mirror-dir> <amendment-commit>
set -euo pipefail
AMEND=$2
M=/data/dev2/runs/dec/m17
D=$M/data/4b-s4
log() { echo "$(date -u +%FT%TZ) prep4 $*" | tee -a "$M/OPERATIONS.log"; }
[ -f "$M/data/READY-m17s4.json" ] && { log "READY-m17s4.json exists; nothing to do"; exit 0; }
mkdir -p "$D"
python3 - "$M" "$D" "$AMEND" << 'EOF' | tee "$D/report.json"
import hashlib, json, os, sys
from pathlib import Path
m, d, amend = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
S17 = m / "data/4b/4b-LHS17SD"
SDML = Path("/data/dev2/runs/dec/m15/data/4b/4b-LHA10SDML")
INPUTS = {
    "s17_train": (S17 / "train.jsonl", "14bce13ce926b354e214581e7cf4718d03f80c975b36dbce6731fbc6517e25a0"),
    "s17_teacher": (S17 / "teacher-s.jsonl", "374f4fa68c8ae32f7b85fc8beeb988a1e4de43be544d4cfbf68193b2fe2fcea2"),
    "sdml_train": (SDML / "train.jsonl", "fef6b036f33de6756dab083fd21ab462ec2975cfa63120f2452d3c9145d33dd4"),
    "sdml_teacher": (SDML / "teacher-ml.jsonl", "b95c5e635921b3f947e5c14627f6b21cf5ca5af913fcc48735471c79aa0793dd"),
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
read = lambda name: INPUTS[name][0].read_bytes().splitlines(keepends=True)
ident = lambda line: json.loads(line)["id"]
copy = lambda line: ident(line).endswith("~m2")
sdml, s17, ib4, ib3 = read("sdml_train"), read("s17_train"), read("ib4p1"), read("ib3r2")
copies = [x for x in sdml if copy(x)]
copy_teacher = [x for x in read("sdml_teacher") if copy(x)]
if [ident(x) for x in copies] != [ident(x) for x in copy_teacher]:
    sys.exit("the multilingual copies and their teacher rows differ")
arms = {
    "4b-SDMLIB4": (sdml, sdml + ib4 + ib3, ("link", INPUTS["sdml_teacher"][0])),
    "4b-LHS17ML": (s17, s17 + copies, ("write", b"".join(read("s17_teacher") + copy_teacher))),
}
report = {"schema": "dec-m17-s4data/1", "inputs": {k: {"path": str(p), "sha256": s} for k, (p, s) in INPUTS.items()},
          "arms": {}}
lock = {"schema": "dec-m17-ready/1", "record": "dec-m17-stage2-amendment-2-2026-10-02.md", "record_commit": amend,
        "arms": {}, "teachers": {}, "weights": {}}
for arm, (base, lines, (how, teacher)) in arms.items():
    ids = [ident(x) for x in lines]
    if len(set(ids)) != len(ids):
        sys.exit(f"{arm}: duplicate ids")
    out = d / arm
    out.mkdir(exist_ok=True)
    (out / "train.jsonl").write_bytes(b"".join(lines))
    if not (out / "teacher-s.jsonl").exists():
        if how == "link":
            os.link(teacher, out / "teacher-s.jsonl")
        else:
            (out / "teacher-s.jsonl").write_bytes(teacher)
    added = lines[len(base):]
    fams = {}
    for x in added:
        f = "ml-copy" if copy(x) else json.loads(x)["family"]
        fams[f] = fams.get(f, 0) + 1
    report["arms"][arm] = {"rows": len(lines), "base_rows": len(base), "added_rows": len(added),
                           "added_families": fams, "train_sha256": sha(out / "train.jsonl"),
                           "teacher_rows": sum(1 for _ in open(out / "teacher-s.jsonl", "rb"))}
    lock["arms"][arm] = report["arms"][arm]["train_sha256"]
    lock["teachers"][arm] = sha(out / "teacher-s.jsonl")
with open(m / "data/READY-m17s4.json", "x") as f:
    json.dump(lock, f, indent=1)
print(json.dumps(report))
EOF
log "wave-3 TRAIN built: $(tr -d '\n' < "$D/report.json" | cut -c1-900)"
