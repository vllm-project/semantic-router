#!/usr/bin/env bash
# Decision-2.0-Nox-4B Index-first release (worker 5e7b8132): node-A release inputs of the chosen candidate, CPU only,
# from the exact mirror holding this file. The candidate's formal run is already T = 1 (the 23:15 rule kept T = 1;
# spec m5-adapter-infer-dec-t1) and was scored on node A by m6-score.sh, so it is the scored run as it is.
#   current  this record's current/ copies of the current revision's (54b084f9, card round 3) sealed gate receipt and
#            final decision -> $IN/current/, checked against make_4bif.py's digests
#   gates    v2.eval.gates types / public231 / paired (vs the current revision's scored run dev2-4b-lh-t1-derived,
#            adopted Nox 1.0, Decider 4B) and lux9b.mlx_paired vs the current revision's mlx-diag run -> $IN/gates/;
#            each paired file is checked equal to the run's own same_panel compare where one exists
# Usage (node A): bash <mirror>/v2/release/records/dev2-4b-indexfirst-2026-10-02/ops/prep4b.sh CAND current|gates
set -euo pipefail
CAND=${1:?CAND} STAGE=${2:?STAGE}
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
R=$S/v2/release/records/dev2-4b-indexfirst-2026-10-02
CARD3=$S/v2/release/records/dev2-card3-2026-10-02
FR=/data/dev2/runs/dec/formal
case "$CAND" in
  a75) RUN=$FR/m16/m16-4b-LHA10SD-a75 LABEL="Decision-2.0-Nox-4B candidate M16 4b-LHA10SD-a75" ;;
  a50) RUN=$FR/m16/m16-4b-LHA10SD-a50 LABEL="Decision-2.0-Nox-4B candidate M16 4b-LHA10SD-a50" ;;
  UP) RUN=$FR/4bif/4bif-4b-LHA10UP LABEL="Decision-2.0-Nox-4B candidate M14 4b-LHA10UP" ;;
  SDB) RUN=$FR/m13/m13-4b-LHA10SD LABEL="Decision-2.0-Nox-4B candidate M13 4b-LHA10SD" ;;
  *) echo "bad CAND $CAND" >&2; exit 2 ;;
esac
MLXR=$RUN-mlx
MLXP=/data/dev2/private/panels/mlx-diag-v1
IN=/data/dev2/runs/release/inputs/dev2-4b-4bif-$CAND
GATES=$IN/gates
CUR=/data/dev2/runs/release/dev2-4b-lh-t1-derived
CURM=/data/dev2/runs/release/dev2-4b-lh-t1-derived-mlx
export PYTHONPATH=$S:$S/v2/9b
cd "$S"
case "$STAGE" in
  current)
    mkdir -p "$IN/current"
    for pair in "$CARD3/4b/release/receipts/gate.json:card3-4b-gate.json" \
      "$CARD3/Decision-2.0-Nox-4B.decision.card3.json:Decision-2.0-Nox-4B.decision.card3.json"; do
      src=${pair%%:*} dst=$IN/current/${pair#*:}
      if [[ -e "$dst" ]]; then cmp "$src" "$dst"; else cp "$src" "$dst"; chmod 444 "$dst"; fi
    done
    python3 - "$IN/current" "$R/ops/make_4bif.py" <<'PY'
import hashlib, re, sys
from pathlib import Path
text = Path(sys.argv[2]).read_text()
for name, key in (("card3-4b-gate.json", "gate_sha256"), ("Decision-2.0-Nox-4B.decision.card3.json", "decision_sha256")):
    want = re.search(rf'"{key}": "([0-9a-f]{{64}})"', text).group(1)
    got = hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest()
    assert got == want, (name, got)
    print(name, got[:12])
PY
    ;;
  gates)
    for f in "$RUN/REPORT.json" "$RUN/SEAL.json" "$RUN/TYPES.json" "$MLXR/output/mlx-diag.predictions.jsonl" \
      "$CUR/REPORT.json" "$CURM/output/mlx-diag.predictions.jsonl"; do
      [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
    done
    python3 - "$RUN/SEAL.json" "$RUN/output" <<'PY'
import hashlib, json, sys
seal = json.load(open(sys.argv[1]))
for panel, v in seal["panels"].items():
    got = hashlib.sha256(open(f"{sys.argv[2]}/{panel}.predictions.jsonl", "rb").read()).hexdigest()
    assert got == v["predictions_sha256"], (panel, got)
    print("sealed predictions match:", panel, got[:12])
PY
    mkdir -p "$GATES"
    [[ -f "$MLXR/mlx-diag.score.json" ]] || python3 -B -m v2.eval.multilingual_panel score --panel $MLXP \
      --predictions "$MLXR/output/mlx-diag.predictions.jsonl" --output "$MLXR/mlx-diag.score.json"
    python3 -B -m v2.eval.gates types --run "$RUN" --label "$LABEL" --output "$GATES/types.json"
    python3 -B -m v2.eval.gates public231 --left "$RUN" --right "$CUR" --left-name "$LABEL" \
      --right-name "Decision-2.0-Nox-4B (current, T = 1)" --output "$GATES/public231-vs-nox-4b.json"
    for cmp in "$CUR nox-4b" "/data/dev2/runs/eval/m1-adopt/nox1 adopted-1.0" \
      "/data/dev2/runs/eval/m1-adopt/decider4b decider4b"; do
      # shellcheck disable=SC2086
      set -- $cmp
      python3 -B -m v2.eval.gates paired --left "$RUN" --right "$1" --left-name "$LABEL" --right-name "$2" \
        --output "$GATES/paired-vs-$2.json"
      if [[ -f "$RUN/PAIRED-vs-$2.json" ]]; then
        python3 - "$GATES/paired-vs-$2.json" "$RUN/PAIRED-vs-$2.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert a["point"] == b["point"] and a["ci95"] == b["ci95"] and a["axis_ci95"] == b["axis_ci95"], "paired differs"
print("paired equal to same_panel compare:", sys.argv[1])
PY
      fi
    done
    python3 -B -m lux9b.mlx_paired --left "$MLXR" --right "$CURM" --panel $MLXP --left-name "$LABEL" \
      --right-name "Decision-2.0-Nox-4B (current, T = 1)" --output "$GATES/mlx-paired-vs-nox-4b.json"
    sha256sum "$GATES"/* "$MLXR/mlx-diag.score.json"
    ;;
  *) echo "usage: prep4b.sh CAND current|gates" >&2; exit 2 ;;
esac
