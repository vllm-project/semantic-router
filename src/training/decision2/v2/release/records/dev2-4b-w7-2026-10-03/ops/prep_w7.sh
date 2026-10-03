#!/usr/bin/env bash
# Decision-2.0-Nox-4B wave-7 release (4B owner, decoder M17b): node-A release inputs, CPU only, from the exact mirror
# holding this file, as the wave-6 release's prep_w6.sh. The formal run is T = 1 and scored on node A by
# m17-fscore.sh, so it is the scored run as it is.
#   current  the current revision's (c60d3b5c, 4b-LHS17IB4-lrh) sealed gate receipt and final decision from the record
#            dev2-4b-w6-2026-10-03 -> $IN/current/, checked against make_w7.py
#   gates    v2.eval.gates types / public231 / paired (vs the current weights' scored run m17-4b-LHS17IB4-lrh, adopted
#            Nox 1.0, Decider 4B) and lux9b.mlx_paired vs its mlx-diag run -> $IN/gates/; each paired file is checked
#            equal to the run's own same_panel compare where one exists
# Usage (node A): bash <mirror>/v2/release/records/dev2-4b-w7-2026-10-03/ops/prep_w7.sh CAND current|gates
set -euo pipefail
CAND=${1:?CAND} STAGE=${2:?STAGE}
[[ "$CAND" =~ ^(LRHxXALL-m50|LRHxXALL-m75|LRHxALL|LRHxALL-L2|LRHxQ|LRQxLRH|SDMLIB4-lrh|LHS17ML-lrh|LHS17IB4X-lrh|SDML-lrh|LHS17IB4-lrq|SDMLIB4-UP|LHS17IB4-UP|SDMLIB4W2)$ ]] || { echo "CAND: a wave-7 candidate (amendments 1 and 2)" >&2; exit 2; }
KEY=${CAND,,}
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
R=$S/v2/release/records/dev2-4b-w7-2026-10-03
XR=$S/v2/release/records/dev2-4b-w6-2026-10-03
FR=/data/dev2/runs/dec/formal
RUN=$FR/m17/m17-4b-$CAND LABEL="Decision-2.0-Nox-4B candidate 4b-$CAND"
MLXR=$RUN-mlx
MLXP=/data/dev2/private/panels/mlx-diag-v1
IN=/data/dev2/runs/release/inputs/dev2-4b-$KEY
GATES=$IN/gates
CUR=$FR/m17/m17-4b-LHS17IB4-lrh
CURM=$CUR-mlx
export PYTHONPATH=$S:$S/v2/9b
cd "$S"
case "$STAGE" in
  current)
    mkdir -p "$IN/current"
    for pair in "$XR/release/receipts/gate.json:current/lhs17ib4-lrh-4b-gate.json" \
      "$XR/Decision-2.0-Nox-4B.decision.lhs17ib4-lrh.json:current/Decision-2.0-Nox-4B.decision.lhs17ib4-lrh.json"; do
      src=${pair%%:*} dst=$IN/${pair#*:}
      if [[ -e "$dst" ]]; then cmp "$src" "$dst"; else cp "$src" "$dst"; chmod 444 "$dst"; fi
    done
    python3 - "$IN/current" "$R/ops/make_w7.py" <<'PY'
import hashlib, re, sys
from pathlib import Path
text = Path(sys.argv[2]).read_text()
for name, key in (("lhs17ib4-lrh-4b-gate.json", "gate_sha256"), ("Decision-2.0-Nox-4B.decision.lhs17ib4-lrh.json", "decision_sha256")):
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
  *) echo "usage: prep_w7.sh CAND current|gates" >&2; exit 2 ;;
esac
