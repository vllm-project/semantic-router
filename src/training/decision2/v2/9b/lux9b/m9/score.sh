#!/usr/bin/env bash
# 9B M9 node-A scoring (prereg "Development readouts" / "Development gates"), CPU only, host python3, from an exact
# mirror. Gold stays on node A. A screen whose predictions are missing is skipped (logged); a failed screen stops.
#
#   score.sh probes-panel                    the M9 probe panel (m9_probes.py): M10 panel minus x60 overlaps
#   score.sh points <REF> <point> [...]      per point vs REF: HT-DEV v2 (m7_htdev2.py), Score5-typed-DEV
#                                            (m8_rules.py score5t; also REF), retention probes (m10_probes.py score),
#                                            hs1-dev (v2.data.hs1.validity), PN1 dev (lux9b.m7_rules pn1),
#                                            MLX-DEV-9B (v2.dec.mlx_dev score / compare REF -> point)
#   score.sh contrast <A> <B>                HT-DEV v2, probes and PN1 of A vs B (report only)
#   score.sh readout <REF> <point> [...]     typed DEV + CSS pilot (v2.dec.dev_readout) -> lines/readout/m9.json
#   score.sh rules <X=REF> [...] [-- <A:B> ...]   m9_rules.py -> select/9b-finalists.json (runs once)
# M9_READOUT / M9_RULES rename the readout and rules outputs (stage 2: m9-s2 / 9b-finalists-s2).
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
M=/data/dev2/runs/9b/m9
L=$M/lines
OPS=$S/v2/dec/ops
GOLD=/data/dev2/private/panels/gold
HT_GOLD=$GOLD/ht-dev2.gold.jsonl HT_GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
P9=/data/dev2/private/9b/m9
PROBE_GOLD=$P9/m9-probes.gold.jsonl
PN1_GOLD=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/27b1d2f130292268b43a618584bebab5d4e4a6b5/m4/pn1/dev/pn1.dev.gold.jsonl
MLX_INDEX=/data/dev2/runs/9b/m7/data/mlxdev/build/panel.jsonl.index.jsonl
log() { echo "$(date -u +%FT%TZ) score $*" | tee -a "$L/OPERATIONS-score.log"; }
die() { log "$*"; exit 1; }
py() { (cd "$S" && PYTHONPATH=$S:$S/v2/9b python3 -B "$@"); }
pred() { echo "$L/$1/$2/$2.predictions.jsonl"; }
have() { [ -s "$(pred "$1" "$2")" ] || { log "$1: no $2 predictions; screen skipped"; return 1; }; }
mkdir -p "$L/diag" "$L/readout" "$M/select"

ht() {  # <left> <right> <out name>
  local out=$L/diag/$3.htdev2.json
  [ -f "$out" ] && return 0
  have "$1" ht-dev2 && have "$2" ht-dev2 || return 0
  py "$OPS/m7/m7_htdev2.py" --gold "$HT_GOLD" --left "$(pred "$1" ht-dev2)" --left-name "$1" \
    --right "$(pred "$2" ht-dev2)" --right-name "$2" --output "$out" > "$L/diag/$3.htdev2.log" 2>&1 \
    || die "$3 HT-DEV v2 FAILED (see $L/diag/$3.htdev2.log)"
  log "$3 ht-dev2: $(tail -1 "$L/diag/$3.htdev2.log" | cut -c1-300)"
}
probes() {  # <left> <right> <out name>
  local out=$L/diag/$3.probes.json
  [ -f "$out" ] && return 0
  have "$1" m9-probes && have "$2" m9-probes || return 0
  py "$OPS/m10/m10_probes.py" score --gold "$PROBE_GOLD" --predictions "$(pred "$1" m9-probes)" \
    --reference "$(pred "$2" m9-probes)" --output "$out" > "$L/diag/$3.probes.log" 2>&1 \
    || die "$3 probes FAILED (see $L/diag/$3.probes.log)"
  log "$3 probes: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ("macro_mmlu_arc_gsm8k","macro_delta","macro_delta_ci95")}, {p: d[p]["accuracy"] for p in ("mmlu","arc","gsm8k") if p in d})' "$out")"
}
s5() {  # <point>
  local out=$L/diag/$1.score5t.json
  [ -f "$out" ] && return 0
  have "$1" score5t-dev || return 0
  py "$OPS/m8/m8_rules.py" score5t --panel-root /data/dev2/private/panels --predictions "$(pred "$1" score5t-dev)" \
    --output "$out" > "$L/diag/$1.score5t.log" 2>&1 || die "$1 Score5-typed-DEV FAILED (see $L/diag/$1.score5t.log)"
  log "$1 score5t: $(tail -1 "$L/diag/$1.score5t.log" | cut -c1-300)"
}
hs1() {  # <left> <right>
  local out=$L/diag/$1.hs1.json
  [ -f "$out" ] && return 0
  have "$1" hs1-dev && have "$2" hs1-dev || return 0
  py -m v2.data.hs1.validity --gold "$GOLD/hs1-dev.gold.jsonl" --left "$(pred "$1" hs1-dev)" --left-name "$1" \
    --right "$(pred "$2" hs1-dev)" --right-name "$2" --output "$out" > "$L/diag/$1.hs1.log" 2>&1 \
    || die "$1 hs1-dev FAILED (see $L/diag/$1.hs1.log)"
  log "$1 hs1-dev false-yes: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))["families"]["hs1_unmet_condition"]["diagnostics"]; print(d)' "$out")"
}
pn1() {  # <left> <right> <out name>
  local out=$L/diag/$3.pn1.json
  [ -f "$out" ] && return 0
  have "$1" pn1-dev && have "$2" pn1-dev || return 0
  py -m lux9b.m7_rules pn1 --gold "$PN1_GOLD" --predictions "$(pred "$1" pn1-dev)" \
    --reference-predictions "$(pred "$2" pn1-dev)" --label "$1" --reference-label "$2" --output "$out" \
    > "$L/diag/$3.pn1.log" 2>&1 || die "$3 PN1 FAILED (see $L/diag/$3.pn1.log)"
  log "$3 pn1: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["delta"], d["delta_ci95"])' "$out")"
}
mlx() {  # <left> <right>
  local out=$L/diag/$1.mlxdev.json a=$L/$2/mlxdev/mlxdev-predictions.jsonl b=$L/$1/mlxdev/mlxdev-predictions.jsonl
  [ -f "$out" ] && return 0
  [ -s "$a" ] && [ -s "$b" ] || { log "$1: no mlxdev predictions; screen skipped"; return 0; }
  py -m v2.dec.mlx_dev score --index "$MLX_INDEX" --predictions "$b" --output "$L/diag/$1.mlxdev.score.json" \
    > "$L/diag/$1.mlxdev.score.log" 2>&1 || die "$1 MLX-DEV-9B score FAILED"
  py -m v2.dec.mlx_dev compare --index "$MLX_INDEX" --a "$a" --b "$b" --output "$out" > "$L/diag/$1.mlxdev.log" 2>&1 \
    || die "$1 MLX-DEV-9B compare FAILED (see $L/diag/$1.mlxdev.log)"
  log "$1 mlxdev: $(python3 -c 'import json,sys; m=json.load(open(sys.argv[1]))["metrics"]; print({k: (round(m[k]["diff"],4), m[k]["ci95"]) for k in ("noul_ml","choice_ml","score_ml") if k in m})' "$out")"
}

case ${1:-} in
  probes-panel)
    mkdir -p "$P9" "$M/panels" "$M/probes-build"
    py "$S/v2/9b/lux9b/m9_probes.py" --prompts "$M/inputs/m10-probes.prompts.jsonl" \
      --gold /data/dev2/private/dec/m10/m10-probes.gold.jsonl --train /data/dev2/runs/9b/m4/data/m4-k-xl-r2-60m/build/train.jsonl \
      --work "$M/probes-build" --out-prompts "$M/panels/m9-probes.prompts.jsonl" --out-gold "$PROBE_GOLD" \
      --report "$M/probes-build/m9-probes-build.json" > "$M/probes-build/build.log" 2>&1 || die "probe panel FAILED"
    log "probe panel: $(cat "$M/probes-build/m9-probes-build.json" | tr -d '\n' | cut -c1-600)"
    ;;
  points)
    ref=$2
    shift 2
    [ "$(sha256sum "$HT_GOLD" | cut -d' ' -f1)" = "$HT_GOLD_SHA" ] || die "HT-DEV v2 gold is not $HT_GOLD_SHA"
    s5 "$ref"
    for p in "$@"; do
      [ "$p" = "$ref" ] && continue
      ht "$p" "$ref" "$p"
      s5 "$p"
      probes "$p" "$ref" "$p"
      hs1 "$p" "$ref"
      pn1 "$p" "$ref" "$p"
      mlx "$p" "$ref"
    done
    ;;
  contrast)
    ht "$2" "$3" "$2-vs-$3"
    probes "$2" "$3" "$2-vs-$3"
    pn1 "$2" "$3" "$2-vs-$3"
    ;;
  readout)
    ref=$2
    shift 2
    args=(--arm "$ref=$(pred "$ref" dev),$(pred "$ref" css-pilot)")
    for p in "$@"; do
      [ "$p" = "$ref" ] && continue
      args+=(--arm "$p=$(pred "$p" dev),$(pred "$p" css-pilot)" --compare "$ref:$p")
    done
    out=$L/readout/${M9_READOUT:-m9}.json
    [ -f "$out" ] && mv "$out" "$L/readout/${M9_READOUT:-m9}.$(date -u +%Y%m%dT%H%M%SZ).json"
    py -m v2.dec.dev_readout --typed-gold "$GOLD/typed-dev.gold.jsonl" --css-gold "$GOLD/css-pilot.gold.jsonl" \
      "${args[@]}" --output "$out" > "$L/readout/${M9_READOUT:-m9}.log" 2>&1 || die "dev_readout FAILED (see $L/readout/${M9_READOUT:-m9}.log)"
    log "readout $(sha256sum "$out" | cut -c1-16) ($ref $*)"
    ;;
  rules)
    shift
    pts=() cons=()
    while [ $# -gt 0 ]; do
      [ "$1" = -- ] && { shift; break; }
      pts+=(--point "$1")
      shift
    done
    for c in "$@"; do cons+=(--contrast "$c"); done
    rules=${M9_RULES:-9b-finalists}
    py "$S/v2/9b/lux9b/m9_rules.py" --lines-root "$L" --readout "$L/readout/${M9_READOUT:-m9}.json" "${pts[@]}" \
      "${cons[@]}" --output "$M/select/$rules.json" > "$M/select/$rules.log" 2>&1 \
      || die "rules FAILED (see $M/select/$rules.log)"
    log "finalists ($rules): $(tail -1 "$M/select/$rules.log" | cut -c1-800)"
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
