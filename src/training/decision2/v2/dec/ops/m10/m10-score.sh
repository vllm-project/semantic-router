#!/usr/bin/env bash
# Decoder M10 node-A scoring (prereg "Development readouts" / "Development gates"), CPU only, from an exact mirror.
# Gold stays on node A; node E / F readouts are pulled gold-free over the temporary transfer key.
#
#   m10-score.sh pull <user@node> <point> [<point> ...]   lines/<point>/ predictions and manifests from node E / F
#   m10-score.sh points <REF> <point> [<point> ...]       per point vs REF: HT-DEV v2 (m7_htdev2.py), Score5-typed-DEV
#                                                         (m8_rules.py score5t; also REF), retention probes
#                                                         (m10_probes.py score); hs1-dev (v2.data.hs1.validity, report)
#   m10-score.sh contrast <A> <B>                         HT-DEV v2 and probes of A vs B (A-vs-B.*.json; report only)
#   m10-score.sh readout <REF> <point> [<point> ...]      typed DEV + CSS pilot (v2.dec.dev_readout) -> readout/m10.json
#   m10-score.sh rules <X=REF> [...] [-- <A:B> ...]       m10_rules.py -> m10/select/4b-finalists.json (runs once)
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
M=/data/dev2/runs/dec/m10
L=$M/lines
OPS=$S/v2/dec/ops
GOLD=/data/dev2/private/panels/gold
HT_GOLD=$GOLD/ht-dev2.gold.jsonl HT_GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
PROBE_GOLD=/data/dev2/private/dec/m10/m10-probes.gold.jsonl
PROBE_GOLD_SHA=5c674e35142bc1aae2526d38b6930a0e570d99b5ca5185b0a7ac428dc8b7bae2
log() { echo "$(date -u +%FT%TZ) score $*" | tee -a "$L/OPERATIONS-nodeA.log"; }
die() { log "$*"; exit 1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
pred() { echo "$L/$1/$2/$2.predictions.jsonl"; }
mkdir -p "$L/diag" "$L/readout" "$M/select"

ht() {  # <left> <right> <out name>
  local out=$L/diag/$3.htdev2.json
  [ -f "$out" ] && return 0
  py "$OPS/m7/m7_htdev2.py" --gold "$HT_GOLD" --left "$(pred "$1" ht-dev2)" --left-name "$1" \
    --right "$(pred "$2" ht-dev2)" --right-name "$2" --output "$out" > "$L/diag/$3.htdev2.log" 2>&1 \
    || die "$3 HT-DEV v2 FAILED (see $L/diag/$3.htdev2.log)"
  log "$3 ht-dev2: $(tail -1 "$L/diag/$3.htdev2.log" | cut -c1-300)"
}
probes() {  # <left> <right> <out name>
  local out=$L/diag/$3.probes.json
  [ -f "$out" ] && return 0
  py "$OPS/m10/m10_probes.py" score --gold "$PROBE_GOLD" --predictions "$(pred "$1" m10-probes)" \
    --reference "$(pred "$2" m10-probes)" --output "$out" > "$L/diag/$3.probes.log" 2>&1 \
    || die "$3 probes FAILED (see $L/diag/$3.probes.log)"
  log "$3 probes: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ("macro_mmlu_arc_gsm8k","macro_delta","macro_delta_ci95")})' "$out")"
}
s5() {  # <point>
  local out=$L/diag/$1.score5t.json
  [ -f "$out" ] && return 0
  py "$OPS/m8/m8_rules.py" score5t --panel-root /data/dev2/private/panels --predictions "$(pred "$1" score5t-dev)" \
    --output "$out" > "$L/diag/$1.score5t.log" 2>&1 || die "$1 Score5-typed-DEV FAILED (see $L/diag/$1.score5t.log)"
  log "$1 score5t: $(tail -1 "$L/diag/$1.score5t.log" | cut -c1-300)"
}
hs1() {  # <left> <right>
  local out=$L/diag/$1.hs1.json
  [ -f "$out" ] || [ ! -f "$(pred "$1" hs1-dev)" ] && return 0
  py -m v2.data.hs1.validity --gold "$GOLD/hs1-dev.gold.jsonl" --left "$(pred "$1" hs1-dev)" --left-name "$1" \
    --right "$(pred "$2" hs1-dev)" --right-name "$2" --output "$out" > "$L/diag/$1.hs1.log" 2>&1 \
    || log "$1 hs1-dev score failed (report only; see $L/diag/$1.hs1.log)"
}

case ${1:-} in
  pull)
    from=$2
    shift 2
    for p in "$@"; do
      rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' --include='*/' --include='*.predictions.jsonl' \
        --include='*.manifest.json' --include='*.launch.json' --include='parity-*.json' --exclude='*' \
        "$from:$L/$p/" "$L/$p/" || die "pull of $p failed"
      log "pulled $p: $(find "$L/$p" -name '*.predictions.jsonl' | wc -l) prediction files"
    done
    ;;
  points)
    ref=$2
    shift 2
    [ "$(sha256sum "$HT_GOLD" | cut -d' ' -f1)" = "$HT_GOLD_SHA" ] || die "HT-DEV v2 gold is not $HT_GOLD_SHA"
    [ "$(sha256sum "$PROBE_GOLD" | cut -d' ' -f1)" = "$PROBE_GOLD_SHA" ] || die "probe gold is not $PROBE_GOLD_SHA"
    s5 "$ref"
    for p in "$@"; do
      [ "$p" = "$ref" ] && continue
      ht "$p" "$ref" "$p"
      s5 "$p"
      probes "$p" "$ref" "$p"
      hs1 "$p" "$ref"
    done
    ;;
  contrast)
    ht "$2" "$3" "$2-vs-$3"
    probes "$2" "$3" "$2-vs-$3"
    ;;
  readout)
    ref=$2
    shift 2
    args=(--arm "$ref=$(pred "$ref" dev),$(pred "$ref" css-pilot)")
    for p in "$@"; do
      [ "$p" = "$ref" ] && continue
      args+=(--arm "$p=$(pred "$p" dev),$(pred "$p" css-pilot)" --compare "$ref:$p")
    done
    out=$L/readout/m10.json
    [ -f "$out" ] && mv "$out" "$L/readout/m10.$(date -u +%Y%m%dT%H%M%SZ).json"
    py -m v2.dec.dev_readout --typed-gold "$GOLD/typed-dev.gold.jsonl" --css-gold "$GOLD/css-pilot.gold.jsonl" \
      "${args[@]}" --output "$out" > "$L/readout/m10.log" 2>&1 || die "dev_readout FAILED (see $L/readout/m10.log)"
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
    py "$OPS/m10/m10_rules.py" --lines-root "$L" --readout "$L/readout/m10.json" "${pts[@]}" "${cons[@]}" \
      --output "$M/select/4b-finalists.json" > "$M/select/rules.log" 2>&1 || die "rules FAILED (see $M/select/rules.log)"
    log "finalists: $(tail -1 "$M/select/rules.log" | cut -c1-600)"
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
