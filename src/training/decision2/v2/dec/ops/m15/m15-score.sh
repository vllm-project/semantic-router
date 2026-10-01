#!/usr/bin/env bash
# Decoder M15 node-A scoring (prereg dec-m15-prereg-2026-10-01.md, "Development gates"), CPU only, from an exact mirror.
# Gold stays on node A; node E / F readouts are pulled gold-free over the temporary transfer key. Points are named
# <tier>-<name> (tier 2b, 08b or 4b); the retention probes are scored on the tier's M15 probe gold (M10's gold minus the
# probe items that hit any M15 TRAIN file of the tier); IB DEV on M11's panel gold (private/dec/m11/ib-dev.gold.jsonl).
#
#   m15-score.sh pull <user@node> <point> [<point> ...]   lines/<point>/ predictions, manifests and MLX-DEV compares
#   m15-score.sh probe-gold <user@node> [<tier> ...]      m15/probes/hits-<tier>.json from node E (ids only), then the
#                                                         tier probe gold files (private/dec/m15/); default 2b 08b 4b
#   m15-score.sh ibdev <REF> <point> [<point> ...]        IB DEV (12 families) and transfer (9 families) macros vs REF
#                                                         -> diag/<point>.ibdev.json (m11_ibdev.py score)
#   m15-score.sh points <REF> <point> [<point> ...]       per point vs REF: HT-DEV v2 (m7_htdev2.py), Score5-typed-DEV
#                                                         (m8_rules.py score5t; also REF), retention probes
#                                                         (m10_probes.py score, tier gold); hs1-dev (v2.data.hs1.validity)
#   m15-score.sh contrast <A> <B>                         HT-DEV v2 and probes of A vs B (A-vs-B.*.json; report only)
#   m15-score.sh readout <tier> <REF> <point> [...]       typed DEV + CSS pilot (v2.dec.dev_readout) -> readout/<tier>.json
#   m15-score.sh rules <tier> <X=REF> [...] [-- <A:B> ...] m15_rules.py -> m15/select/<tier>-finalists.json (runs once)
set -u
S=$(cd "$(dirname "$0")/../../../.." && pwd)
MIRROR=${S%/src/training/decision2}
[ -f "$MIRROR/.dev2-mirror.json" ] || { echo "run from an exact mirror under /data/dev2/src" >&2; exit 2; }
M=/data/dev2/runs/dec/m15
L=$M/lines
OPS=$S/v2/dec/ops
GOLD=/data/dev2/private/panels/gold
HT_GOLD=$GOLD/ht-dev2.gold.jsonl HT_GOLD_SHA=659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc
PROBE_GOLD_M10=/data/dev2/private/dec/m10/m10-probes.gold.jsonl
PROBE_GOLD_SHA=5c674e35142bc1aae2526d38b6930a0e570d99b5ca5185b0a7ac428dc8b7bae2
PG=/data/dev2/private/dec/m15
IB_GOLD=/data/dev2/private/dec/m11/ib-dev.gold.jsonl
IB_GOLD_SHA=a69cf36d9fa6f85b1c1139dd3ecc1d613479ff6ed589675e9bc4ef3af0adb1d3
log() { echo "$(date -u +%FT%TZ) score $*" | tee -a "$L/OPERATIONS-nodeA.log"; }
die() { log "$*"; exit 1; }
py() { (cd "$S" && PYTHONPATH=$S python3 -B "$@"); }
pred() { echo "$L/$1/$2/$2.predictions.jsonl"; }
mkdir -p "$L/diag" "$L/readout" "$M/select"
# one scoring process at a time (M11 amendment 2: two concurrent runs raced the run-once rules step)
exec 9> "$M/select/.m15-score.lock"
flock -n 9 || die "another m15-score.sh is running"

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
  local gold=$PG/m10-probes.${1%%-*}.gold.jsonl
  [ -f "$gold" ] || die "no tier probe gold $gold (m15-score.sh probe-gold)"
  # the tier gold drops the probe items that hit the tier's TRAIN; the scorer expects gold for every prediction,
  # so both sides' predictions are restricted to the tier gold's ids first
  local lp=$L/diag/$3.left.m10-probes.jsonl rp=$L/diag/$3.right.m10-probes.jsonl
  python3 - "$gold" "$(pred "$1" m10-probes)" "$lp" "$(pred "$2" m10-probes)" "$rp" << 'EOF2' || die "$3 probe filter FAILED"
import json, sys
ids = {json.loads(line)["id"] for line in open(sys.argv[1])}
for src, dst in ((sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5])):
    kept = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if json.loads(line)["id"] in ids:
                out.write(line)
                kept += 1
    if kept != len(ids):
        raise SystemExit(f"{src}: {kept} predictions for {len(ids)} gold items")
EOF2
  py "$OPS/m10/m10_probes.py" score --gold "$gold" --predictions "$lp" \
    --reference "$rp" --output "$out" > "$L/diag/$3.probes.log" 2>&1 \
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
        --include='*.manifest.json' --include='*.launch.json' --include='parity-*.json' \
        --include='mlxdev-predictions.jsonl' --include='*.mlxdev.json' --include='*.mlxscore.json' --exclude='*' \
        "$from:$L/$p/" "$L/$p/" || die "pull of $p failed"
      log "pulled $p: $(find "$L/$p" -name '*.predictions.jsonl' | wc -l) prediction files"
    done
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
    done
    ;;
  probe-gold)
    from=$2
    shift 2
    [ "$(sha256sum "$PROBE_GOLD_M10" | cut -d' ' -f1)" = "$PROBE_GOLD_SHA" ] || die "M10 probe gold is not $PROBE_GOLD_SHA"
    mkdir -p "$M/probes" "$PG"
    [ $# -gt 0 ] || set -- 2b 08b 4b
    for t in "$@"; do
      rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' "$from:$M/probes/hits-$t.json" "$M/probes/" \
        || die "pull of hits-$t.json failed"
      [ -f "$PG/m10-probes.$t.gold.jsonl" ] && continue
      python3 - "$M/probes/hits-$t.json" "$PROBE_GOLD_M10" "$PG/m10-probes.$t.gold.jsonl" << 'EOF2' || die "tier gold $t FAILED"
import json, sys
hits = set(json.load(open(sys.argv[1]))["panel_hits"])
kept = dropped = 0
with open(sys.argv[2]) as src, open(sys.argv[3], "x") as out:
    for line in src:
        if json.loads(line)["id"] in hits:
            dropped += 1
            continue
        out.write(line)
        kept += 1
print(kept, dropped)
EOF2
      log "tier probe gold $t: $(sha256sum "$PG/m10-probes.$t.gold.jsonl" | cut -c1-16) ($(wc -l < "$PG/m10-probes.$t.gold.jsonl") items)"
    done
    ;;
  ibdev)
    ref=$2
    shift 2
    [ "$(sha256sum "$IB_GOLD" | cut -d' ' -f1)" = "$IB_GOLD_SHA" ] || die "IB DEV gold is not $IB_GOLD_SHA"
    for p in "$@"; do
      out=$L/diag/$p.ibdev.json
      [ -f "$out" ] && continue
      py "$OPS/m11/m11_ibdev.py" score --gold "$IB_GOLD" --predictions "$(pred "$p" ib-dev)" --name "$p" \
        --reference "$(pred "$ref" ib-dev)" --reference-name "$ref" --exclude isarc,hover,gsm2 --output "$out" \
        > "$L/diag/$p.ibdev.log" 2>&1 || die "$p IB DEV FAILED (see $L/diag/$p.ibdev.log)"
      log "$p ib-dev: $(tail -1 "$L/diag/$p.ibdev.log" | cut -c1-300)"
    done
    ;;
  contrast)
    ht "$2" "$3" "$2-vs-$3"
    probes "$2" "$3" "$2-vs-$3"
    ;;
  readout)
    tier=$2 ref=$3
    shift 3
    args=(--arm "$ref=$(pred "$ref" dev),$(pred "$ref" css-pilot)")
    for p in "$@"; do
      [ "$p" = "$ref" ] && continue
      args+=(--arm "$p=$(pred "$p" dev),$(pred "$p" css-pilot)" --compare "$ref:$p")
    done
    out=$L/readout/$tier.json
    [ -f "$out" ] && mv "$out" "$L/readout/$tier.$(date -u +%Y%m%dT%H%M%SZ).json"
    py -m v2.dec.dev_readout --typed-gold "$GOLD/typed-dev.gold.jsonl" --css-gold "$GOLD/css-pilot.gold.jsonl" \
      "${args[@]}" --output "$out" > "$L/readout/$tier.log" 2>&1 || die "dev_readout FAILED (see $L/readout/$tier.log)"
    log "readout $(sha256sum "$out" | cut -c1-16) ($ref $*)"
    ;;
  rules)
    tier=$2
    shift 2
    pts=() cons=()
    while [ $# -gt 0 ]; do
      [ "$1" = -- ] && { shift; break; }
      pts+=(--point "$1")
      shift
    done
    for c in "$@"; do cons+=(--contrast "$c"); done
    py "$OPS/m15/m15_rules.py" --tier "$tier" --lines-root "$L" --readout "$L/readout/$tier.json" "${pts[@]}" \
      "${cons[@]}" --output "$M/select/$tier-finalists.json" > "$M/select/rules-$tier.log" 2>&1 \
      || die "rules FAILED (see $M/select/rules-$tier.log)"
    log "finalists $tier: $(tail -1 "$M/select/rules-$tier.log" | cut -c1-600)"
    ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
