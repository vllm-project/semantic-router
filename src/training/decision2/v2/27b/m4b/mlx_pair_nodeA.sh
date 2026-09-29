#!/usr/bin/env bash
# M4b mlx-diag paired intervals vs F1 for successor-rule item 4 (run on the workstation: the nodes do not reach each
# other; CPU only). Node A holds the mlx-diag gold and the formal gold panels, node B the finalists' sealed formal runs
# and gate outputs. Writes under P=/data/dev2/runs/27b/m4b-mlx-pair on node A (and P on node B for the copies back)
# plus run_gates.sh's verdicts-mlx.json. Read in place after a SHA-256 check against node B: node A's copy of F1's
# formal run (M3-A-soup/formal), the relayed mlx-diag collections (score_mlx_nodeA.sh, m3f2/f2-mlx-score.sh) and the
# flagged file (eval m5 overlap-effects final/flagged.json 0ed60dd7, node B's final-27b/flagged.json).
# Usage: mlx_pair_nodeA.sh MIRROR_SHA [SLOT...]
#   MIRROR_SHA  node-A mirror /data/dev2/src/<sha>-src_training_decision2: v2.eval.overlap_effects (byte-identical to
#               1bbebc2fe, run_gates.sh's overlap stage), lux9b.mlx_paired and overlap-spec-27b-m4b.json
#   SLOT        sealed finalist slots (default F-a F-b F-c)
# Stages (STAGES, default verify,relay,spec,run,card,back,verdicts):
#   verify    mirror_to_node.sh --verify of MIRROR_SHA on node A and of VERDICTS_MIRROR (default ca1f24f5b) on node B
#   relay     per SLOT the overlap_effects inputs of node B formal/ (SEAL.json, REPORT.json, PAIRED-vs-M3-A-soup.json,
#             typed FINAL / CSS15 / public 231 predictions) -> node A P/relay/SLOT/formal (tar stream, SHA-256 on both
#             sides, then read-only); F1's node-A copy, the mlx-diag collections and the flagged file must hash as on
#             node B, and each mlx-diag collection must name its formal run's checkpoint path and revision
#   spec      P/spec.json from the committed template: F1 and each SLOT with its mlx_run, the mlx panel, one tier per
#             SLOT with F1 as the only comparator (and threshold pool, which the tool requires: these tier rules are
#             not the M4b gates), the SLOT's stored PAIRED-vs-M3-A-soup.json to reproduce
#   run       v2.eval.overlap_effects run -> P/overlap-effects-mlx.json; its mlx interval is the type macro over
#             Choice, Noul and Score (items within type x language, 5,000 draws, seed 20260927)
#   card      lux9b.mlx_paired SLOT - F1 on the card-eligible Choice + Noul parts (Score is XNLI, CC BY-NC, internal
#             only; same strata, draws and seed) -> P/card/SLOT-vs-F1.json
#   back      P's aggregate outputs (not the relayed predictions) -> node B P, SHA-256 on both sides; the SLOT - F1
#             v3, focus-task, public 231 and contamination entries must equal node B's overlap stage outputs
#   verdicts  node B run_gates.sh STAGES=verdicts with MLX_OVERLAP=P/overlap-effects-mlx.json per SLOT, from
#             VERDICTS_MIRROR -> /data/dev2/runs/27b/m4b/SLOT/verdicts-mlx.json
set -euo pipefail
echo "m4b mlx-diag pair $*: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

MIRROR_SHA=${1:?MIRROR_SHA}
shift
[[ "$MIRROR_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
SLOTS=("$@")
[ ${#SLOTS[@]} -gt 0 ] || SLOTS=(F-a F-b F-c)
for slot in "${SLOTS[@]}"; do
  case "$slot" in F-a | F-b | F-c) ;; *) echo "unknown finalist slot $slot" >&2; exit 2 ;; esac
done
STAGES=${STAGES:-verify,relay,spec,run,card,back,verdicts}
VERDICTS_MIRROR=${VERDICTS_MIRROR:-ca1f24f5b4887c9bc544b64e399ef7f3ad246926}
JOBS=${JOBS:-16}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
nodes=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$nodes" | cut -d= -f2-)
B=$(grep '^node-b=' "$nodes" | cut -d= -f2-)
R=/data/dev2/runs/27b
P=$R/m4b-mlx-pair
M=/data/dev2/src/$MIRROR_SHA-src_training_decision2/src/training/decision2
GATES=/data/dev2/src/$VERDICTS_MIRROR-src_training_decision2/src/training/decision2/v2/27b/m4b/run_gates.sh
F1_RUN=$R/M3-A-soup/formal
F1_MLX=$R/m3-f2/mlx-diag/M3-A-soup
PANEL=/data/dev2/private/panels/mlx-diag-v1
FLAGGED=/data/dev2/runs/eval/m5/overlap-effects/final/flagged.json
FLAGGED_B=/data/dev2/runs/eval/m5/overlap-effects/final-27b/flagged.json
FLAGGED_SHA=0ed60dd751f2f33326b1b58859e27065166ea18e529812caa6ccec9fa894f41e
RUN_FILES="SEAL.json REPORT.json output/typed-final.predictions.jsonl output/css15.predictions.jsonl output/public231.predictions.jsonl"
ENV="PYTHONPATH=$M:$M/v2/9b PYTHONDONTWRITEBYTECODE=1 TMPDIR=$P/tmp"
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
same() {  # WHAT COMMAND: the command's output is non-empty and identical on both nodes
  local what=$1 a b
  b=$(on_b "$2") a=$(on_a "$2")
  [ -n "$b" ] && [ "$a" = "$b" ] || { echo "$what differs between node A and node B" >&2; exit 1; }
}

if has verify; then
  (cd "$HERE" && bash ../../common/mirror_to_node.sh --verify --path src/training/decision2 node-a "$MIRROR_SHA")
  (cd "$HERE" && bash ../../common/mirror_to_node.sh --verify --path src/training/decision2 node-b "$VERDICTS_MIRROR")
fi
if has relay; then
  on_a "test ! -e '$P/relay'" || { echo "$P/relay exists on node A" >&2; exit 66; }
  [ "$(on_a "sha256sum < '$FLAGGED'" | cut -c1-64)" = "$FLAGGED_SHA" ] || { echo "node A $FLAGGED changed" >&2; exit 1; }
  [ "$(on_b "sha256sum < '$FLAGGED_B'" | cut -c1-64)" = "$FLAGGED_SHA" ] || { echo "node B $FLAGGED_B changed" >&2; exit 1; }
  same "F1's formal run" "cd '$F1_RUN' && sha256sum $RUN_FILES"
  for mlx in "$F1_MLX" "${SLOTS[@]/#/$R/m4b/mlx-diag/}"; do
    same "$mlx" "cd '$mlx' && sha256sum COLLECT.json output/mlx-diag.predictions.jsonl"
  done
  on_b "python3 - '$R' ${SLOTS[*]}" <<'EOF'
import json, sys
r, slots = sys.argv[1], sys.argv[2:]
runs = [("M3-A-soup/formal", "m3-f2/mlx-diag/M3-A-soup")]
runs += [(f"m4b/{s}/formal", f"m4b/mlx-diag/{s}") for s in slots]
for formal, mlx in runs:
    a, b = (json.load(open(f"{r}/{d}/COLLECT.json")) for d in (formal, mlx))
    for key in ("model_path", "model_revision"):
        if a[key] != b[key]:
            raise SystemExit(f"{mlx}: {key} differs from {formal}")
    print(f"{mlx}: the checkpoint of {formal} ({b['model_revision']})")
EOF
  for slot in "${SLOTS[@]}"; do
    src=$R/m4b/$slot/formal dst=$P/relay/$slot/formal files="$RUN_FILES PAIRED-vs-M3-A-soup.json"
    hb=$(on_b "cd '$src' && sha256sum $files")
    on_a "mkdir -p '$dst'"
    on_b "tar -C '$src' -cf - $files" | on_a "tar -C '$dst' -xf -"
    [ "$(on_a "cd '$dst' && sha256sum $files")" = "$hb" ] || { echo "$slot: relayed files differ" >&2; exit 1; }
    on_a "cd '$dst' && sha256sum $files > ../SHA256SUMS.txt && chmod -R a-w ."
    echo "$slot: node B $src -> node A $dst ($(wc -l <<< "$hb") files, SHA-256 equal)"
  done
fi
if has spec; then
  on_a "test ! -e '$P/spec.json'" || { echo "$P/spec.json exists" >&2; exit 66; }
  on_a "python3 - '$M/v2/27b/m4b/overlap-spec-27b-m4b.json' '$P' '$F1_RUN' '$F1_MLX' '$R/m4b/mlx-diag' '$PANEL' \
    ${SLOTS[*]}" <<'EOF'
import json, sys
from pathlib import Path
template_path, p, f1_run, f1_mlx, mlx_root, panel, *slots = sys.argv[1:]
template = json.load(open(template_path))
f1 = "DEV2.0-27B (F1)"
if template["comparators"][f1]["run"] != f1_run:
    raise SystemExit(f"the template's F1 run is not {f1_run}")
stored = template["reproduce_names"][f1]
models = {f1: {"run": f1_run, "mlx_run": f1_mlx,
               "note": template["comparators"][f1]["note"] + "; node A copy, SHA-256 equal to node B's"}}
tiers, reproduce = {}, []
for slot in slots:
    cfg = template["m4b"][slot]
    run = f"{p}/relay/{slot}/formal"
    if not (Path(run, "SEAL.json").is_file() and Path(run, stored).is_file()):
        raise SystemExit(f"{slot}: run the relay stage first")
    models[cfg["label"]] = {"run": run, "mlx_run": f"{mlx_root}/{slot}",
                            "note": f"{cfg['note']}; overlap_effects inputs of node B {cfg['run']}, SHA-256 checked"}
    tiers[slot] = {"candidate": cfg["label"], "own_1_0": [], "peers": [], "internal_peers": [f1],
                   "threshold_pool": [f1],
                   "note": "F1 is the only comparator and, because the tool needs one, the threshold pool: "
                           "these tier rules and ranks are not the M4b gates"}
    reproduce.append({"left": cfg["label"], "right": f1, "stored": f"{run}/{stored}"})
spec = {**{k: template[k] for k in ("label", "panel_root", "focus_task")}, "mlx_panel": panel,
        "note": "M4b mlx-diag pairs vs F1 on node A (mlx_pair_nodeA.sh), successor-rule item 4",
        "models": models, "tiers": tiers, "reproduce": reproduce}
with open(f"{p}/spec.json", "x", encoding="utf-8") as out:
    out.write(json.dumps(spec, indent=1) + "\n")
print(json.dumps({"models": list(models), "tiers": {s: t["candidate"] for s, t in tiers.items()}}))
EOF
fi
if has run; then
  on_a "test ! -e '$P/overlap-effects-mlx.json'" || { echo "$P/overlap-effects-mlx.json exists" >&2; exit 66; }
  status=0
  on_a "mkdir -p '$P/tmp' && cd '$M' && date -u +%FT%TZ > '$P/run.start' && $ENV python3 -m v2.eval.overlap_effects \
    run --spec '$P/spec.json' --flagged '$FLAGGED' --output '$P/overlap-effects-mlx.json' --jobs $JOBS \
    > '$P/run.log' 2>&1; s=\$?; echo \$s > '$P/run.exit'; date -u +%FT%TZ > '$P/run.end'; exit \$s" || status=$?
  # a failed run's log can name an item id: it stays on node A
  [ "$status" = 0 ] || { echo "overlap_effects run failed ($status): node A $P/run.log" >&2; exit 1; }
  on_a "tail -n 1 '$P/run.log'"
  on_a "python3 - '$P/overlap-effects-mlx.json'" <<'EOF'
import json, sys
out = json.load(open(sys.argv[1]))
print("problems:", out["problems"] or "none", "| stored paired files reproduced:",
      sum(r["match"] for r in out["reproduction"]), "of", len(out["reproduction"]))
for name, m in out["models"].items():
    print(f"{name}: mlx-diag type macro {m['full']['mlx']['type_macro_accuracy']:.4f}"
          f" (without the flagged items {m['reduced']['mlx']['type_macro_accuracy']:.4f})")
for key, e in out["pairs"].items():
    for v in ("full", "reduced"):
        x = e["mlx"][v]
        print(f"{key} mlx-diag [{v}]: {x['delta']:+.4f} [{x['ci95']['low']:+.4f}, {x['ci95']['high']:+.4f}]"
              f" {e['ci_status']['mlx'][v]}")
EOF
fi
if has card; then
  on_a "test ! -e '$P/card'" || { echo "$P/card exists" >&2; exit 66; }
  on_a "mkdir -p '$P/card' '$P/tmp' && cd '$M' && $ENV python3 - '$P' ${SLOTS[*]}" <<'EOF'
import json, sys
from lux9b import mlx_paired
p, slots = sys.argv[1], sys.argv[2:]
spec = json.load(open(f"{p}/spec.json"))
f1 = "DEV2.0-27B (F1)"
for slot in slots:
    label = spec["tiers"][slot]["candidate"]
    out = f"{p}/card/{slot}-vs-F1.json"
    mlx_paired.main(["--left", spec["models"][label]["mlx_run"], "--right", spec["models"][f1]["mlx_run"],
                     "--panel", spec["mlx_panel"], "--left-name", label, "--right-name", f1, "--output", out])
    r = json.load(open(out))
    parts = [("Choice + Noul", r["overall"])] + [(k, r["by_type"][k]) for k in ("choice", "noul")]
    print(f"{label} - {f1}: " + "; ".join(
        f"{n} {x['delta']:+.4f} [{x['ci95']['low']:+.4f}, {x['ci95']['high']:+.4f}]" for n, x in parts))
EOF
fi
if has back; then
  on_b "test ! -e '$P'" || { echo "$P exists on node B" >&2; exit 66; }
  files="spec.json overlap-effects-mlx.json overlap-effects-mlx.md run.start run.end run.exit run.log"
  for slot in "${SLOTS[@]}"; do files+=" card/$slot-vs-F1.json"; done
  ha=$(on_a "cd '$P' && sha256sum $files")
  on_b "mkdir -p '$P'"
  on_a "tar -C '$P' -cf - $files" | on_b "tar -C '$P' -xf -"
  [ "$(on_b "cd '$P' && sha256sum $files")" = "$ha" ] || { echo "the copies on node B differ" >&2; exit 1; }
  on_a "cd '$P' && sha256sum $files > SHA256SUMS.txt"
  on_b "cd '$P' && sha256sum $files > SHA256SUMS.txt"
  echo "copied $(wc -l <<< "$ha") files to node B $P, SHA-256 equal"
  on_b "python3 - '$P/overlap-effects-mlx.json' '$R/m4b'" <<'EOF'
import json, sys
mine, root = json.load(open(sys.argv[1])), sys.argv[2]
f1 = "DEV2.0-27B (F1)"
for slot, tier in mine["tiers"].items():
    label = tier["models"][0]
    key = f"{label} - {f1}"
    theirs = json.load(open(f"{root}/{slot}/overlap/overlap-effects.json"))
    a, b = mine["pairs"][key], theirs["pairs"][key]
    checks = {
        "v3": all(a["v3"][v] == b["v3"][v] for v in ("full", "full_seed2", "reduced")),
        **{part: a[part] == b[part] for part in ("focus_task", "public231", "contamination")},
        "models": all(mine["models"][n][v][k] == theirs["models"][n][v][k]
                      for n in (label, f1) for v in ("full", "reduced")
                      for k in ("T", "H", "v3", "tasks", "public231")),
    }
    if not all(checks.values()):
        raise SystemExit(f"{slot}: differs from node B's overlap stage: {checks}")
    print(f"{slot}: {key} v3, focus task, public 231, contamination and both models' values"
          " equal node B's overlap stage")
EOF
fi
if has verdicts; then
  on_b "test -f '$P/overlap-effects-mlx.json'" || { echo "run the back stage first" >&2; exit 2; }
  for slot in "${SLOTS[@]}"; do
    on_b "MLX_OVERLAP='$P/overlap-effects-mlx.json' STAGES=verdicts bash '$GATES' '$slot'"
  done
  on_b "python3 - '$R/m4b' ${SLOTS[*]}" <<'EOF'
import json, sys
root, slots = sys.argv[1], sys.argv[2:]
for slot in slots:
    new, old = (json.load(open(f"{root}/{slot}/{name}")) for name in ("verdicts-mlx.json", "verdicts.json"))
    items, before = new["successor_rule"]["items"], old["successor_rule"]["items"]
    item4 = items["4_mlx_diag_not_significantly_below_F1"]
    print(f"{slot}: item 4 {item4['status']} {item4['value']}; successor rule {new['successor_rule']['verdict']}"
          f" (verdicts.json {old['successor_rule']['verdict']}); items changed:"
          f" {sorted(k for k in items if items[k] != before.get(k))}")
EOF
fi
echo "m4b mlx-diag pair stages $STAGES complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
