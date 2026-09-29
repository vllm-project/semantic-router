#!/usr/bin/env bash
# 0.6B M8 staging dry run of m8-s5-b05 on node A: the pre-upload steps of v2/release/release.sh
# (build, bundle verify, two-process examples, repeat, card example, scored-panel parity) plus a
# negative control, into a fresh work dir outside /data/dev2/runs/release. No Hugging Face step:
# nothing is ensured, uploaded, downloaded, read back or collected.
# Usage (node A, detached, GPU leased to track 06b-encoder and idle):
#   nohup setsid bash .../dryrun.sh <sha>-src_training_decision2 <0|1> /data/dev2/runs/06b/m8/release-dryrun \
#     < /dev/null > /data/dev2/logs/06b/m8-release-dryrun.log 2>&1 &
set -euo pipefail

sha="$1"
gpu="$2"
work="$3"
src="/data/dev2/src/$sha"
S="$src/src/training/decision2"
image="decision20-train-fast:host2"
G=/data/dev2/private/panels/goldfree
R=/data/dev2/runs/06b/m8/formal/m8-s5-b05
N=/data/dev2/runs/06b/m6/formal/m6-mxcx-soup
[[ -f "$src/.dev2-mirror.json" ]] || { echo "no verified mirror at $src" >&2; exit 1; }
[[ "$gpu" =~ ^[01]$ ]] || { echo "GPU0 or GPU1 only" >&2; exit 2; }
[[ "$work" == /data/dev2/runs/06b/m8/* ]] || { echo "work dir must be under /data/dev2/runs/06b/m8/" >&2; exit 2; }
[[ ! -e "$work" ]] || { echo "$work already exists; use a new work dir" >&2; exit 1; }
lease="/data/dev2/leases/gpu$gpu.lock/owner"
grep -qx 'track=06b-encoder' "$lease" || { echo "gpu$gpu is not leased to the 0.6B track" >&2; exit 1; }
grep -q '^status=idle' "$lease" || { echo "gpu$gpu owner entry is not idle" >&2; exit 1; }

spec_src="$S/v2/06b/records/m8/release/dev2-0p6b-m8-staging.json"
repo="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo_id"])' "$spec_src")"
name="${repo#*/}"
mkdir -p "$work/receipts" "$work/package" "$work/logs" "$work/inputs"
cp "$spec_src" "$work/receipts/spec.json"
spec="$work/receipts/spec.json"
cp "$lease" "$work/logs/lease-before.txt"
export PYTHONPATH="$S"
log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$work/logs/dryrun.log"; }

status=1
release_lease() {
  printf 'track=06b-encoder\nstatus=idle (0.6B M8 staging dry run finished; no job running)\npurpose=allocation retained by the 0.6B track pending coordinator decision\nlast_job_end_utc=%s\nlast_job_exit=%s\nupdated_utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lease"
}
printf 'track=06b-encoder\nstatus=running\npurpose=0.6B M8 staging dry run of m8-s5-b05 (release examples and parity; no HF)\nstart_utc=%s\nexpected_end_utc=%s\nrun_dir=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(date -u -d '+30 min' +%Y-%m-%dT%H:%M:%SZ)" "$work" > "$lease"
trap release_lease EXIT

gpu_flags=(--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined
           -e ROCR_VISIBLE_DEVICES="$gpu" -e HIP_FORCE_DEV_KERNARG=1)
mounts=("$G" "$R/output" "$R-mlx/output" "$N/output" "$work/inputs")
examples() {
  local step="$1"; shift
  local volumes=(-v "$src:$src:ro" -v "$work/package:$work/package:ro" -v "$work/receipts:$work/receipts")
  local m
  for m in "${mounts[@]}"; do volumes+=(-v "$m:$m:ro"); done
  docker run --rm --name "dev2-06b-m8dry-$step" --network none --ipc host "${gpu_flags[@]}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false "${volumes[@]}" \
    --entrypoint python3 "$image" -I -B "$S/v2/release/examples.py" "$@"
}
device_args=(--threads 4 --device cuda:0)
timing="$work/receipts/gpu-steps.tsv"
timed() {
  local step="$1"; shift
  local t0 t1 rc=0
  t0="$(date +%s.%N)"
  examples "$step" "$@" > "$work/logs/$step.log" 2>&1 || rc=$?
  t1="$(date +%s.%N)"
  printf '%s\t%s\t%s\t%s\n' "$step" "$t0" "$t1" "$rc" >> "$timing"
  return "$rc"
}

log "build $repo from $sha"
python3 -m v2.release.build --spec "$spec" --output "$work/package/$name" > "$work/logs/build.log"
cp "$work/package/$name.build/BUILD.json" "$work/receipts/build.json"
pkg="$work/package/$name"

log "bundle verify (host, stdlib)"
python3 -I -B - "$pkg" "$work/receipts/verify.json" <<'EOF'
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from decision2 import verify_bundle
manifest = verify_bundle(sys.argv[1])
json.dump({"verified": True, "files": len(manifest["files_sha256"]) + 1,
           "manifest_sha256": hashlib.sha256(open(sys.argv[1] + "/MODEL_MANIFEST.json", "rb").read()).hexdigest(),
           "parameters": manifest["parameters"]["packaged"]}, open(sys.argv[2], "x"), indent=2, sort_keys=True)
EOF

log "negative-control inputs (typed FINAL subsets by question type, file order)"
python3 - "$G/typed-final.prompts.jsonl" "$work/inputs" <<'EOF'
import json, sys
lines = [l for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
def kind(line):
    qs = json.loads(line)["questions"].values()
    return {q["type"] + (str(len(q["criteria"])) if q["type"] == "score" else "") for q in qs}
picks = {"choice": 50, "noul": 50, "score5": 100}
for label, count in picks.items():
    rows = [l for l in lines if kind(l) == {label}][:count]
    assert len(rows) == count, label
    with open(f"{sys.argv[2]}/neg-{label}.prompts.jsonl", "x", encoding="utf-8") as out:
        out.writelines(rows)
EOF

log "pre-upload examples (two processes)"
timed pre-a run --package "$pkg" --output "$work/receipts/pre-a.json" "${device_args[@]}"
timed pre-b run --package "$pkg" --output "$work/receipts/pre-b.json" "${device_args[@]}"
python3 "$S/v2/release/examples.py" compare "$work/receipts/pre-a.json" "$work/receipts/pre-b.json" \
  --output "$work/receipts/repeat-pre.json"
log "card example"
timed card-pre card --package "$pkg" --reference "$work/receipts/pre-a.json" --output "$work/receipts/card-pre.json"
log "scored-panel parity at tolerance 0"
timed parity-pre parity --package "$pkg" --output "$work/receipts/parity-pre.json" --tolerance 0 "${device_args[@]}" \
  --panel "typed-final:$G/typed-final.prompts.jsonl:$R/output/typed-final.predictions.jsonl:1600" \
  --panel "public231:$G/public231.prompts.jsonl:$R/output/public231.predictions.jsonl:231" \
  --panel "css15:$G/css15.prompts.jsonl:$R/output/css15.predictions.jsonl:400" \
  --panel "mlx-diag:$G/mlx-diag.prompts.jsonl:$R-mlx/output/mlx-diag.predictions.jsonl:300"
log "negative control vs uncorrected m6-mxcx-soup predictions (expected to differ on Score only)"
neg_rc=0
timed parity-negative parity --package "$pkg" --output "$work/receipts/parity-negative.json" --tolerance 0 \
  "${device_args[@]}" \
  --panel "neg-choice:$work/inputs/neg-choice.prompts.jsonl:$N/output/typed-final.predictions.jsonl:50" \
  --panel "neg-noul:$work/inputs/neg-noul.prompts.jsonl:$N/output/typed-final.predictions.jsonl:50" \
  --panel "neg-score5:$work/inputs/neg-score5.prompts.jsonl:$N/output/typed-final.predictions.jsonl:100" \
  || neg_rc=$?
python3 - "$work/receipts/parity-negative.json" "$neg_rc" <<'EOF'
import json, sys
panels = json.load(open(sys.argv[1]))["panels"]
same = lambda p: p["category_changes"] == 0 and p["missing"] == 0 and p["max_abs_drift"] == 0 and p["input_mismatch"] == 0
ok = sys.argv[2] == "1" and same(panels["neg-choice"]) and same(panels["neg-noul"]) and panels["neg-score5"]["max_abs_drift"] > 0
print(json.dumps({"negative_control_passed": ok}))
sys.exit(0 if ok else 1)
EOF
status=0
log "done"
