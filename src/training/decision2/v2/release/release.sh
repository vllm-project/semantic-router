#!/usr/bin/env bash
# One-command private release (or staging dry run) of a Decision 2.0 package. Run on a node.
#
# Usage (SRC is a mirror directory name under /data/dev2/src: <sha> or <sha>-src_training_decision2):
#   release.sh --spec SPEC.json --src SRC --work /data/dev2/runs/release/<id> \
#       [--cpu | --gpu N --track TRACK] [--image IMAGE] [--python PY] [--mount PATH]... \
#       [--mount-rw PATH]... [--env KEY=VALUE]... [--site DIR]... [--require-kernels] \
#       [--threads N] [--base-path DIR] [--parity NAME:PROMPTS:PREDICTIONS:COUNT]... \
#       [--parity-tolerance X] [--shared-lease NAME] [--upload] [--collect [--already-collected]]
# --env takes non-secret runtime settings only (e.g. TRITON_CACHE_DIR of a persisted autotune
# cache mounted with --mount-rw); --site names an image directory of kernel packages that the
# isolated interpreter must import (e.g. /opt/decision-fla); --require-kernels makes the example
# and parity processes fail unless the Qwen3.5 kernels and the persisted cache are in use.
# --shared-lease NAME (a GPU the allocation table marks as shared) writes only
# /data/dev2/leases/gpuN.lock/owner.NAME and never reads or rewrites the owner's entry.
# --already-collected (with --collect): a new revision of a release repository that an earlier
# final decision already added; the pre-collect readback then expects its collection item.
#
# Steps (each writes <work>/receipts/<step>.json; any failure stops the run):
#   build        v2.release.build: exact scored bytes + runtime + card -> <work>/package/<repo-name>
#   pre-a/pre-b  native Choice/Noul/Score examples in two separate container processes
#   repeat-pre   pre-a vs pre-b (bit-identical answers required)
#   card-pre     the README's Python example, executed as written, vs pre-a
#   parity-pre   package answers vs sealed same-panel predictions (optional --parity)
#   --upload:    ensure (private) -> upload -> real `hf download` into <work>/download/<repo-name>
#                -> tree (re-hash) -> post (examples on the download) -> repeat-post (vs pre-a)
#                -> card-post -> parity-post -> readback (private, hashes, Hub card, collection)
#   --collect:   gate seal (the spec's coordinator decision + every verification receipt, bound to the
#                uploaded revision) -> collect (private Decision 2.0 collection) -> readback-collected
# A package that ships the Transformers remote code (v2/release/automap) also runs, before upload,
#   automap-pre (AutoConfig / AutoTokenizer / AutoModel / pipeline with trust_remote_code vs pre-a,
#   bit-identical), automap-card-pre (the card's Transformers block on the package) and, with --parity,
#   automap-parity-pre plus automap-vs-native-pre (per-prompt answers of AutoModel vs parity-pre's native
#   run); after the download automap-post, and automap-hub[-<label>] (the card's Transformers block exactly
#   as written, from the Hub, in a fresh cache over the host network; --hub-site LABEL=DIR adds a run with
#   that site first on the path, e.g. another Transformers version). Answers stay in <work>/answers.
# The container never mounts gold. --cpu exposes no GPU; --gpu takes a leased GPU like the eval runner,
# passing only that GPU's render node and /dev/kfd (ROCR_VISIBLE_DEVICES=0 inside).
set -euo pipefail

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

spec="" sha="" work="" device="cpu" gpu="" track="" image="decision20-train-fast:host2" python_bin="python3"
threads="4" base_path="" upload=0 gate="" parity_tolerance="1e-4" mounts=() parity=()
rw_mounts=() envs=() site_args=() kernel_args=() shared="" readback_args=() hub_sites=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec) spec="$2"; shift 2 ;;
    --src) sha="$2"; shift 2 ;;
    --work) work="$2"; shift 2 ;;
    --cpu) device="cpu"; shift ;;
    --gpu) gpu="$2"; device="cuda:0"; shift 2 ;;
    --track) track="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --mount) mounts+=("$2"); shift 2 ;;
    --mount-rw) rw_mounts+=("$2"); shift 2 ;;
    --env)
      [[ "$2" =~ ^[A-Z][A-Z0-9_]*=.*$ && ! "${2%%=*}" =~ (TOKEN|SECRET|PASSWORD|_KEY$) ]] \
        || { echo "--env takes a non-secret KEY=VALUE setting" >&2; exit 2; }
      envs+=(-e "$2"); shift 2 ;;
    --site) site_args+=(--site "$2"); shift 2 ;;
    --require-kernels) kernel_args=(--require-kernels); shift ;;
    --shared-lease) shared="$2"; shift 2 ;;
    --threads) threads="$2"; shift 2 ;;
    --base-path) base_path="$2"; mounts+=("$2"); shift 2 ;;
    --parity) parity+=("$2"); shift 2 ;;
    --parity-tolerance) parity_tolerance="$2"; shift 2 ;;
    --upload) upload=1; shift ;;
    --collect) gate=1; shift ;;
    --already-collected) readback_args=(--already-collected); shift ;;
    --hub-site)
      [[ "$2" =~ ^[a-z0-9.-]+=/.+$ ]] || { echo "--hub-site takes LABEL=/absolute/dir" >&2; exit 2; }
      hub_sites+=("$2"); shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$spec" && -n "$sha" && -n "$work" ]] || usage
[[ "$work" == /data/dev2/runs/release/* ]] || { echo "work dir must be under /data/dev2/runs/release/" >&2; exit 2; }
[[ -z "$gate" || "$upload" == 1 ]] || { echo "--collect needs --upload" >&2; exit 2; }
[[ ${#readback_args[@]} -eq 0 || -n "$gate" ]] || { echo "--already-collected needs --collect" >&2; exit 2; }
src="/data/dev2/src/$sha"
S="$src/src/training/decision2"
[[ -f "$src/.dev2-mirror.json" ]] || { echo "no verified mirror at $src" >&2; exit 1; }
[[ ! -e "$work" ]] || { echo "$work already exists; use a new work dir" >&2; exit 1; }
hf_python="/data/dev2/tools/hf-cli/bin/python"
repo="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo_id"])' "$spec")"
name="${repo#*/}"
mkdir -p "$work/receipts" "$work/package" "$work/logs" "$work/answers"
cp "$spec" "$work/receipts/spec.json"
spec="$work/receipts/spec.json"
python3 - "$work/receipts/launcher.json" "$image" "$(docker image inspect --format '{{.Id}}' "$image")" \
  "$device" "$gpu" "$track" "${envs[*]:-}" "${rw_mounts[*]:-}" "${site_args[*]:-}" "${kernel_args[*]:-}" <<'EOF'
import json, sys
path, image, image_id, device, gpu, track, envs, rw, sites, kernels = sys.argv[1:]
json.dump({"schema": "dev2-release-launcher/1", "image": image, "image_id": image_id, "device": device,
           "gpu": gpu or None, "track": track or None, "env": [e for e in envs.split() if e != "-e"],
           "rw_mounts": rw.split(), "sites": [s for s in sites.split() if s != "--site"],
           "require_kernels": bool(kernels)}, open(path, "x"), indent=2, sort_keys=True)
EOF
export PYTHONPATH="$S"
log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$work/logs/release.log"; }

gpu_flags=()
if [[ "$device" != "cpu" ]]; then
  [[ "$gpu" =~ ^[0-7]$ && -n "$track" ]] || { echo "--gpu N needs --track" >&2; exit 2; }
  lease="/data/dev2/leases/gpu$gpu.lock"
  lease_file="$lease/owner"
  if [[ -n "$shared" ]]; then
    [[ "$shared" =~ ^[a-z0-9-]+$ ]] || { echo "--shared-lease takes a short lowercase name" >&2; exit 2; }
    lease_file="$lease/owner.$shared"
  elif [[ -f "$lease/owner" ]] && ! grep -qx "track=$track" "$lease/owner"; then
    echo "gpu$gpu is leased by another track" >&2; exit 1
  fi
  # Only this GPU's render node: on shared nodes the other GPUs may belong to someone else.
  bdf="$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 ~ "^"g"$" {getline; print tolower($2)}')"
  render="$(readlink -f "/dev/dri/by-path/pci-${bdf}-render" 2>/dev/null || true)"
  [[ -n "$bdf" && -c "$render" ]] || { echo "no render node for GPU $gpu" >&2; exit 1; }
  mkdir -p "$lease"
  printf 'track=%s\npurpose=%s\nstart_utc=%s\nexpected_end_utc=unknown\nrun_dir=%s\n' \
    "$track" "release verification $repo" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$work" > "$lease_file"
  gpu_flags=(--device /dev/kfd --device "$render" --group-add video --group-add render
             --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0)
else
  gpu_flags=(-e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES=)
fi

# One isolated container process per call: package read-only, no network, no gold, no bytecode.
examples() {
  local volumes=(-v "$src:$src:ro" -v "$work/package:$work/package:ro" -v "$work/receipts:$work/receipts"
                 -v "$work/answers:$work/answers")
  [[ -d "$work/download" ]] && volumes+=(-v "$work/download:$work/download:ro")
  local m
  for m in "${mounts[@]}"; do volumes+=(-v "$m:$m:ro"); done
  for m in "${rw_mounts[@]}"; do volumes+=(-v "$m:$m"); done
  docker run --rm --network none --ipc host "${gpu_flags[@]}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e TOKENIZERS_PARALLELISM=false "${envs[@]}" "${volumes[@]}" --entrypoint "$python_bin" "$image" \
    -I -B "$S/v2/release/examples.py" "$@"
}
device_args=(--threads "$threads" "${site_args[@]}" "${kernel_args[@]}")
[[ "$device" == "cpu" ]] && device_args+=(--device cpu) || device_args+=(--device cuda:0)
# The remote code fetches a base-bound adapter's pinned base itself (here from the mounted HF_HUB_CACHE).
automap_args=("${device_args[@]}")
[[ -n "$base_path" ]] && device_args+=(--base-path "$base_path")
parity_args=()
for p in "${parity[@]}"; do parity_args+=(--panel "$p"); done

# The card's Transformers block exactly as written, from the Hub: fresh HF_HOME, host network, the node's
# token file mounted read-only (never in argv or env); optional extra site (e.g. another Transformers).
hub_smoke() {
  local label="$1" extra="$2" suffix="${1:+-$1}"
  local home="$work/hub$suffix"
  mkdir -p "$home/hf-home"
  local volumes=(-v "$src:$src:ro" -v "$work/package:$work/package:ro" -v "$work/receipts:$work/receipts"
                 -v "$home:$home" -v /root/.cache/huggingface/token:/run/decision2-hf/token:ro)
  local sites=("${site_args[@]}") keep=() e m
  for m in "${rw_mounts[@]}"; do volumes+=(-v "$m:$m"); done
  if [[ -n "$extra" ]]; then
    volumes+=(-v "$extra:$extra:ro")
    sites=(--site "$extra" "${site_args[@]}")
  fi
  for e in "${envs[@]}"; do
    [[ "$e" =~ ^(HF_HOME|HF_HUB_CACHE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)= ]] || keep+=("$e")
  done
  # The image sets HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1; a user's environment is online.
  docker run --rm --network host --ipc host "${gpu_flags[@]}" -e HF_HOME="$home/hf-home" \
    -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
    -e HF_TOKEN_PATH=/run/decision2-hf/token -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
    "${keep[@]}" "${volumes[@]}" --entrypoint "$python_bin" "$image" -I -B "$S/v2/release/examples.py" \
    automap-card --hub --package "$pkg" --reference "$work/receipts/pre-a.json" --expect-revision "$revision" \
    --output "$work/receipts/automap-hub$suffix.json" "${sites[@]}" > "$work/logs/automap-hub$suffix.log" 2>&1 \
    || { rm -rf "$home/hf-home"; return 1; }
  rm -rf "$home/hf-home"
}

started="$(date +%s.%N)"
log "build $repo from $sha"
python3 -m v2.release.build --spec "$spec" --output "$work/package/$name" > "$work/logs/build.log"
cp "$work/package/$name.build/BUILD.json" "$work/receipts/build.json"
pkg="$work/package/$name"
remote_code="$(python3 -c 'import json,sys; print(1 if json.load(open(sys.argv[1])).get("remote_code") else 0)' \
  "$pkg/MODEL_MANIFEST.json")"
answers_args=()
[[ "$remote_code" == 1 ]] && answers_args=(--answers "$work/answers/native-pre.jsonl")

log "pre-upload examples (two processes)"
examples run --package "$pkg" --output "$work/receipts/pre-a.json" "${device_args[@]}" > "$work/logs/pre-a.log" 2>&1
examples run --package "$pkg" --output "$work/receipts/pre-b.json" "${device_args[@]}" > "$work/logs/pre-b.log" 2>&1
python3 "$S/v2/release/examples.py" compare "$work/receipts/pre-a.json" "$work/receipts/pre-b.json" \
  --output "$work/receipts/repeat-pre.json"
log "card example (pre-upload)"
examples card --package "$pkg" --reference "$work/receipts/pre-a.json" --output "$work/receipts/card-pre.json" \
  "${site_args[@]}" > "$work/logs/card-pre.log" 2>&1
if [[ ${#parity_args[@]} -gt 0 ]]; then
  log "scored-panel parity (pre-upload)"
  examples parity --package "$pkg" --output "$work/receipts/parity-pre.json" --tolerance "$parity_tolerance" \
    "${device_args[@]}" "${parity_args[@]}" "${answers_args[@]}" > "$work/logs/parity-pre.log" 2>&1
fi
if [[ "$remote_code" == 1 ]]; then
  log "Transformers remote code (pre-upload)"
  examples automap --package "$pkg" --reference "$work/receipts/pre-a.json" \
    --output "$work/receipts/automap-pre.json" "${automap_args[@]}" > "$work/logs/automap-pre.log" 2>&1
  examples automap-card --package "$pkg" --reference "$work/receipts/pre-a.json" \
    --output "$work/receipts/automap-card-pre.json" "${site_args[@]}" > "$work/logs/automap-card-pre.log" 2>&1
  if [[ ${#parity_args[@]} -gt 0 ]]; then
    log "scored-panel parity through AutoModel (pre-upload)"
    examples automap-parity --package "$pkg" --output "$work/receipts/automap-parity-pre.json" \
      --tolerance "$parity_tolerance" --answers "$work/answers/automap-pre.jsonl" "${automap_args[@]}" \
      "${parity_args[@]}" > "$work/logs/automap-parity-pre.log" 2>&1
    python3 "$S/v2/release/examples.py" compare-answers "$work/answers/native-pre.jsonl" \
      "$work/answers/automap-pre.jsonl" --tolerance "$parity_tolerance" \
      --output "$work/receipts/automap-vs-native-pre.json"
  fi
fi

if [[ "$upload" == 1 ]]; then
  kind="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["kind"])' "$spec")"
  model_name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_name"])' "$spec")"
  log "ensure private $repo"
  "$hf_python" -m v2.release.hub ensure --repo "$repo" --kind "$kind" --model-name "$model_name" \
    --output "$work/receipts/ensure.json"
  log "upload"
  "$hf_python" -m v2.release.hub upload --repo "$repo" --package "$pkg" \
    --message "Decision 2.0 package $model_name ($kind) from $sha" --output "$work/receipts/upload.json"
  revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$work/receipts/upload.json")"
  log "real hf download of $revision"
  mkdir -p "$work/download"
  "$hf_python" -m v2.release.hub download --repo "$repo" --revision "$revision" --dest "$work/download/$name" \
    --output "$work/receipts/download.json"
  python3 -m v2.release.hub tree --package "$pkg" --download "$work/download/$name" --output "$work/receipts/tree.json"
  log "post-download examples"
  examples run --package "$work/download/$name" --output "$work/receipts/post.json" "${device_args[@]}" \
    > "$work/logs/post.log" 2>&1
  python3 "$S/v2/release/examples.py" compare "$work/receipts/pre-a.json" "$work/receipts/post.json" \
    --output "$work/receipts/repeat-post.json"
  examples card --package "$work/download/$name" --reference "$work/receipts/pre-a.json" \
    --output "$work/receipts/card-post.json" "${site_args[@]}" > "$work/logs/card-post.log" 2>&1
  if [[ ${#parity_args[@]} -gt 0 ]]; then
    examples parity --package "$work/download/$name" --output "$work/receipts/parity-post.json" \
      --tolerance "$parity_tolerance" "${device_args[@]}" "${parity_args[@]}" > "$work/logs/parity-post.log" 2>&1
  fi
  if [[ "$remote_code" == 1 ]]; then
    log "Transformers remote code on the download; card example from the Hub (fresh cache)"
    examples automap --package "$work/download/$name" --reference "$work/receipts/pre-a.json" \
      --output "$work/receipts/automap-post.json" "${automap_args[@]}" > "$work/logs/automap-post.log" 2>&1
    hub_smoke "" ""
    for hs in "${hub_sites[@]}"; do hub_smoke "${hs%%=*}" "${hs#*=}"; done
  fi
  log "readback"
  "$hf_python" -m v2.release.hub readback --repo "$repo" --revision "$revision" --package "$pkg" \
    "${readback_args[@]}" --output "$work/receipts/readback.json"
  if [[ -n "$gate" ]]; then
    log "gate seal and collection add"
    python3 -m v2.release.gate seal --work "$work"
    "$hf_python" -m v2.release.hub collect --repo "$repo" --revision "$revision" --package "$pkg" \
      --gate "$work/receipts/gate.json" --output "$work/receipts/collect.json"
    "$hf_python" -m v2.release.hub readback --repo "$repo" --revision "$revision" --package "$pkg" \
      --expect-collected --output "$work/receipts/readback-collected.json"
  fi
fi
ended="$(date +%s.%N)"
python3 -m v2.release.summary --work "$work" --wall-seconds "$(python3 -c "print($ended - $started)")" \
  --device "$device" --image "$image" --source "$sha"
