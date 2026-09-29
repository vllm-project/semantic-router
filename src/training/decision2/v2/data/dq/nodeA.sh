#!/usr/bin/env bash
# Node A: PN1 / HS1 data-quality closure (prereg v2/data/records/m4-dq-prereg-2026-09-30.md).
#
#   nodeA.sh <sha>-src_training_decision2 <step>...
#   steps: hashes isolation shorttext sample embed-manifest embed
#
# CPU steps run in the pinned image with --network none. The embedding scan runs on GPU1 under
# the shared lease entry owner.data-dq. Private outputs (keys, packets, answers, private
# receipts) are created O_EXCL under /data/dev2/private/data/m4-dq (mode 700); public receipts
# under /data/dev2/runs/data/m4-dq. Step embed-manifest expects the node-B house embedding
# manifest at $P/embed/pi-v4-embed-nodeB.json (copied over the local pipe, hash-checked).
set -euo pipefail
umask 077
[[ $# -ge 2 ]] || { echo "usage: nodeA.sh <sha>-src_training_decision2 <step>..." >&2; exit 2; }
MIRROR=/data/dev2/src/$1; shift
S=$MIRROR/src/training/decision2
[[ -f $MIRROR/.dev2-mirror.json && -d $S ]] || { echo "no mirror at $MIRROR" >&2; exit 1; }
IMAGE=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
P=/data/dev2/private/data/m4-dq
R=/data/dev2/runs/data/m4-dq
PN1=/data/dev2/private/data/m4-pn1/e7bedd642175
HS1=/data/dev2/private/data/hs1/21bdb5e90e2b/build
H=/data/dev2/runs/data/m3b/gap/c2/final
SHO=/data/dev2/private/sealed/m3b
GF=/data/dev2/private/panels/goldfree
A7=/data/dev2/runs/data/m3b/hf/v2/a7/arms
MODEL=/data/dev2/hf-cache/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
LEASE=/data/dev2/leases/gpu1.lock/owner.data-dq
GPU=1
mkdir -p "$P/review" "$P/embed" "$R"
chmod 700 "$P" "$R"
RUN=(docker run --rm --network none -v /data:/data:ro -v "$P:$P" -v "$R:$R"
  -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 -w "$S" --entrypoint python3)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/OPERATIONS.log"; }

PINS="f6f6a5315a0aebc9d83b656ff90d378a0e1f2083f95bbcafdc18399969a64616  $PN1/pn1.train.jsonl
c3b68ac113fc1f2afb5554530112f472a9c62f7997285bcaf89a6b655bb7f4cf  $PN1/pn1.dev.jsonl
c90ef3164d90d3fd6a1ab0397529faec172760b1756de7d4d8cbf2677a131a71  $HS1/hs1.train.jsonl
7c4133b05664de492bb80b40df5eef0f028257f10327a492564bea5f62813a32  $H/h7.train.jsonl
0eaf41c0bfdbcd0c2a262a93804a16d8ba2698a8d621e17e83e554f50748a2b1  $H/h7.aho.jsonl
3906f68abd4a945443986d58640c25e36d71c7fcbc50d3621a66e3fde6785122  $SHO/h7.sho.jsonl
1f19e5ab84e6b80182e99cd8e2b7efc79ec17275048e6daf7987afc335e7d361  $H/h8.train.jsonl
3c1ba01ab3bb9244201f741c428f0e75324a422d5744783016f002154cacd627  $H/h8.aho.jsonl
c1983981284671e8dd24dde43e4fba2ac2086457dc56a2a266f6c7e6768c2efa  $SHO/h8.sho.jsonl"

for step in "$@"; do
  log "step $step start (code $(basename "$MIRROR"))"
  case "$step" in
    hashes)
      sha256sum -c --quiet <<< "$PINS"
      ;;
    isolation)
      "${RUN[@]}" "$IMAGE" -m v2.data.freeze isolation \
        --partition "pn1/train=$PN1/pn1.train.jsonl" --partition "pn1dev/dev=$PN1/pn1.dev.jsonl" \
        --partition "existing/h7-train=$H/h7.train.jsonl" --partition "existing/h7-aho=$H/h7.aho.jsonl" \
        --partition "existing/h7-sho=$SHO/h7.sho.jsonl" --partition "existing/h8-train=$H/h8.train.jsonl" \
        --partition "existing/h8-aho=$H/h8.aho.jsonl" --partition "existing/h8-sho=$SHO/h8.sho.jsonl" \
        --report "$R/isolation-h7h8.json" | tail -c 400
      ;;
    shorttext)
      "${RUN[@]}" "$IMAGE" -m v2.data.m4.pn1_scan \
        --rows "$PN1/pn1.train.jsonl" --rows "$PN1/pn1.dev.jsonl" \
        --role "h7_train=$H/h7.train.jsonl" --role "h7_aho=$H/h7.aho.jsonl" --role "h7_sho=$SHO/h7.sho.jsonl" \
        --role "h8_train=$H/h8.train.jsonl" --role "h8_aho=$H/h8.aho.jsonl" --role "h8_sho=$SHO/h8.sho.jsonl" \
        --receipt "$R/shorttext-h7h8.json" | tail -c 600
      ;;
    sample)
      "${RUN[@]}" "$IMAGE" -m v2.data.dq.blind_review sample-pn1 --rows "$PN1/pn1.train.jsonl" \
        --sha256 f6f6a5315a0aebc9d83b656ff90d378a0e1f2083f95bbcafdc18399969a64616 --out-dir "$P/review"
      "${RUN[@]}" "$IMAGE" -m v2.data.dq.blind_review sample-hs1 --rows "$HS1/hs1.train.jsonl" \
        --sha256 c90ef3164d90d3fd6a1ab0397529faec172760b1756de7d4d8cbf2677a131a71 --out-dir "$P/review"
      ;;
    embed-manifest)
      maps=(--map "/data/dev2/private/data/pi-v3-extra/mlx-diag.prompts.jsonl=$GF/mlx-diag.prompts.jsonl")
      for x in A7g A7h A7i A7m A7o A7p; do
        maps+=(--map "/data/dev2/private/a7/runs/a7-dec10-v3/7cef77052a94/final/$x.aho.jsonl=$A7/$x/aho.jsonl")
      done
      for x in A7k A7q A7s A7x; do
        maps+=(--map "/data/dev2/private/a7/runs/a7-enc10-v4/a8e01e6ca917/final/$x.aho.jsonl=$A7/$x/aho.jsonl")
      done
      maps+=(--map "/data/dev2/private/a7/runs/a7-rec10-v2/83ab31ba3e7d/final/A7r.aho.jsonl=$A7/A7r/aho.jsonl")
      "${RUN[@]}" "$IMAGE" -m v2.data.dq.embed_manifest --source "$P/embed/pi-v4-embed-nodeB.json" \
        --source-sha256 deb4b7e9fd9ef5dd2856e4d2a3468828eabcaf96c98c80657ef5800abc0b7d1c "${maps[@]}" \
        --extend "ext_ht_dev=$GF/ht-dev.prompts.jsonl" --extend "ext_score5_dev=$GF/score5-dev.prompts.jsonl" \
        --extend "ext_score5t_dev=$GF/score5t-dev.prompts.jsonl" --extend "ext_hs1_dev=$GF/hs1-dev.prompts.jsonl" \
        --out "$P/embed/pi-v4-embed-nodeA-ext.json" --receipt "$R/embed-manifest.json"
      ;;
    embed)
      [[ ! -e $R/embed.public.json ]] || { echo "embed receipt exists" >&2; exit 1; }
      purpose="PN1 house embedding scan (m4-dq prereg; research & data), recorded co-tenant (shared lease data-dq)"
      start=$(date -u +%FT%TZ); t0=$(date +%s)
      printf 'track=data\npurpose=%s\nstart_utc=%s\nexpected_end_utc=%s\nstatus=busy\n' "$purpose" "$start" \
        "$(date -u -d "@$((t0 + 1800))" +%FT%TZ)" > "$LEASE"
      rc=0
      docker run --rm --name dev2-data-dq-embed --network none --device /dev/kfd --device /dev/dri \
        --group-add video --ipc host --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$GPU \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -v /data:/data:ro -v "$P:$P" -v "$R:$R" \
        -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 -w "$S" --entrypoint python3 "$IMAGE" \
        -m v2.data.embed_scan --candidates "$PN1/pn1.train.jsonl" --candidates "$PN1/pn1.dev.jsonl" \
        --protected-inventory "$P/embed/pi-v4-embed-nodeA-ext.json" --model-path "$MODEL" --batch 256 \
        --private-receipt "$P/embed/embed.private.json" --public-receipt "$R/embed.public.json" \
        > "$R/embed.stdout" 2> "$R/embed.stderr" || rc=$?
      t1=$(date +%s); wall=$((t1 - t0))
      gpuh=$(python3 -c "print(f'{$wall / 3600:.4f}')")
      printf 'track=data\npurpose=%s\nstart_utc=%s\nlast_job_end_utc=%s\nlast_job_exit=%s\ngpu_hours_node_a=%s\nstatus=released\n' \
        "$purpose" "$start" "$(date -u +%FT%TZ)" "$rc" "$gpuh" > "$LEASE"
      printf '{"step":"embed","rc":%s,"wall_s":%s,"gpu_hours":%s,"gpu":"node A GPU%s","image":"%s","code":"%s"}\n' \
        "$rc" "$wall" "$gpuh" "$GPU" "$IMAGE" "$(basename "$MIRROR")" > "$R/embed-run.json"
      [[ $rc -eq 0 ]] || { log "embed failed rc=$rc"; exit 1; }
      ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
  log "step $step done"
done
