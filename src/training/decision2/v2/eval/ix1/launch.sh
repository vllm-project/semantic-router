#!/usr/bin/env bash
# IX1 node-side launcher: parity gate or sharded full run of one released DEV2.0 package.
#
# Usage:
#   launch.sh parity --src DIR --model NAME --gpu N --run DIR --rows FILE
#   launch.sh run    --src DIR --model NAME --gpus "N ..." --run DIR --rows-dir DIR --cache DIR [--only "K ..."]
#   launch.sh resume --src DIR --model NAME --gpus "N ..." --run DIR --rows-dir DIR --only "K ..."
#   launch.sh ref    --src DIR --model NAME --gpu N --run DIR --rows FILE --cache DIR
#   launch.sh extra  --src DIR --model NAME --gpu N --run DIR --rows FILE --cache DIR --tag K
#
# ref: the package's own entry point (v2.eval.ix1.native_ref) over FILE into <run>/ref.jsonl, e.g.
#   the calibration partition. extra: the kit runner with the adapter over FILE into <run>/extra-K,
#   for requests rerun alone. Both run in the foreground with a copy of the frozen cache.
# --only starts just the listed shard indices (shard k still runs on the k-th listed GPU).
# resume restarts ended shards in place: the kit runner skips their final rows and retries errors;
# the previous start/end/exit markers are kept with a numeric suffix (GPU-hours sum every interval).
# A shard directory holding rows.override.jsonl.gz (its rows minus requests that abort the device,
# listed in skipped.json) resumes over that file; skipped requests are rerun alone into extra-<k>/.
# <src> is a mirror_to_node.sh --path src/training/decision2 directory; NAME a package of the table
# below, downloaded at its pinned revision to /data/dev2/models/ix1/<NAME>-<rev8>, or a DIAGNOSTIC
# name: a private restaged package (v2.eval.ix1.restage) answering as the listed repository.
# parity: on one GPU, (1) the package's own entry point (v2.eval.ix1.native_ref) over the gold-free
#   compatibility rows with a fresh Triton autotune cache, which is then frozen to <run>/cache-frozen;
#   (2) the kit runner with the adapter over the same rows with a copy of it; (3) v2.eval.ix1.parity.
#   Runs in the foreground.
# run: one detached kit-runner container per listed GPU (shard k of n on the k-th GPU), each with a
#   copy of the frozen cache (digest-checked); returns once the containers are started.
# Every container sees only its own GPU (its render node, no other /dev/dri entry), has no network,
# a private IPC namespace and a ${IX1_MEMORY:-256g} memory limit, mounts the source, kit, package,
# base and rows read-only and only its run directory writable. Shards start one at a time: the next
# waits until at most one shard is still loading and the host has >= ${IX1_MIN_FREE_GIB:-400} GiB
# available (these are shared Kubernetes control-plane hosts). The image's kernel path
# (/opt/decision-fla) stays on PYTHONPATH; a container that cannot import the flash-linear-attention
# and causal-conv1d kernels exits 97, and a parity log showing the reference-kernel fallback fails.
# Leases: /data/dev2/leases/gpu<N>.lock/owner is written only when absent or already track=eval-ix1;
# a GPU leased by anyone else (node C GPU0: a K8s pod) is refused, and so is a busy GPU.
set -euo pipefail

IMAGE="decision20-train-fast:host2"
IMAGE_ID_PREFIX="sha256:f83b1d10"
IMAGE_PYTHONPATH="/opt/decision-fla"
FALLBACK="falling back to its reference PyTorch implementation"
KIT="/data/dev2/private/eval/index021/kit-87d4650b"
KIT_REVISION="87d4650b42b377c0291a89c1f1a879f9b31082bf"
MODELS="/data/dev2/models/ix1"
HF_CACHE="/data/dev2/hf-cache"
ENGINE="publication.decision_index_release_engine:ReleasedDecisionIndexEngine"
declare -A REVISION=(
  [DEV2.0-0.6B]=476fe984a2316519f3e583b7f31b1670295b2477
  [DEV2.0-0.8B]=bede7938a8c209c09f27400b79eed57948d6b75e
  [DEV2.0-2B]=a53cf66a0d9d492a84b6617b61e7ce35fcd03af0
  [DEV2.0-4B]=fadbba4ff671b4948fb7530fa6748f522b1ac9e4
  [DEV2.0-9B]=e51f9881b92f646cb0bd62b2876d4878cc8d16ec
  [DEV2.0-27B]=5323310327e52d4eadd119cd10accac9b106c97d
)
declare -A DIAGNOSTIC=(  # name -> "repository revision package-dir"
  [M6-IB]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M6-IB-re876fbe"
  [M6-IBX]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M6-IBX-re876fbe"
  [M6-IB2]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M6-IB2-re876fbe"
  [M6-IB2PN]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M6-IB2PN-re876fbe"
  [M6-IBxIB2-m50]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M6-IBxIB2-m50-re876fbe"
  [M6-IBxIB2-m67]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M6-IBxIB2-m67-re876fbe"
  [M7-IB124ML]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M7-IB124ML-re876fbe"
  [M7-IB14ML]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M7-IB14ML-re876fbe"
  [M8-IB14]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M8-IB14-re876fbe"
  [M8-IB124]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/M8-IB124-re876fbe"
  [X7-IBxIB2xIB14ML]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X7-IBxIB2xIB14ML-re876fbe"
  [X7-4ARM]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X7-4ARM-re876fbe"
  [X8-IBxIB2-8]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X8-IBxIB2-8-re876fbe"
  [X8-ML]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X8-ML-re876fbe"
  [X9-LRH2]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X9-LRH2-re876fbe"
  [X9-LRH2xM50]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X9-LRH2xM50-re876fbe"
  [X9-LRH]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X9-LRH-re876fbe"
  [X9-LRHxM50]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X9-LRHxM50-re876fbe"
  [X9-ML0]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X9-ML0-re876fbe"
  [X9-IBxIB2-10]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/m6/X9-IBxIB2-10-re876fbe"
  [M5-L128]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/fix2/M5-L128-95d61175-re876fbe"
  [DEV2.0-27B-budget]="DEV2.0-27B 4e89288d6146034743a14e3fbb98b5864e693c52 /data/dev2/models/ix1/fix2/DEV2.0-27B-4e89288d-re876fbe"
  [DEV2.0-4B-LH]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/DEV2.0-4B-13d42143"
  [DEV2.0-0.8B-08bRA]="DEV2.0-0.8B bede7938a8c209c09f27400b79eed57948d6b75e /data/dev2/models/ix1/dec-08bfast/DEV2.0-0.8B-08bRA-83926edf-rbede7938"
  [K-a13IB]="DEV2.0-9B e51f9881b92f646cb0bd62b2876d4878cc8d16ec /data/dev2/models/ix1/9b/K-a13IB-e51f9881"
  [K-a13-fp32]="DEV2.0-9B e51f9881b92f646cb0bd62b2876d4878cc8d16ec /data/dev2/models/ix1/9b/K-a13-fp32-e51f9881"
  [K-a13IB-bf16]="DEV2.0-9B e51f9881b92f646cb0bd62b2876d4878cc8d16ec /data/dev2/models/ix1/9b/K-a13IB-bf16-e51f9881"
  [DEV2.0-4B-LHA10SD]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m15/DEV2.0-4B-LHA10SD-255021e0-r13d42143"
  [DEV2.0-4B-LHS10SD-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS10SD-bf16-r13d42143"
  [DEV2.0-4B-LHS17SD-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17SD-bf16-r13d42143"
  [DEV2.0-4B-LHS17UP-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17UP-bf16-r13d42143"
  [DEV2.0-4B-LHS23SD-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS23SD-bf16-r13d42143"
  [DEV2.0-4B-LHS17IB4-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17IB4-bf16-r13d42143"
  [DEV2.0-4B-LHS17IB4X-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17IB4X-bf16-r13d42143"
  [DEV2.0-4B-LHS17UP-m50-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17UP-m50-bf16-r13d42143"
  [DEV2.0-4B-LHS23SD-m50-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS23SD-m50-bf16-r13d42143"
  [DEV2.0-4B-LHS17IB4-m50-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17IB4-m50-bf16-r13d42143"
  [DEV2.0-4B-LHS17IB4X-m50-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17IB4X-m50-bf16-r13d42143"
  [DEV2.0-4B-SDMLIB4-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLIB4-bf16-r13d42143"
  [DEV2.0-4B-LHS17ML-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17ML-bf16-r13d42143"
  [DEV2.0-4B-SDMLIB4-m50-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLIB4-m50-bf16-r13d42143"
  [DEV2.0-4B-LHS17ML-m50-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17ML-m50-bf16-r13d42143"
  [DEV2.0-4B-LHS17IB4-x3-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17IB4-x3-bf16-r13d42143"
  [DEV2.0-4B-SDMLIB4-x3-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLIB4-x3-bf16-r13d42143"
  [DEV2.0-4B-SDMLxS17xIB4-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLxS17xIB4-bf16-r13d42143"
  [DEV2.0-4B-SDMLxALL-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLxALL-bf16-r13d42143"
  [DEV2.0-4B-SDMLxALL9-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLxALL9-bf16-r13d42143"
  [DEV2.0-4B-SDMLxALL15-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLxALL15-bf16-r13d42143"
  [DEV2.0-4B-SDMLxALL-hub]="vllm-sr/Decision-2.0-Nox-4B d55528d1635fc474061ec59e31a7c722d3e7ab95 /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLxALL-hub"
  [DEV2.0-4B-LHS17IB4-lrh-hub]="vllm-sr/Decision-2.0-Nox-4B c60d3b5ca7d71a36669392bc4fb94a1596df6739 /data/dev2/models/ix1/dec-m17/DEV2.0-4B-LHS17IB4-lrh-hub"
  [HUB-Kai-0.6B-881bee41]="vllm-sr/Decision-2.0-Kai-0.6B 881bee413681d80ebeac86afcda8b4138dae516e /data/dev2/models/ix1/hub/Decision-2.0-Kai-0.6B-881bee41"
  [HUB-Eos-0.8B-ad0aa724]="vllm-sr/Decision-2.0-Eos-0.8B ad0aa724c924f7c4194be94b1b8441caf2d61c01 /data/dev2/models/ix1/hub/Decision-2.0-Eos-0.8B-ad0aa724"
  [HUB-Sol-2B-4b75b521]="vllm-sr/Decision-2.0-Sol-2B 4b75b52114583b4519001492e8dfb0926c89cfe1 /data/dev2/models/ix1/hub/Decision-2.0-Sol-2B-4b75b521"
  [HUB-Nox-4B-ce1bdc9d]="vllm-sr/Decision-2.0-Nox-4B ce1bdc9d91333aae2bf496ec48c66e1a913eb0a0 /data/dev2/models/ix1/hub/Decision-2.0-Nox-4B-ce1bdc9d"
  [HUB-Lux-9B-214ffa43]="vllm-sr/Decision-2.0-Lux-9B 214ffa4322bc1bce3215c1bd5de6168402c76969 /data/dev2/models/ix1/hub/Decision-2.0-Lux-9B-214ffa43"
  [HUB-Vega-27B-9b067a95]="vllm-sr/Decision-2.0-Vega-27B 9b067a95560284dac8c98ef4130fd5a2c5a92ff9 /data/dev2/models/ix1/hub/Decision-2.0-Vega-27B-9b067a95"
  [DEV2.0-4B-SDMLxS17-m50-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLxS17-m50-bf16-r13d42143"
  [DEV2.0-4B-LHA10UP-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-4bif/DEV2.0-4B-LHA10UP-bf16-r13d42143"
  [DEV2.0-4B-LHA10SD-a75-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-4bif/DEV2.0-4B-LHA10SD-a75-bf16-r13d42143"
  [DEV2.0-4B-LHA10SD-a50-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-4bif/DEV2.0-4B-LHA10SD-a50-bf16-r13d42143"
  [DEV2.0-4B-LHA10SD-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-4bif/DEV2.0-4B-LHA10SD-bf16-r13d42143"
  [DEV2.0-4B-LHA10SDML-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/dec-4bif/DEV2.0-4B-LHA10SDML-bf16-r13d42143"
  [IS-08b-RASD]="DEV2.0-0.8B bede7938a8c209c09f27400b79eed57948d6b75e /data/dev2/models/ix1/index-sweep/08b-RASD-02c17086-rbede7938"
  [IS-08b-RAUP]="DEV2.0-0.8B bede7938a8c209c09f27400b79eed57948d6b75e /data/dev2/models/ix1/index-sweep/08b-RAUP-0bf0401d-rbede7938"
  [IS-08b-RASDML]="DEV2.0-0.8B bede7938a8c209c09f27400b79eed57948d6b75e /data/dev2/models/ix1/index-sweep/08b-RASDML-db5cccad-rbede7938"
  [IS-2b-RA]="DEV2.0-2B a53cf66a0d9d492a84b6617b61e7ce35fcd03af0 /data/dev2/models/ix1/index-sweep/2b-RA-3d4fde06-ra53cf66a"
  [IS-2b-RASD]="DEV2.0-2B a53cf66a0d9d492a84b6617b61e7ce35fcd03af0 /data/dev2/models/ix1/index-sweep/2b-RASD-341c2bd2-ra53cf66a"
  [IS-2b-RAUP]="DEV2.0-2B a53cf66a0d9d492a84b6617b61e7ce35fcd03af0 /data/dev2/models/ix1/index-sweep/2b-RAUP-37c85fff-ra53cf66a"
  [IS-2b-RA-a75]="DEV2.0-2B a53cf66a0d9d492a84b6617b61e7ce35fcd03af0 /data/dev2/models/ix1/index-sweep/2b-RA-a75-bbba9fad-ra53cf66a"
  [IS-L9IB]="DEV2.0-9B e51f9881b92f646cb0bd62b2876d4878cc8d16ec /data/dev2/models/ix1/index-sweep/L9IB-d7c48f9a-re51f9881"
  [IS-K-a12IB]="DEV2.0-9B e51f9881b92f646cb0bd62b2876d4878cc8d16ec /data/dev2/models/ix1/index-sweep/K-a12IB-68fed4cb-re51f9881"
)
for _m10 in KUP-a33 KUP-a25 KUP-a40 KIBM-a33 KIBM-a25 KIBM-a40 KSW-a33 KSW-a25 KSW-a40 KIB4-a33 KIB4-a25 KIB4-a40 KX-a33 KX-a25 KX-a40 \
  X{1..6}-a33 X{1..6}-a25 X{1..6}-a40 KIB4P-a33 KIB4P-a25 KIB4P-a40 KXP-a33 KXP-a25 KXP-a40; do  # 9B M10 BF16 release copies (X: amendment 5 cross-arm points)
  DIAGNOSTIC[M10-$_m10-bf16]="DEV2.0-9B e51f9881b92f646cb0bd62b2876d4878cc8d16ec /data/dev2/models/ix1/9b-m10/$_m10-bf16-re51f9881"
done
unset _m10
for _af in 4b-LHS17IB4-lrh 4b-LHS17IB4ML 4b-LHS23IB4 4b-LHS17IB4-s45 4b-SDMLIB4-s45 4b-LHS17UP-s34 4b-AFxALL 4b-AFxALL2 4b-AFxALL3 4b-XALLx 4b-XALLU2 4b-LRHxXALL-m50 4b-SDMLIB4-lrh 4b-LRHxALL 4b-LHS17IB4-lrq 4b-LHS17ML-lrh 4b-LHS17IB4X-lrh 4b-SDML-lrh ; do  # arm factory 4B BF16 release copies (4B owner: 4b-XALLx, 4b-XALLU2, wave 7 low-LR points)
  DIAGNOSTIC[AF-$_af-bf16]="DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f /data/dev2/models/ix1/af/AF-$_af-bf16-r13d42143"
done
for _af in KIB4W2-a40 KIB4L2-a40 KIB4Q-a40 KF-a33 KF-a40 KF-a50 KFxKIB-a40 KFxKIB-a50 KFxKIB2-a40 KF2-a40 KF2-a50 ; do  # arm factory 9B BF16 release copies
  DIAGNOSTIC[AF-$_af-bf16]="DEV2.0-9B e51f9881b92f646cb0bd62b2876d4878cc8d16ec /data/dev2/models/ix1/af/AF-$_af-bf16-re51f9881"
done
unset _af

mode="${1:-}"; shift || true
src="" model="" gpu="" gpus="" run="" rows="" rows_dir="" cache="" only="" tag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) only="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --src) src="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --run) run="$2"; shift 2 ;;
    --rows) rows="$2"; shift 2 ;;
    --rows-dir) rows_dir="$2"; shift 2 ;;
    --cache) cache="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' >&2; exit 2; }
[[ "$mode" =~ ^(parity|run|resume|ref|extra)$ ]] || usage
[[ -f "$src/.dev2-mirror.json" && -n "${REVISION[$model]:-}${DIAGNOSTIC[$model]:-}" && "$run" == /data/dev2/private/* ]] \
  || usage
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE")" == "$IMAGE_ID_PREFIX"* ]] \
  || { echo "image $IMAGE is not the frozen build" >&2; exit 1; }
[[ "$(git -C "$KIT" rev-parse HEAD)" == "$KIT_REVISION" ]] || { echo "kit is not at $KIT_REVISION" >&2; exit 1; }
S="$src/src/training/decision2"
if [[ -n "${DIAGNOSTIC[$model]:-}" ]]; then
  read -r repo revision pkg <<< "${DIAGNOSTIC[$model]}"
  [[ "$repo" == */* ]] || repo="llm-semantic-router/$repo"
else
  repo="llm-semantic-router/$model" revision="${REVISION[$model]}" pkg="$MODELS/$model-${REVISION[$model]:0:8}"
fi
manifest_sha="$(sha256sum "$pkg/MODEL_MANIFEST.json" | cut -c1-64)"
base_dir="$(python3 - "$pkg/MODEL_MANIFEST.json" "$HF_CACHE" <<'EOF'
import json, sys
base = json.load(open(sys.argv[1])).get("base")
if base:
    print(f"{sys.argv[2]}/models--{base['repo_id'].replace('/', '--')}/snapshots/{base['revision']}")
EOF
)"
[[ -z "$base_dir" || -d "$base_dir" ]] || { echo "pinned base snapshot missing" >&2; exit 1; }

render_nodes() {  # rocm-smi index -> "/dev/dri/renderDN /dev/dri/cardM"
  local bus r dev c
  bus="$(rocm-smi --showbus --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["card"+sys.argv[1]]["PCI Bus"].lower())' "$1")"
  for r in /sys/class/drm/renderD*; do
    dev="$(readlink -f "$r/device")"
    if [[ "$(basename "$dev")" == "$bus" ]]; then
      for c in "$dev"/drm/card*; do
        echo "/dev/dri/$(basename "$r") /dev/dri/$(basename "$c")"
        return 0
      done
    fi
  done
  return 1
}

take_lease() {  # gpu purpose hours
  local lease="/data/dev2/leases/gpu$1.lock"
  mkdir -p "$lease"
  if [[ -s "$lease/owner" ]] && ! grep -qx "track=eval-ix1" "$lease/owner"; then
    echo "gpu$1 is leased by another owner; refusing" >&2; return 1
  fi
  local tries=0
  until rocm-smi --showuse --showmeminfo vram --json | python3 -c '
import json, sys
card = json.load(sys.stdin)["card" + sys.argv[1]]
use, used = float(card["GPU use (%)"]), int(card["VRAM Total Used Memory (B)"])
if use > 5 or used > 2 * 2**30:
    sys.exit(f"GPU{sys.argv[1]} is busy: use {use}%, VRAM used {used / 2**30:.1f} GiB")
' "$1"; do
    tries=$((tries + 1))
    (( tries < 12 )) || return 1
    sleep 15
  done
  printf 'track=eval-ix1\npurpose=%s\nstart_utc=%s\nexpected_end_utc=%s\nrun_dir=%s\n' "$2" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(date -u -d "+$3 hours" +%Y-%m-%dT%H:%M:%SZ)" "$run" > "$lease/owner"
}

digest_dir() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-64); }

container() {  # name gpu workdir detach(0|1) script
  local name="$1" g="$2" work="$3" detach="$4" script="$5" nodes node devs=()
  nodes="$(render_nodes "$g")" || { echo "no render node for gpu$g" >&2; return 1; }
  for node in $nodes; do devs+=(--device "$node"); done
  local envs=(-e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
              -e HF_HUB_CACHE="$HF_CACHE" -e TOKENIZERS_PARALLELISM=false -e PYTHONDONTWRITEBYTECODE=1
              -e PYTHONPATH="$S:$KIT:$IMAGE_PYTHONPATH" -e HOME="$work/home" -e DECISION2_PACKAGE_DIR="$pkg"
              -e TRITON_CACHE_DIR="$work/triton" -e TRITON_CACHE_AUTOTUNING=1 -e HIP_FORCE_DEV_KERNARG=1)
  [[ -n "$base_dir" ]] && envs+=(-e DECISION2_BASE_DIR="$base_dir")
  local vols=(-v "$src:$src:ro" -v "$KIT:$KIT:ro" -v "$pkg:$pkg:ro" -v "$HF_CACHE:$HF_CACHE:ro" -v "$work:$work")
  [[ -n "$rows" ]] && vols+=(-v "$(dirname "$rows"):$(dirname "$rows"):ro")
  [[ -n "$rows_dir" ]] && vols+=(-v "$rows_dir:$rows_dir:ro")
  mkdir -p "$work/home" "$work/triton"
  local flags=(--rm)
  [[ "$detach" == 1 ]] && flags=(-d --rm)
  docker run "${flags[@]}" --name "$name" --network none --shm-size 8g --memory "${IX1_MEMORY:-256g}" \
    --device /dev/kfd "${devs[@]}" --group-add video --security-opt seccomp=unconfined \
    "${envs[@]}" "${vols[@]}" -w "$S" --entrypoint bash "$IMAGE" \
    -c "python3 -c 'import fla.ops.gated_delta_rule, causal_conv1d' || exit 97; $script"
}

no_fallback() {  # the kernels the package was scored with must have been used
  if grep -l "$FALLBACK" "$@" 2>/dev/null; then
    echo "a run used the reference kernel path" >&2; return 1
  fi
}

loading_shards() {  # shards launched but not yet past model loading
  local count=0 w
  for w in "$run"/shard-*; do
    [[ -f "$w/launched" && ! -f "$w/end_epoch" ]] || continue
    grep -q '"event": *"\(ready\|progress\|complete\)"' "$w/status.json" 2>/dev/null || count=$((count + 1))
  done
  echo "$count"
}

wait_for_room() {
  local deadline=$(( $(date +%s) + 3600 )) free
  while :; do
    free=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
    if (( $(loading_shards) <= 1 && free >= ${IX1_MIN_FREE_GIB:-400} )); then return 0; fi
    (( $(date +%s) < deadline )) || { echo "no room to start another shard" >&2; return 1; }
    sleep 15
  done
}

kit_run() {  # rows out -> shell command
  printf 'python3 -m decision_index run --engine %s --option model_id=%s --option revision=%s --option package_manifest_sha256=%s --option device=cuda:0 --rows %q --out %q --compact' \
    "$ENGINE" "$repo" "$revision" "$manifest_sha" "$1" "$2"
}

umask 077
mkdir -p "$run"
record="$run/launcher-$mode${only:+-only-${only// /_}}${tag:+-$tag}.json"
python3 - "$record" "$mode" "$model" "$revision" "$manifest_sha" "$IMAGE" "$KIT_REVISION" "$src" "${gpu:-$gpus}" <<'EOF'
import json, sys
path, mode, model, revision, manifest, image, kit, src, gpus = sys.argv[1:]
mirror = json.load(open(src + "/.dev2-mirror.json"))
json.dump({"schema": "ix1-launcher/1", "mode": mode, "model": model, "revision": revision,
           "package_manifest_sha256": manifest, "image": image, "kit_revision": kit,
           "decision2_mirror": {k: mirror[k] for k in ("commit", "tree", "content_manifest_sha256")},
           "gpus": gpus.split()}, open(path, "x"), indent=2, sort_keys=True)
EOF

if [[ "$mode" == parity ]]; then
  [[ "$gpu" =~ ^[0-7]$ && -f "$rows" ]] || usage
  take_lease "$gpu" "IX1 parity gate $model" 1
  tag="ix1-parity-$(tr 'A-Z.' 'a-z_' <<< "$model")-g$gpu"
  ref="$run/ref" kit="$run/kit"
  mkdir -p "$ref" "$kit"
  container "$tag-ref" "$gpu" "$ref" 0 "date +%s > $ref/start_epoch; python3 -m v2.eval.ix1.native_ref --package $pkg --rows $rows --out $ref/ref.jsonl ${base_dir:+--base-path $base_dir} > $ref/native_ref.log 2>&1; echo \$? > $ref/exit_code; date +%s > $ref/end_epoch"
  [[ "$(cat "$ref/exit_code")" == 0 ]] || { echo "reference pass failed" >&2; exit 1; }
  no_fallback "$ref/native_ref.log"
  cp -a "$ref/triton" "$run/cache-frozen"
  digest_dir "$run/cache-frozen" > "$run/cache-frozen.sha256"
  mkdir -p "$kit/triton"
  cp -a "$run/cache-frozen/." "$kit/triton/"
  container "$tag-kit" "$gpu" "$kit" 0 "date +%s > $kit/start_epoch; $(kit_run "$rows" "$kit") > $kit/runner.log 2>&1; echo \$? > $kit/exit_code; date +%s > $kit/end_epoch"
  [[ "$(cat "$kit/exit_code")" == 0 ]] || { echo "kit pass failed" >&2; exit 1; }
  no_fallback "$kit/runner.log"
  PYTHONPATH="$S" python3 -m v2.eval.ix1.parity --kit "$kit/results.jsonl" --ref "$ref/ref.jsonl" --out "$run/parity.json"
  exit 0
fi

if [[ "$mode" == ref || "$mode" == extra ]]; then
  [[ "$gpu" =~ ^[0-7]$ && -f "$rows" && -d "$cache" && -f "$cache.sha256" ]] || usage
  [[ "$(digest_dir "$cache")" == "$(cat "$cache.sha256")" ]] || { echo "frozen cache $cache changed" >&2; exit 1; }
  take_lease "$gpu" "IX1 $mode $model" 1
  if [[ "$mode" == ref ]]; then
    work="$run"
    [[ ! -e "$work/ref.jsonl" ]] || { echo "$work/ref.jsonl exists" >&2; exit 1; }
    script="python3 -m v2.eval.ix1.native_ref --package $pkg --rows $rows --out $work/ref.jsonl ${base_dir:+--base-path $base_dir} > $work/native_ref.log 2>&1"
  else
    [[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || usage
    work="$run/extra-$tag"
    [[ ! -e "$work/results.jsonl" ]] || { echo "$work already has results" >&2; exit 1; }
    script="$(kit_run "$rows" "$work") > $work/runner.log 2>&1"
  fi
  mkdir -p "$work/triton"
  cp -a "$cache/." "$work/triton/"
  container "ix1-$mode-$(tr 'A-Z.' 'a-z_' <<< "$model")-g$gpu" "$gpu" "$work" 0 \
    "date +%s > $work/start_epoch; $script; echo \$? > $work/exit_code; date +%s > $work/end_epoch"
  no_fallback "$work"/*.log
  exit "$(cat "$work/exit_code")"
fi

[[ -n "$gpus" && -d "$rows_dir" && -f "$rows_dir/panel.json" ]] || usage
if [[ "$mode" == run ]]; then
  [[ -d "$cache" && -f "$cache.sha256" ]] || usage
  [[ "$(digest_dir "$cache")" == "$(cat "$cache.sha256")" ]] || { echo "frozen cache $cache changed" >&2; exit 1; }
else
  [[ -n "$only" ]] || usage
fi
read -r -a gpu_list <<< "$gpus"
n="${#gpu_list[@]}"
[[ -f "$rows_dir/shard-0-of-$n.jsonl.gz" ]] || { echo "no $n-way shards in $rows_dir" >&2; exit 1; }
selected() { [[ -z "$only" || " $only " == *" $1 "* ]]; }
for k in "${!gpu_list[@]}"; do
  if selected "$k"; then take_lease "${gpu_list[$k]}" "IX1 full run $model" 4; fi
done
for k in "${!gpu_list[@]}"; do
  selected "$k" || continue
  g="${gpu_list[$k]}"
  work="$run/shard-$k"
  if [[ "$mode" == run ]]; then
    [[ ! -e "$work/results.jsonl" ]] || { echo "$work already has results; use resume" >&2; exit 1; }
    mkdir -p "$work/triton"
    cp -a "$cache/." "$work/triton/"
  else
    [[ -f "$work/results.jsonl" && -f "$work/end_epoch" ]] || { echo "$work has not ended" >&2; exit 1; }
    i=1
    while [[ -e "$work/end_epoch.$i" ]]; do i=$((i + 1)); done
    for f in start_epoch end_epoch exit_code; do mv "$work/$f" "$work/$f.$i"; done
    rm -f "$work/launched"
  fi
  shard="$rows_dir/shard-$k-of-$n.jsonl.gz"
  if [[ "$mode" == resume && -f "$work/rows.override.jsonl.gz" ]]; then
    [[ -f "$work/skipped.json" ]] || { echo "$work override without skipped.json" >&2; exit 1; }
    shard="$work/rows.override.jsonl.gz"
  fi
  wait_for_room
  touch "$work/launched"
  container "ix1-$(tr 'A-Z.' 'a-z_' <<< "$model")-s$k-g$g" "$g" "$work" 1 \
    "date +%s > $work/start_epoch; $(kit_run "$shard" "$work") > $work/runner.log 2>&1; echo \$? > $work/exit_code; date +%s > $work/end_epoch" > /dev/null
  echo "started shard $k on gpu$g"
done
