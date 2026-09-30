# ~27B M5 state (resume file)

Updated: 2026-09-30 08:40 UTC+8 (M5 worker; prereg signed, training launch next)
Branch: `xunzhuo/decision-2-training-27b` (worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into
`xunzhuo/decision-2-training`). Gist file: `06-decision-2-27b.md`. Assignment: COORDINATION 2026-09-30 07:20.
Prereg: `m5-prereg-2026-09-30.md`.

## Target and budget

- **Goal:** post-key v3 paired lower bound > 0 vs AutoJev-27B (72.133), human transfer not below. Successor-rule
  reference: A20r's scored run (`/data/dev2/runs/27b/M4-A20r-soup/formal`, node B).
- **Budget:** 72 GPU-h cap. **Used: 0.486** (HT-DEV v2 references).

## Infrastructure

- **Private node link (node B → node A):** temporary ed25519 key on node B (`/data/dev2/tmp/27b-m5-xfer/`, mode 700),
  authorized on node A for rsync only (`from=<node B private address>,restrict,command="/usr/bin/rrsync
  /data/dev2/xfer/27b-m5"`, tagged `dev2-27b-m5-xfer-temp`; node A's `authorized_keys` backed up as
  `authorized_keys.bak.27b-m5-<UTC>`). ≈ 850 MB/s. rsync paths are relative to `/data/dev2/xfer/27b-m5` on node A.
  **Remove the key line and the node B key directory at milestone end.**
- **Mirrors:** `~/.cache/m5-work/mirror2.sh <commit>` mirrors to node A (workstation stream) and then to node B over
  the link; `mirror_to_node.sh` re-verifies node B's copy.
- **Staged on node A** (`/data/dev2/xfer/27b-m5/stage/`, per-file SHA-256 equal to node B): A20r soup checkpoint +
  package; F-b (`m4b/A1-soup/checkpoint`, 96 GB FP32; list `84437b35…`) + package.
- **ht-dev2 installed on node B** (`/data/dev2/private/panels/{goldfree,gold}/ht-dev2.*`, copied from node A,
  `panels verify` = registry hashes `90cd409a…` / `659c92b4…`), so node B collects and scores HT-DEV v2 itself.
- **Mixtures** (built twice on node B, identical; `MIXTURES.json` `9ad3da72…`): `a20` `4aa0dc96…` (= M4), `a20h`
  `4a9d93f5…` (71,753 rows, 33,033,018 tokens). Node A copy `/data/dev2/private/27b/m5-data/mixtures-m5-1/` hash-equal.
  C1 source-registry check on `a20h`'s rows new vs M3-A: clean (node B `m5-logs/c1-sources-a20h.json` `53af68da…`).

## Step 1 done: HT-DEV v2 references (node A, 0.486 GPU-h)

| Key (`/data/dev2/runs/27b/m5/htdev2/<key>`, node A) | H_dev2 | vs A20r |
| --- | ---: | --- |
| `dev2-27b-f1` (DEV2.0-27B current weights) | .5779 | +.0125 TIE |
| `m4-a20r-soup` | .5655 | reference |
| `m4b-f-b` | .5527 | −.0128 TIE |
| `eikos27b` | .5558 | −.0097 TIE |
| `autojev27` | .5443 | −.0211 FLAG |

## Step 2: training (prereg arms; running from mirror `7f07c4904`)

- Launch 1 (mirror `b31cd76e9`, 00:04Z) was refused by launch3's co-tenant check before any container (0 GPU-h);
  fixed by amendment 1 (`7f07c4904`), relaunched 00:09Z.
- **Node B lane GPU5–7** (`logs/lane-b.log`): FF20-s1 one-step 0.251 + reload 0.025 (0 / 32 argmax changes); full
  from 00:16:14Z (cap 13.5 GPU-h = 4.5 h wall), then FF20H-s1.
- **Node A lane GPU2–4** (`logs/lane-a.log`): FF20X-s1 probe (one-step), FF20-s2 one-step + reload passed; full from
  00:21:33Z, then FF20H-s2. Node A BESTs are hardlinked to `/data/dev2/xfer/27b-m5/relay/<ARM>-<SEED>/` with
  `SHA256SUMS`; node B pulls them with `m5-tail.sh pull` (private peer address in `/data/dev2/tmp/27b-m5-xfer/peer`).
- **Cross-node probe FF20X:** update-1 loss 1.148860306944698 and pre-clip gradient norm 424.828857421875 equal on both
  nodes; the one-step parameters differ bytewise in 25 of 27 backbone shards (FSDP / autotune float order; AdamW's
  first step amplifies it, as M4b amendment 2 found). Report only.
- Speed: ≈ 15.5–16 s per update after warm-up (F-b 12.5 s at 24.9k tokens per update; a20 28.1k). Projection ≈ 4.1 h
  wall per FF20 attempt (cap 4.5), ≈ 5.1 h per FF20H attempt (cap 5.83).
- Run dirs `/data/dev2/runs/27b/m5/<ARM>-<SEED>/`; status helper `~/.cache/m5-work/status.sh` (workstation).
  Liveness: `docker ps --filter name=d2-27b-m5-`. Stop files `/data/dev2/runs/27b/m5/STOP-<ARM>` (G1 / B1).

## Tail tooling (mirror `c82d3938a` or later)

- `m5-tail.sh {lease GPU | pull ARM-SEED | soup NAME CKPT... | reference GPU | readout NAME CKPT GPU | devgates NAME... |
  formal NAME CKPT GPU | mlx NAME GPU}` (node B; aux GPUs 0–2; node B GPU0 leased to 27b at 00:17Z).
- `m5-gates.sh MIRROR {gates | overlap | verdicts} NAME...` (node B CPU), `m5_verdicts.py` (items 1–7, beats-AutoJev).
- A20r node B reference readout (typed DEV + CSS pilot + HT-DEV v2, 03b172f1 cache): `readouts/m4-a20r-soup`
  (symlink `A20r-ref`), running on node B GPU0 from 00:17:55Z.

## Next

- G1 at ≥ 100 updates; FF20 seeds done ≈ 04:30Z → pull FF20-s2, soup M5-FF20, readout, devgates → B1.
