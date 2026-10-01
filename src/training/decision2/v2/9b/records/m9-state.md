# 9B M9 state (resume file)

Updated: 2026-10-01 15:10 UTC+8 (07:10Z). Branch `xunzhuo/decision-2-training-9b-m9` (worktree `vllm-sr-dev2-9b-m9`).
Prereg `records/lux9b-m9-prereg-2026-10-01.md` (`8570a5896`), pushed before any GPU job.

## Stage 1 (node C `/data/dev2/runs/9b/m9/`)

| Item | State |
| --- | --- |
| Inputs | x60 TRAIN / own-Lux targets, SELECT700 / CAL698, Lux 1.0 package and the 9B training cache staged from node A over the transfer key; Qwen3.5-9B-Base `68c46c4b…` downloaded with the host HF CLI (byte-identical to node A's copy on all 12 weight / config / tokenizer files); `data/READY.json` written 06:58Z |
| Chains (mirror `8570a5896`) | c1 L9-s1 (pre-warm: zero-step 2 min, one-step 2.5 min, marker 07:03Z) → gate → full; c2 L9-s2, c3 L9L-s1, c4 L9L-s2 started 07:03:41Z after the marker |
| B0 (retention ceiling) | **zero-step FAILED** at model build: the label-token readout needs a tied LM head and Qwen3.5-9B-Base's head is untied. Not rerun; B0 is diagnostic only (no arm depends on it) |
| Leases | node C GPU1–4 `track=9b-m9` busy, GPU5 idle (ours); GPU6–7 are IX1's (its follow-up started there at 07:03Z); GPU0 foreign, untouched |

## Node A (readouts, formal)

- SELECT700 / CAL698 and M9's readout cache (one copy of `m4/triton-cache`, 11,553 files, tree `579ca841…`) staged.
- The M10 probe prompts (`a324f1e2…`, 3,089 items) relayed from node E; the M9 probe panel (minus x60 overlaps) is built
  next, then the C0 (K-a13) and Lux 1.0 reference readouts on GPU6–7.

## GPU-hours

≈ 0.1 used (preflights) at 07:10Z; cap 120.
