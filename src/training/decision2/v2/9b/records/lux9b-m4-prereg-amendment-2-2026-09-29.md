# 9B Milestone 4 preregistration, amendment 2: three queued seeds move to node B GPU0–2

Frozen 2026-09-29 ~00:15 UTC, before any K, P or KN development result was read (the Lux
reference and the D line, gist 06:05, are the only Milestone 4 results read so far). Trigger:
the coordinator's 07:35 UTC+8 reallocation (node B GPU0–2 → 9B; the 9B track may train on
node B with the same image and a frozen autotune cache; its formal comparisons stay on node A)
and the 07:55 UTC+8 handoff (the Milestone 4 worker's session ended while its chains ran).

Run names follow the chain drivers: `-s1` = seed 20260926, `-s2` = seed 1, `-s3` = seed 2.
The [preregistration](lux9b-m4-prereg-2026-09-29.md)'s "P-s1" (P seed 1, last in line) is run
**P-s2**.

## State at freeze (node A)

| Run | Chain | State |
| --- | --- | --- |
| lux-16k, D-a14 / a13 / a23 / a12 / a1 | chain-gpu7 | done (exit 0), 21:14–21:44 UTC |
| K-s1 (with preflights) | chain-gpu6 | done (exit 0), 21:14:08–23:50:28 UTC |
| K-s2 | chain-gpu6 | running since 23:50:28 UTC (GPU6) |
| P-s1 (with preflights) | chain-gpu7 | running since 21:44:28 UTC (GPU7) |
| K-s3 | chain-gpu6 | queued after K-s2 |
| KN-s1 (with preflights), KN-s2 | chain-gpu7 | queued after P-s1 (KN-s2 only if KN-s1 completes) |
| P-s2 | — | in no chain |

## Moves

| Run | Was | Now |
| --- | --- | --- |
| **K-s3** (K, seed 2) | chain-gpu6, after K-s2 | node B GPU0, same command plus `--preflight` (node-B load / parity / reload check) |
| **KN-s2** (KN, seed 1) | chain-gpu7, after KN-s1 | node B GPU1, once KN's preflight (run by KN-s1 on node A GPU7) and the node-B preflight have passed |
| **P-s2** (P, seed 1) | no chain | node B GPU2, once the node-B preflight has passed, and only if the projected milestone total stays within the 22 GPU-hour cap |

KN-s1 stays on node A GPU7, started by chain-gpu7 as planned.

- **Dequeue method.** A chain driver is stopped by SIGTERM to its bash process only. Each
  driver is a session leader without a terminal, so the running step's process tree (`arm.sh` →
  `job.sh` → docker client → container) keeps running and still writes its own start / end /
  exit files. A note goes into the chain log. chain-gpu6 is stopped before K-s3 starts on node
  B. chain-gpu7 is stopped after it has started KN-s1 and before KN-s2 starts on node B.
- Hand-launched steps run through `lux9b/m4/chain-step.sh` (same lease and log conventions as
  the chain drivers; logs `m4/logs/nodeb-gpu<N>.log` on node B).
- **Fallback.** If a node-B condition below fails, the moved runs go back to node A GPU6–7 in the
  original order (via `chain-step.sh`), and the failure is recorded. A node-B preflight failure
  is an environment failure, not an arm result: the arm itself does not stop.

## Node-B conditions (all checked and recorded before the first node-B GPU step)

1. The image is `sha256:f83b1d10…` (moved from node A by `docker save` / `docker load` if absent;
   ID verified).
2. Inputs are byte-identical to node A by SHA-256: the Lux 1.0 start directory, SELECT700,
   CAL698, the x60 / xn60 builds (train, teacher, manifest) and the gold-free typed-DEV / CSS-pilot
   prompts.
3. The code is an exact mirror of the pushed commit. The trainer and wrappers are unchanged
   from `9d90212dd`; this amendment adds only `chain-step.sh`.
4. The training autotune cache is one copy of node A's Milestone 4 training cache, which is
   itself one copy of the Milestone 3 cache. The copy is taken before the first node-B step.
   Its tree hash and file count are logged on both nodes and again after the node-B runs.
5. Node B GPU0–2 leases are written per COORDINATION. Writes go only under `/data`.

## Readouts stay on node A

Every development readout that feeds the seed rule, the soups, the α rule or the proxy drop
rule is made on node A:

- Node-B seeds' run records and SELECT-chosen checkpoints are copied to the same paths under
  node A's `/data/dev2/runs/9b/m4/` (SHA-256 verified).
- Their typed-DEV and CSS-pilot predictions are then re-made on node A with the node-B CAL698
  temperatures. Per-type temperatures do not change the argmax, so T, c_t, F_f, H3 and P do not
  depend on them.
- The in-arm node-B readouts are kept as a cross-node repeat. They are reported but not used.
- The SELECT checkpoint choice itself is made inside training on the training node (trainer
  unchanged).

## Unchanged

Arms, data, seeds, the α rule (including its HT-DEV switch clause), the finalist priority, the
formal runner on node A, the gate and the budget are unchanged. GPU-hours = wall-clock × GPUs on
either node, including the node-B preflight and the node-A re-reads. The storage rules of
amendment 1 also apply to node-B copies.
