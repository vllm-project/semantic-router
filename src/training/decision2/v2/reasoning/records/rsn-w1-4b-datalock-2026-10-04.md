# Reasoning wave 1 (4B) — data lock (amendment 1 to the prereg)

Locked 2026-10-04 14:31 UTC+8, before any wave-1 training step. Build: `v2.reasoning.build_v1` at `9dc430ad3`,
node F, `/data/dev2/runs/reasoning/data/4b-v1/`, `MANIFEST.json` SHA-256 `765dc83b505e92cd5e77d51e135b5debf8a80e88056e350c369f17a0d20bc872`.

## Change against the prereg (disclosed)

The teacher run (25,441 pool rows) was not finished at lock time. The lock takes the 16,532 teacher records written
by 14:29 (`graphs-lock-w1.jsonl`, SHA-256 `1b4999bdce94ec077655134c78805dd5c6f7e67f28cdc8a8e5d57e139498aa62`). The
teacher processes the pool in a hash order, so these are a random subset; nothing else depends on which records
finished. Reason: the GPUs of node F were leased and idle, and the coordinator asked for them to be used or released
(14:06). The complete teacher set feeds the 9B / 2B waves.

## Inputs

| File | SHA-256 |
| --- | --- |
| `nl-pool-v1/pool.jsonl` (25,441 rows) | `02d7b1067dc4784d164aa2447c37b6ced7948627b7253b1b815b20ef787f1a1d` |
| `replay.jsonl` (30,000 rows) | `d6e58ccd9f2a00e654c9ac128d6c1a98b9ba59741d5255e72ba82ece0055a506` |
| self-labels (released Nox-4B, T = 1, 55,441 rows, three shards) | `26402f60…`, `00b3ec44…`, `6cfe51e3…` |
| `train-tf.jsonl` = `train-f0.jsonl` (197,510 rows) | `13403d1ec9d7ae032d5c774edb2b04114011904e2bb773a88336865908dc7d1c` |
| `train-tfm.jsonl` (197,510 rows) | `a1f04c2a941ac8bb993c3b438f6cb69636e749f81ff8ead28190a0f9aad8426c` |
| `weights-tf.jsonl` | `b40cc5f3c990e6d78452ff71edb724fafae127d3f349ae259b446184363bd595` |
| `weights-tfm.jsonl` | `acc7d59711e1c24e652b7a370156f6623ca27f567df56662369953d37711c7d3` |
| `weights-f0.jsonl` | `5169da7acd8ae3dd14a411235972e1c2717e350ffbf80a22c60e0710bc87dd2d` |
| `teacher-self.jsonl` (44,491 rows: replay and teacher-problem finals) | `6a8489eb1ba75c4519ce60a72233c7daaf5f397dca5cdf58dc4439377c1b87aa` |
| `rpdev-final.jsonl` (1,193 rows) | `695b19dd7eaf4016f81d41830dc8c1b9d71787663f553c8b1605ab620ed743a0` |
| `rpdev-nodes.jsonl` (7,362 rows) | `bb92aefa36be7b0d0c5378ad5c0181590c70cb1d15f1f9cff24f2c26dbfa099a` |

## Counts

- Problems: program 17,500 train / 1,193 dev; teacher 14,786 train / 620 dev (teacher status of the locked records:
  15,286 with a verified graph, 1,113 unsolved, 1 without verified nodes … as counted by the build).
- Rows per arm 197,510: choice 73,676, noul 117,327, score 6,507; node views native 73,917, statement 40,784,
  multi 20,115, score 703.
- Decontamination (13-gram, 171,017 evaluation records, 5 boilerplate grams): 3,121 rows flagged; 298 problems
  dropped (all teacher problems: GSM8K 135, HotpotQA 71, HoVer 41, MuSiQue 24, 2WikiMultihopQA 21, NLI 3), 2,826
  further node views dropped. No program problem was flagged.

## Baseline (released Nox-4B, dev readout before training)

RP-DEV finals: arithmetic graphs .453, code traces .567, causal queries .687, logic arithmetic .700, boolean .947,
ordering .718, rule chaining .778, swaps .490, truth chains .667. Teacher dev node views .83–1.00.
