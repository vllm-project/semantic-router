# Reasoning wave 1 (9B and 2B) — data lock (amendment 1 to the 9B / 2B prereg)

Locked 2026-10-04 15:10–15:20 UTC+8 before any 9B / 2B training step; builds `v2.reasoning.build_v1` at
`b289c312`, node D.

## Teacher run (complete)

`teacher/nl-v1/graphs.jsonl` SHA-256 `eb6d1b2b9a4d455d7304bb4df76227ae8dd2eb02b9e2de3ace401af4429857c2`, 25,441
records: 23,447 with a verified graph, 1,870 unsolved (no solution reached the gold key), 88 without a verified
node and 36 errors. The last two groups come from stopping four of the eight replicas at 14:55 to start the 9B self-
labels (in-flight requests were aborted); they carry no graph and are excluded. Teacher GPU time ≈ 12.5 GPU-h.

## Starts (identities checked with `dec_fingerprint` on node D)

- 9B `runs/reasoning/starts/9b-KIB4-a40`: `b6984475a3804714d1018b4ba18c2dc28ffa48d8bc293d26211f620bdd062ee3`, the FP32
  identity of the Lux-9B release (its BF16 copy `0ece5faa…` is the Index reference run's model).
- 2B `runs/reasoning/starts/2b-RASDML`: `8c8e98e34eb3200492a9f22d76a4cacfd4a6dee8855c1b6aec303d3cbbf88b9a`, the scored
  FP32 identity of the Sol-2B release (BF16 copy `e20df76c…`).

## Files

| | 9B (`data/9b-v1`) | 2B (`data/2b-v1`) |
| --- | --- | --- |
| released TRAIN | `2e72bcfd…` (KIB4) | `97157068…` (2b-RASDML) |
| replay (30,000) | `b727aebb…` | `ef298503…` |
| `train-tf` = `train-f0` (242,821 rows) | `b3e25298…` | `ea48865f…` |
| `weights-tf` / `weights-f0` | `b7ecd39a…` / `6db87e9f…` | `b06bad10…` / `ed03d832…` |
| `teacher-self` (52,007 rows) | `9fd384d7…` | `17b9fcd6…` |
| `rpdev-final` (1,193) / `rpdev-nodes` | `695b19dd…` / `affe0189…` | `695b19dd…` / `affe0189…` |
| `MANIFEST.json` | `b4396b4f…` | `69854988…` |

Problems: program 17,500 train / 1,193 dev (identical instances to the 4B build: same `rpdev-final` hash); teacher
22,508 train / 939 dev. Decontamination dropped the flagged teacher problems as at 4B (GSM8K 239, multi-hop QA and
HoVer the rest; no program problem).

## Disclosed observation before launch (4B wave, mid-training)

At steps 361 / 721 of 2,885 the 4B seeds' SELECT family-macro accuracy fell from .897 to .69–.88, almost entirely
in two small programmatic families (`targeted_quantized_median`, 90 rows; `pilot_string_composition`, 40 rows); the
RP-DEV final macro rose from .667 to .764 (`R4-TF-s1`) and .739 (`R4-F0-s1`). The recipe was kept unchanged for 9B /
2B as registered; the registered SELECT floor on the interpolation points is the guard.
