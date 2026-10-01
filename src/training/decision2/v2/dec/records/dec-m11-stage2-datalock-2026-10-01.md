# Decoder M11 stage 2 — data lock (2026-10-01)

Prereg [`dec-m11-stage2-prereg-2026-10-01.md`](dec-m11-stage2-prereg-2026-10-01.md) (`9c8798576`); code `698787dee`
(`ops/m11/m11_s2data.py`, `m11-s2prep.sh`). Built at 07:46Z on node E and 07:48Z on node F in the decoder image
(CPU, `--network none`); the two builds are **byte-identical** (every SHA-256 below on both nodes). Written before any
stage-2 GPU job.

## Inputs (hash-checked on both nodes)

| Input | Path under `runs/dec` | SHA-256 |
| --- | --- | --- |
| M10 4B TRAIN `m10-4b-base` | `m10/data/m10-4b-base/train.jsonl` | `c385406e8f78a2ae…` |
| N4XF own-Lux teacher | `m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl` | `e2ff27ce2fc397c8…` |
| IB1-r3 TRAIN (`@31b200a3`, node A read-back) | `m11/inputs/ib1/ib1.train.jsonl` | `1e1b08f3d37f9051…` |
| IB2 TRAIN (`@c5dbdd0a`, node A read-back) | `m11/inputs/ib2/ib2.train.jsonl` | `ee137efa8bbf86e5…` |
| Tokenizer | `Qwen/Qwen3.5-4B-Base@1001bb4d…` | (M10's model manifest) |

## Build (`m11/data/s2/report.json` `99dc7ea5f2af5310…`)

- **T** = 29,404,539 tokens (the trainer's head encoding; equal to M10's count for LH). **B** = 0.25·T = 7,351,135 (the
  transfer-only pool holds 7,504,930 tokens, so the 25% share binds for both arms; all-IB pool 10,173,355). Longest
  row 7,754 tokens (≤ 8,192).
- **Shared base subsample:** 44,025 of 58,739 rows, 21,992,229 tokens (74.8% of T), 28,418 groups.

| Arm | `train.jsonl` SHA-256 | Rows | Tokens (vs T) | IB rows / tokens (share) | IB pool fraction |
| --- | --- | ---: | --- | --- | ---: |
| `4b-LHB` | `b3d83c1278fa17b5779d5b83b2fa6bd3f8adb38a5eb4de57ddb9f0e9c79b1bd9` | 79,328 | 29,342,093 (.998) | 35,303 / 7,349,864 (.2505) | .723 |
| `4b-LHBx` | `1152137c8445abeba20a50bfd8239aa18b510e27f6e755a2f4850767f9c2a6c2` | 81,133 | 29,342,683 (.998) | 37,108 / 7,350,454 (.2505) | .980 |

IB rows / tokens by family:

| Family | `4b-LHB` | `4b-LHBx` |
| --- | --- | --- |
| `args` | 3,764 / 987,948 | 5,074 / 1,339,197 |
| `copa` | 634 / 63,164 | 858 / 85,604 |
| `poem` | 2,589 / 496,628 | 3,509 / 673,250 |
| `sentfin` | 4,917 / 528,506 | 6,669 / 716,523 |
| `sms` | 834 / 102,981 | 1,127 / 139,524 |
| `snips_rel` | 1,951 / 617,473 | 2,645 / 837,295 |
| `snips_sel` | 1,930 / 397,011 | 2,617 / 538,302 |
| `fc_rel` | 4,152 / 1,194,387 | 5,614 / 1,619,017 |
| `fc_ready` | 1,867 / 462,459 | 2,529 / 626,977 |
| `ytspam` | 1,042 / 139,206 | 1,419 / 188,810 |
| `argq` | 3,722 / 432,234 | 5,047 / 585,955 |
| `w2c` (in-distribution) | 333 / 151,520 | — |
| `isarc` (in-distribution) | 631 / 85,176 | — |
| `hover` (in-distribution) | 2,906 / 1,088,977 | — |
| `gsm2` (in-distribution) | 4,031 / 602,194 | — |

`train.ids.jsonl`: `4b-LHB` `143c6c2e88b6c5b6…`, `4b-LHBx` `3332969e30d56b67…`. IB rows carry no teacher record
(`--teacher-partial`: gold-only); base rows keep N4XF's own-Lux targets.

## Retention-probe overlap (prereg rule)

M10's 13-gram check against both TRAIN files (`m11/probes/hits-4b.json` `9239b286f9ec4f6f…`): **628 of 3,089 probe
items hit, all GSM8K-train hold-out items** (628 of its 1,000; IB2's `gsm2` is GSM8K train); MMLU (1,265) and ARC (824)
have no hit. The stage-2 probe gold (`m10-probes.4b.gold.jsonl`, node A) drops those 628: 2,461 items, GSM8K 372. LH
and both arms are scored on it, so the stage-2 retention numbers are not comparable with M10's (disclosed).

## READY

`m11/data/READY-4b-s2.json` (`arms` = the two `train.jsonl` SHA-256 above, `teacher_sha256` = `e2ff27ce…`, this
record's name) is written on both nodes after this record is pushed; every stage-2 seed re-hashes its TRAIN and the
teacher against it.
