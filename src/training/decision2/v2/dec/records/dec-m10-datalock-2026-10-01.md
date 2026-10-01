# Decoder Milestone 10 — data lock (2026-10-01)

Preregistration [`dec-m10-prereg-2026-10-01.md`](dec-m10-prereg-2026-10-01.md) (`2dea44d6f`); tooling `06206fef9`.
Written before any M10 GPU job. Mirror `2dea44d6f06c0da85be417ee463b7d21aad20c5a-src_training_decision2` on node E
and node F (content manifest `b2b2d24b…`, 5,113 files); the decoder test suite passes in image `dbe5f32b` on node E
(131 tests). Every value below was computed on both nodes and is identical on both.

## Inputs (copied from node B over the temporary transfer key, then hash-checked)

| Input | Node path under `/data/dev2/` | SHA-256 |
| --- | --- | --- |
| N4XF TRAIN `m4-xl-full-29m` | `runs/dec/m10/inputs/base/m4-xl-full-29m/train.jsonl` | `c7d51219e9f0fdd1…` |
| N4XF own-Lux teacher | `runs/dec/m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl` | `e2ff27ce2fc397c8…` |
| SELECT700 / CAL698 | `runs/dec/m10/inputs/sel700-cal698/` | `32a4352d…` / `19cc1a8c…` (`SHA256SUMS` OK) |
| Qwen3.5-4B-Base `1001bb4d826a52d1f399e183466143f4da7b741b` | `models/Qwen--Qwen3.5-4B-Base/<rev>/` | shards `df547074…`, `590fbaac…` (= their LFS ids); root-file list `d7097eaf…`; no `lm_head` tensor (tied) |
| Nox 1.0 `cde2a68dbaa557ea65dc458104d410a0802ee259` | `models/Decision-1.0-Nox-4B/<rev>/` | file list `b3e8aec1…` (NT only) |
| C0 = DEV2.0-4B weights (the N4XF soup) | `runs/dec/m10/inputs/N4XF-soup/` | per-file list `df602ca9…` (= M9's `4b-I`) |
| Decoder Triton cache, image `dbe5f32b` (node B) | seed `runs/dec/m10/inputs/triton-T0-dbe5f32b2263/`, working copy `runs/dec/m10/triton-cache/dbe5f32b2263/` | 30,182 files, manifest `a4be2c23…` |
| Panels (gold-free) | `runs/dec/panels/` | typed DEV `a17ec4b6…`, CSS pilot `598319a4…`, HT-DEV v2 `90cd409a…`, Score5-typed-DEV `8e35bfff…`, `hs1-dev` `49f192a7…`, PN1 dev `79dbf999…` |

## TRAIN (`ops/m10/m10_data.py`; `runs/dec/m10/data/m10-4b-base/`)

| Field | Value |
| --- | --- |
| `train.jsonl` | **`c385406e8f78a2ae257cf2479b7088a523be18009e55e16caf59327c34260f09`** |
| `train.ids.jsonl` | `be8c5d4d067f9df8…` |
| Rows | 58,739 (Choice 21,400 / Noul 23,070 / Score 14,269); 3 quarantine rows dropped (`m2:multihop:41be17b1…`) |
| Teacher coverage | 54,980 rows own-Lux KL; 3,759 gold-only (H7 / H8); the 3 teacher records outside TRAIN are the quarantined rows; 0 input-hash mismatches |

The rows are N4XF's lines byte for byte in N4XF's order, so the tokens are M7 / M9's 4B base (29,404,539 by the Nox /
Eos tokenizer); the trainer's provenance reports the count with the base tokenizer.

## Label-token alphabet (real tokenizer, in the image)

255 one-token Choice labels (A..Z, then AA..JT among the single-token capital pairs), SHA-256
`c493343e7a9a72aa93011d2e62c46b0b2cc3fae55cbd74ebbf2c57f93bb6c028`; Noul `no` / `yes` = ids 2083 / 9405; the answer
cue ends with the newline token, so a label is read at line start, as in the options list.

## Retention probes (built after this lock, before any readout)

Sources downloaded on node E's host at pinned revisions: cais/mmlu `c30699e8356da336a370243923dbaf21066bb9fe`
(`all/validation`, `all/dev`), allenai/ai2_arc `210d026faf9955653af8916fad021475a3f00453` (Challenge / Easy
`validation`), openai/gsm8k `740312add88f781978c0658806c59bc2815b9866` (`main/train`). The suite check reads the
Index suite on node C (it uses `test` splits for MMLU, ARC and GSM8K; only hit counts leave node C). The probe build
record states the counts and hashes.

## READY

`runs/dec/m10/data/READY.json` (`train_sha256`, `teacher_sha256`) is written on both nodes after this record is
pushed; every seed re-hashes TRAIN and the teacher against it before it starts.
