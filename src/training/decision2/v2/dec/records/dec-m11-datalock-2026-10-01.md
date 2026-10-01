# Decoder Milestone 11 — data lock (stage 1: 2B and 0.8B; 2026-10-01)

Identities of every M11 stage-1 input, recorded and pushed before any M11 GPU job. Preregistration
[`dec-m11-prereg-2026-10-01.md`](dec-m11-prereg-2026-10-01.md) (`b8c3caf22`); ops code `cef1a1b66`. Built on nodes E and F
(CPU only) by `ops/m11/m11-prep.sh` from the exact mirror of `cef1a1b66`, 07:02–07:06Z; inputs relayed from node B over
the temporary transfer key (`rsync -aL`, 18 / 18 transfers OK; the relay script was deleted after use).

## Starts and references (`/data/dev2/models/`, identical on E and F: model manifest `da8866c4…`)

| Item | Identity |
| --- | --- |
| Qwen3.5-2B-Base `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` (HF `main`) | `model.safetensors-00001-of-00001` `928acbf1…` = its HF LFS id |
| Qwen3.5-0.8B-Base `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` (HF `main`) | `model.safetensors-00001-of-00001` `c2b1e5a1…` = its HF LFS id |
| Sol 1.0 `ce0c018a…`, Eos 1.0 `363c4a5e…` | node B's hf-cache snapshots, dereferenced |
| C0 2B = DEV2.0-2B `a53cf66a…` (weights of current `main` `56950ec5`) | node B's M8s start copy (checkpoint identity `32872f29…`, M8s data lock part 1) |
| C0 0.8B = DEV2.0-0.8B `bede7938…` (weights of current `main` `e13a40f8`) | node B's M8s start copy (identity `3f02f0e5…`) |

## TRAIN and teacher

| Tier | TRAIN (byte for byte the released r2-clean recipe) | Rows / native tokens | Teacher |
| --- | --- | --- | --- |
| 2B | `m4-v2m-ret-r2` `1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c` | 56,141 / 29,162,107 | `v2.dec.compose_teacher` of S2T's own-Sol targets (`947bc65b…`) onto TRAIN: **`3f92e8c108c5a9949fda4b08638907fcc80175f68efec18bece912d0edc8eea1`**, 56,141 / 56,141 rows covered (same hash on E and F); KL 0.5 on every row |
| 0.8B | `m6-e8f-r2clean` `f9f3c0229551357ad1cb8e6923c64c636ee697e71d162f059548ae223e78d8ae` | 162,696 / 138,244,874 | none (E8F's objective) |

Neither file holds a row of the M7 quarantine group. SELECT700 `32a4352d…` / CAL698 `19cc1a8c…` (M10's copies on each
node, `m10/inputs/sel700-cal698`).

## NT max length (`m11_lengths.py`, the 1.0 tokenizer, label-token prompt, no limit)

| Tier | Longest label prompt (TRAIN / SELECT / CAL) | TRAIN rows > 8,192 / > 8,448 | NT `--max-length` |
| --- | --- | --- | --- |
| 2B | 8,252 / 240 / 240 | 1 / 0 | **8,448** |
| 0.8B | 8,272 / 240 / 240 | 78 / 0 | **8,448** |

(`lengths-2b.json` `3ad8697f…`, `lengths-08b.json` `60c937bc…`.) LH keeps 8,192: the head prompt is the released
recipes' prompt.

## Retention probes per tier (`m10_probes.py overlap --kind train`, 13-grams, against each tier's TRAIN)

| Tier | Candidates hit | Probe-panel items hit (dropped from the tier's probe gold) |
| --- | ---: | --- |
| 2B | 831 (all GSM8K) | **100 of 3,089** (all GSM8K: the 2B recipe holds GSM8K train rows) → 2,989 items |
| 0.8B | 0 | 0 of 3,089 |

(`overlap-train-2b.json` `d34c96e1…` / `hits-2b.json` `55b90734…`; `overlap-train-08b.json` `2a3e3727…` / `hits-08b.json`
`250ad11a…`.) The tier probe gold files are built on node A from M10's gold (`5c674e35…`) by `m11-score.sh probe-gold`
before any probe is scored.

## Caches

Per node, `m11/triton-cache/{2b,08b}-{train,read}`: `cp -a` copies of that node's M10 decoder cache
(`m10/triton-cache/dbe5f32b2263`, 30,182 files).

## Gates enforced by the chains (stricter than the prereg's global rule)

No new training seed starts once a node's M11 GPU-hours exceed 45 (both nodes together stay below the prereg's 95);
no NT seed starts once node E's exceed 30. Arm caps 2B 7, 0.8B 9 GPU-h.

## READY

`runs/dec/m11/data/READY-2b.json` (`train_sha256`, `teacher_sha256`, `nt_max_length` 8448) and `READY-08b.json`
(`train_sha256`, `nt_max_length` 8448) are written on both nodes after this record is pushed; every seed re-hashes
TRAIN (and the teacher) against them before it starts.
