# Decoder Milestone 8-small — data lock (2026-09-30)

Identities of every M8-small training input, recorded and pushed before any training job. The chains re-hash each
TRAIN and teacher file against a `READY` file written only after the lock part naming its hash is committed.
Preregistration [`dec-m8s-prereg-2026-09-30.md`](dec-m8s-prereg-2026-09-30.md) (`c99cf9b7f`).

## Part 1 — starts, top-up files, control teacher, exposure (locked 2026-09-30 ≈17:40 UTC+8 / 09:40Z)

Built on node B, CPU only, by `ops/m8s/m8s-prep.sh` from the exact mirror of `c99cf9b7f` (09:31–09:34Z); lock check
`m8s_lock.py check --part 1` from the mirror of `da11d3c68` (the lock's top-up path fix): **PASS** for both tiers
(`lock-2b-part1.json` `d619864f…`, `lock-08b-part1.json` `7582a29f…`).

**Starts** (`hf download --local-dir`, every `MODEL_MANIFEST.json` file re-hashed, checkpoint identity recomputed):

| Tier | Repository @ revision | Files | Checkpoint identity (= BF16 copy of the scored soup) |
| --- | --- | ---: | --- |
| 2B | `llm-semantic-router/DEV2.0-2B@a53cf66a0d9d492a84b6617b61e7ce35fcd03af0` | 32 | `32872f2968e38aa99901797aaab5e12e32e281bf67bb924037d443394411170d` |
| 0.8B | `llm-semantic-router/DEV2.0-0.8B@bede7938a8c209c09f27400b79eed57948d6b75e` | 29 | `3f02f0e5fc68a1fd985ce48bf3a3bb2d4ed16d1a1705c2258317b78a4d6514f4` |

**Top-up files** (`m8s_compose.py`, seeds `dec-m8s-topup-<tier>-v1`, shared tokenizer `06b95093…`):

| Item | 2B | 0.8B |
| --- | --- | --- |
| Recipe (r2-clean) | `m4-v2m-ret-r2` `1527b38b…` (human 12.94M / typed 16.22M native tokens) | `m6-e8f-r2clean` `f9f3c022…` (human 18.02M / typed 120.22M) |
| TRAIN sha256 | `24a8a36cf39391d8383e25d0d915516f31fce0ecccabd89ed4c602eb7d23227a` | `e1a3ec209e6780d935ac77bf5fd957134b7d6dcb69a18b49f00d0eb21348b8cc` |
| Rows (human / typed) | 12,247 (8,762 / 3,485) | 10,914 (7,978 / 2,936) |
| Native tokens (human / typed) | 6,061,994 (3,037,668 / 3,024,326) | 6,029,501 (3,006,774 / 3,022,727) |
| Rows C / N / S (human; typed) | 2,115 / 4,800 / 1,847; 1,404 / 1,167 / 914 | 6,158 / 872 / 948; 1,748 / 774 / 414 |
| Parts file | `cbf0af2b…` | `069f82e0…` |
| compose.json | `030dd2c3…` | `eaa252c4…` |
| Mixed groups / quarantined rows | 0 / 0 | 0 / 0 |
| Exposure receipt (r2 payload `2194716a…`) | `e6650160…`: 0 groups, methods agree | `cc090bb7…`: 0 groups, methods agree |
| C1 registry source keys | 0 hits (family-name match `pilot_narrative_reading` ↔ "narrative": project-generated rows, as in M6; report only) | same |

Largest human sources — 2B: GoEmotions 652, GSM8K 384, HotpotQA 329, ArgQ 282, DRCD 282, CMRC 256; 0.8B: CLINC150
1,821, MultiNLI (non-fiction) 1,795, Banking77 1,225, GoEmotions 466, ABCD 440. Typed — 2B: `generated_stage4_v2`
1,017, verifiable a2 819 / a6 615 / a4v2h 204; 0.8B: `generated_stage4_v2` 1,394, `generated_stage1_3` 1,116.

**Control teacher (2B C only):** `v2.dec.compose_teacher` from S2T's own-Sol targets (`947bc65b…`, package
temperature 1.30036): `teacher/2b-C/teacher.jsonl` **`bbb2b5d93cdde755cb02c1f49555518510f5811b3aa89a2609e1785e7d96df3b`**,
12,247 / 12,247 rows. Own-Sol argmax = gold: human C / N / S .750 / .724 / .446, typed .694 / .731 / .581. The 0.8B C
arm has no teacher (E8F's objective).

**READY after this commit:** `data/2b/topup`, `data/08b/topup`, `teacher/2b-C`.

## Part 2 — A20r teacher files (pending)

After the parity gate and the three label shards: `m8s_lock.py teachers`, then `check --part 2` per tier.
