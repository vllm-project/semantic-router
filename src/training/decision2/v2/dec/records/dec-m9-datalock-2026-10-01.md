# Decoder Milestone 9 — data lock (HR2 efficacy pilot at 4B; 2026-10-01)

Preregistration [`dec-m9-prereg-2026-10-01.md`](dec-m9-prereg-2026-10-01.md) (`f22a5797c`). Built on node A, CPU only,
by `ops/m9/m9-prep.sh` from mirror `e9a357ed408cd569fa41b436c9e578354beaa99a-src_training_decision2` (tree
`36e2274b…`), image `dbe5f32b`, `--network none`; 2026-10-01 02:55Z. Lock check `ops/m9/m9_lock.py`: **PASS**, no
failure. Nothing was uploaded; no GPU time.

## Inputs (all SHA-256-verified before use)

| Input | Where | SHA-256 |
| --- | --- | --- |
| Base `m4-xl-full-29m` TRAIN (N4XF's mixture) | node B → node A over the direct link | `c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60` |
| Pool `m6-xl-full-59m` TRAIN | node B → node A | `160812e2c3d39bd09fcbe6c98e2b4b1633f6680bd435241d3a83e791d5dfe33b` |
| N4XF composed own-Lux targets | node B → node A | `e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c` |
| `lux-all-59m` own-Lux targets | node B → node A | `a1bafad59411901cd8a13451b741a186fece445ab78ce9b0466fc5adc7f1f793` |
| XL r2 recipe id list `m3/mixtures/xl-r2/mx-xl-full-r2.ids.jsonl@100536133e19` | private dataset | `7843afb7b2bbb315902b6748d6559532f384283d8efd836b935daac5f55dbd4a` |
| **HR2 TRAIN** `m5/hr2/hr2.train.jsonl@afc3bc1e1d68` | private dataset | `0fd9b2dbe68f863dc8acd13b3466b7e9aaf32d2308bc740f012e5c9717502f3a` |
| HR2 DEV `m5/hr2/hr2.dev.jsonl@afc3bc1e1d68` | private dataset | `697c31428ae821d7c0238a64ced8b997cc5671be47307f88a273d082bc5aaebe` |
| Quarantine `ops/m7/specs/m7-quarantine-groups.json` | repo | `29dadde119eb6353f9a8d19e5f303efa8d1c666445b87c3aed6e0240e86f3dfa` |
| N7C `train.ids.jsonl` (M7, containment report only) | node B → node A | `0dcede5cdce3ff567b9a2fb94d9a42dac5e59d57514461ad6506391e7ab17711` |
| r2 exposure payload `excluded-groups.json` | node A | `2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914` |
| Tokenizer (Eos / Nox `363c4a5e`) `tokenizer.json` / `tokenizer_config.json` | node A HF cache | `06b95093…` / `bee8eba3…` |

## Composition (`compose.json` `acc6255b…`)

- **Base:** 58,739 rows / 29,404,539 tokens after the quarantine (3 base rows and 3 pool rows dropped).
- **HR2 block:** 27,697 rows, 22,336 groups, **16,754,968 tokens** (equal to the data track's count): Choice 10,207 /
  Noul 10,346 / Score 7,144; `hs3_pref` 7,099, `hs3_help` 4,019, `eth_util` 3,108, `prm_step` 3,562, `vitc` 3,526,
  `allegro` 3,125, `eth_cs` 1,084, `eth_deon` 1,094, `eth_just` 1,080; en 22,166, pl 3,182, zh 879, others ≤ 318. No HR2
  row shares an id, group or input hash with the base.
- **C9 filler:** 34,325 rows / 21,186 groups (Choice 11,845 / Noul 13,440 / Score 9,040; 2,425 H7 / H8 rows gold
  only) of 32,136 eligible pool groups (25.3M tokens), seed `dec-m7-fill-4b-v1`, budget 16,603,410. It **contains all
  28,986 filler rows of M7's N7C**, as designed.
- **Match:** H9 46,159,507 vs C9 46,159,962 tokens (C9 − H9 = +455, 0.001%; limit 0.5%).

## Locked files (node A, `/data/dev2/runs/dec/m9/`)

| Arm | TRAIN rows / tokens | TRAIN SHA-256 | Teacher SHA-256 (covered / rows; gold-only pools) |
| --- | --- | --- | --- |
| **H9** | 86,436 / 46,159,507 | `ba746c95fba92f6938a8a59a8a1651f0e2856e5486562f9bb032cd15dde3dfaf` | `819848be5607686a03f9eae93ede2a28e858570cd33c59b91aff2ef397166d80` (54,980 / 86,436; `base-gold` 3,759, `hr2` 27,697) |
| **C9** | 93,064 / 46,159,962 | `6bc068d916a2713371f2cba23d25daa7d8cc80d0f631afe7ff4d9187bd8de583` | `97da8fe6b9a36db3c993287f7f5825fc055b4a2a5f09e3c04f94e8cd51b93682` (86,880 / 93,064; `base-gold` 3,759, `fill-gold` 2,425) |

- `train.ids.jsonl`: H9 `38058ff8…`, C9 `247b85eb…`; C9 `teacher-new.jsonl` `ab7b2925…`; `lock.json` `aa6e9c6d…`.
- **Exposure receipts** (r2 payload): 0 groups for both files (H9 `bb931db7…`, C9 `ee001f7c…`).
- **Teacher:** every row outside the gold-only pools has an own-Lux target; **no HR2 row has a target** (27,697 of
  27,697 gold only).
- **Isolation:** no TRAIN row of either arm shares a canonical state with SELECT700 (`32a4352d…`), CAL698
  (`19cc1a8c…`), typed DEV, the CSS pilot, HT-DEV v2 (`90cd409a…`), Score5-typed-DEV (`8e35bfff…`), `hs1-dev` or the
  HR2 DEV prompts (0 hits in every panel).
- **C1 registry guard** (registry `db00d4fe…`, 38 keys): no TRAIN source hit; one family-name hit reported only
  (`pilot_narrative_reading` ~ "narrative", a base family already in DEV2.0-4B's mixture).
- `v2.common.eval_only` guard and row check passed in the composer.

## HR2 DEV slice prompts

1,615 gold-free prompts (`69d507b6…`, every row rendered exactly as the TRAIN row format) in the node-A decoder panel
directory; gold (`c0de2876…`) under `/data/dev2/private/dec/m9/` (mode 600); 9 families (`allegro` 250, `eth_cs` 94,
`eth_deon` 100, `eth_just` 106, `eth_util` 354, `hs3_help` 126, `hs3_pref` 263, `prm_step` 60, `vitc` 262).
Score5-typed-DEV gold-free prompts copied into the same directory (`8e35bfff…`, equal to the eval panel file).

## Release

READY files (`<sha>` of TRAIN and teacher, this commit cited) are written after this record is pushed; the chains
verify both hashes before each seed.
