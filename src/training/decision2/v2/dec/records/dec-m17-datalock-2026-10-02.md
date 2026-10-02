# Decoder M17 — data lock (2026-10-02)

Prereg [`dec-m17-prereg-2026-10-02.md`](dec-m17-prereg-2026-10-02.md) (`da770d98a`). Built from mirror `1ce8b2220`
(`ops/m17/m17_data.py` through `m17-prep.sh`, host python3, CPU) on node E and node F at 01:58Z; the two builds are
**byte-identical** (digest of the sorted per-file SHA-256 list `f956f8cc…` on both nodes; every hash below equal).
Written before any M17 GPU job.

## Inputs (hash-checked on both nodes)

LH's released TRAIN `c385406e…` (58,739 rows; T = 29,404,539 tokens), M12 `4b-LHA10` TRAIN `d41cdd1a…` with ids file
`e429ce9b…` (its base block = LH's TRAIN byte for byte, checked), M12 `4b-LHA` TRAIN `e9d30c8f…` with ids file
`c7193bd8…` (base block checked; its IB block is the pool), M13's 4B SD targets `7639fab1…` (M15's copy; covers LH's
TRAIN ids exactly, checked). The ids-file hashes match the M12 data lock's published prefixes.

## LH's TRAIN and the pool

- English rows 33,777 / 17,224,201 tokens; non-English 24,962 rows / 12,180,338 tokens (share .4142).
- Candidates (rows of all-`en` groups): 33,771 rows / 17,221,416 tokens, 20,704 groups, 23,147 (family, group)
  units. Correction to the prereg's parenthesis: 126 groups have more than one language, but only **2** of them
  contain English rows; those 6 English rows are kept by the rule (the rule itself is unchanged).
- IB pool: `4b-LHA`'s IB block without `sentfin` (dropped 4,917 rows / 528,506 tokens): 30,386 rows / 6,821,358 tokens,
  14 families.

## TRAIN files (`m17/data/4b/<ARM>/`)

| Arm | `train.jsonl` SHA-256 | Rows | Tokens (− T) | IB rows / tokens (share) | English removed: rows / tokens (share of English tokens) | Updates (64 rows) |
| --- | --- | ---: | ---: | --- | --- | ---: |
| `4b-LHS10SD` | `72fa2d844fbf94be890858b9b66af0e26e12a62011929bf1eb0de1ebe93ef025` | 66,004 | 29,404,603 (+64) | 13,093 / 2,939,701 (.1000) | 5,828 / 2,939,637 (.171) | 1,032 |
| `4b-LHS17SD` | `14bce13ce926b354e214581e7cf4718d03f80c975b36dbce6731fbc6517e25a0` | 71,088 | 29,404,607 (+68) | 22,250 / 4,998,155 (.1700) | 9,901 / 4,998,087 (.290) | 1,111 |

- LH itself: 58,739 rows, 918 updates. Multilingual token share .4148 / .4152 (LH .4142): no non-English row removed.
- Removed by type (rows / tokens): 10% choice 2,653 / 1.108M, Noul 2,183 / 1.252M, Score 992 / .580M; 17% choice
  4,534 / 1.889M, Noul 3,722 / 2.130M, Score 1,645 / .979M. Stratified pass then fill: 31 / 62 fill units; shortfall
  64 / 68 tokens.
- IB by family (tokens, 10% / 17%): `fc_rel` 514,828 / 875,246; `hover` 469,345 / 797,954; `args` 425,826 / 723,869;
  `snips_rel` 266,010 / 452,366; `gsm2` 259,514 / 441,294; `poem` 214,009 / 363,932; `fc_ready` 199,310 / 338,791;
  `argq` 186,321 / 316,742; `snips_sel` 171,085 / 290,923; `w2c` 65,297 / 111,012; `ytspam` 59,959 / 101,977; `sms`
  44,309 / 75,454; `isarc` 36,689 / 62,405; `copa` 27,199 / 46,190.
- `train.ids.jsonl`: `9bea6e81…` / `deaf3f19…`; `report.json` `2d6c7770…`.

## Self-distillation targets (`teacher-s.jsonl`: M13's target lines of the kept LH rows)

| Arm | Rows | SHA-256 |
| --- | ---: | --- |
| `4b-LHS10SD` | 52,911 | `cc01d1ebf2f5b23e77967406a40eea1adac3ff1392e86c25f51e220d1045d654` |
| `4b-LHS17SD` | 48,838 | `374f4fa68c8ae32f7b85fc8beeb988a1e4de43be544d4cfbf68193b2fe2fcea2` |

## READY

`m17/data/READY-m17.json` (`arms` = the two TRAIN SHA-256 above, `teachers` = the two teacher SHA-256, this record's
name) is written on node F after this record is pushed; every seed re-hashes its TRAIN and teacher against it.
