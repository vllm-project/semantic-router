# Decoder Milestone 7 — data lock (2026-09-30)

Identities of every M7 training input, recorded and pushed before any M7 training job. The chains re-hash each
TRAIN and teacher file against a `READY` file written only after this record is committed. Preregistration:
[`dec-m7-prereg-2026-09-30.md`](dec-m7-prereg-2026-09-30.md) (`58e41dd60`).

- **Builds:** node B, CPU only, `ops/m7/m7-prep.sh` from the exact mirrors of `58e41dd60` (H, C) and `393c0b0d0`
  (the P rebuild below). Receipts sit beside every output under `/data/dev2/runs/dec/m7/`.
- **Lock check:** `ops/m7/m7_lock.py` from the mirror of `8ff0a1fca` gives **PASS** for all six training files:
  `lock-4b-HC.json` `9e2cfeb0…`, `lock-4b-P-r2.json` `fab11de0…`, `lock-2b-HC.json` `3030c090…`,
  `lock-2b-P-r2.json` `49004292…`.
- **PN1 revision rule applied.** PN1-r2 passed the data track's certification and was published before any P seed
  (`llm-semantic-router/decision-2.0-training-data@c41d65d4a5c9d2cbfde7c098d7d943838b32fd12`,
  `m4/pn1/arms/pn1.train.jsonl` `c1cec06bd5f1caadb7631338796054aae120cbf8b17dcb0bbf4c694470c52cfb`, 4,364 rows; data
  record `m4-dq-results-2026-09-30.md`). So both P arms were rebuilt with PN1-r2 ×3. The rebuild left H and C
  byte-identical (checked). The first P builds on PN1 `@5ad36287` (`a919ca9a…` 4B, `765465b1…` 2B) are never
  trained.
- **HS1 is within the cleared revision.** The data track republished HS1 with the template fix
  (`@27b1d2f130292268b43a618584bebab5d4e4a6b5`, `m4/hs1/train.jsonl` `0dfaa6eb…`); exactly the 150 defect rows
  changed. M7 drops those 75 groups, and the lock check confirms that all 16,230 HS1 rows of each H arm equal the
  rows of the same id in the cleared file. Data record: "PN1-r2 and HS1 — cleared for released models at the
  revisions below".

## Inputs

| Input | SHA-256 |
| --- | --- |
| 4B base `m4-xl-full-29m` | `c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60` |
| 2B base `m4-v2m-ret-r2` | `1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c` |
| Pool (LP, 4B filler) `m6-xl-full-59m` | `160812e2c3d39bd09fcbe6c98e2b4b1633f6680bd435241d3a83e791d5dfe33b` |
| XL r2 recipe ids (gold-only pools) `mx-xl-full-r2.ids.jsonl` | `7843afb7b2bbb315902b6748d6559532f384283d8efd836b935daac5f55dbd4a` |
| HS1 TRAIN `@171e6f0c` | `c90ef3164d90d3fd6a1ab0397529faec172760b1756de7d4d8cbf2677a131a71` |
| PN1-r2 TRAIN `@c41d65d4` | `c1cec06bd5f1caadb7631338796054aae120cbf8b17dcb0bbf4c694470c52cfb` |
| 4B teachers: N4XF's `e2ff27ce…` (base); `lux-all-59m` (added rows) | `a1bafad59411901cd8a13451b741a186fece445ab78ce9b0466fc5adc7f1f793` |
| 2B teachers: S2T's own-Sol (base, replay); `sol-59m` (LP) | `947bc65b15f881aafaae4e3b0fda0885a78f752451dc1187663854d03d0ae4d4`; `53e4adc80d3833fb7a690259660cb5d598b4f4d4640bb9a2e5fda18a8a8e0115` |
| Quarantine (`ops/m7/specs/m7-quarantine-groups.json`) | 1 group, 3 rows dropped from the 4B base and the pool |
| Tokenizer | Eos / Sol / Nox tokenizer (`363c4a5e` snapshot), native `encode` lengths |

## TRAIN files (all six: 0 exposed groups vs the r2 payload `2194716a…`; no C1 registry source; family-name hit `pilot_narrative_reading` reported only, as in M6)

| Arm | TRAIN SHA-256 | Rows | Native tokens | Teacher SHA-256 (covered; gold-only rows) |
| --- | --- | ---: | ---: | --- |
| N7H | `343f9351806b99189db2ad4b7676a637666198d9e321332efae3bfd7c9aeb304` | 76,413 | 43,508,298 | `3c2be9d3532d5116f96b96ed6353c6e38394aff6fab011777104c877f20ad2b7` (56,010; H7 / H8 base 3,759, HS1 16,230, LP H7 / H8 414) |
| N7C | `4aae0a125ad3e4b0bd21d2894bc3997b67fa418dfd76d281f2dd48cbc24667c0` | 87,725 | 43,507,920 | `c2f4c31c944ec0375b6bd72764c77ca7d6668f7a2cd5ef4cc8b5b084144516f5` (81,917; H7 / H8 3,759 + 2,049) |
| N7P | `e83eb4f2f19479d69eafeb079dff3d74e23f2d5a6462cca825ab525297b63045` | 97,755 | 43,510,839 | `7709faa5e2a2aafa5290d3f60cd4bd63a90d4f67dec307b2bf0b4e36d62d2091` (79,072; H7 / H8 3,759 + 1,832, PN1 13,092) |
| S7H | `e789ee41e239c1ce8c72719b6088d1180572d50bc4c4b7b575c49d6929869667` | 73,757 | 43,292,147 | `6da7550da68efa2c3144d905fd413229f48ca9d91f868caf0953a813c10c285a` (57,527; HS1 16,230) |
| S7C | `260b3811eb590aba616d4a0a2977dcac45d6bd1c97da99ac711f38fe29810cab` | 83,278 | 43,292,097 | `4fc9567f5cbb424eedb792c54be3db73a4c77f29718d7062ca312fe827096e23` (83,278; none) |
| S7P | `5c88369224a8548d7b3116b59a9c944deb53278f46c3cf52b2d009b0b8176b23` | 93,624 | 43,292,614 | `97cea7d614aaab6a275af5c66c6a073d304d451d06da327f77cff6f1b3dc1717` (80,532; PN1 13,092) |

Token matching: 4B C − H = −378, P − H = +2,541 (≤ 0.006%); 2B C − H = −50, P − H = +467. P's filler is a
per-stratum prefix of C's (nested).

## Blocks

| Block | 4B | 2B |
| --- | --- | --- |
| Base | 58,739 rows, 29.40M tokens (3 quarantined rows dropped) | 56,141 rows, 29.16M tokens |
| HS1 | 16,230 rows, 10.98M tokens: F2 2,248 (after the 75-group / 150-row defect drop), F3 10,392, ½ F1 3,590 (1,795 of 3,589 groups, 2.80M of 5.59M tokens); C / N / S 4,566 / 9,732 / 1,932 | same rows |
| LP | 1,444 rows, 3.12M tokens, 724 of 1,966 eligible groups; C / N / S 156 / 853 / 435; en 1,056, zh 117, es 92, fr 29, hi 28, fa 28, ru 23 | 1,386 rows, 3.15M tokens, 696 of 3,534 groups; en 987, zh 122, es 76 |
| PN1-r2 ×3 | 13,092 rows (4,364 unique), 1.47M tokens: ja 6,708, zh 2,844, de 2,466, ru 444, ar 324, ko 306 | same rows |
| Filler | C 28,986 rows / 14.10M tokens; P 25,924 / 12.63M (the recipe's next XL r2 rows) | replay: C 27,137 / 14.13M; P 24,391 / 12.66M |

- **LP sources (4B):** A7g generated long states (`dec10:generated_stage4_v2`) 237, MuSiQue 202, HotpotQA 194,
  OASST1 167, HoVer 134, MIRACL 108, NQ 94, and others. The preregistered rule selects by state length, so about a
  sixth of LP is long programmatic A7g states rather than natural prose; disclosed.
- **Teacher agreement with gold on the added rows** (argmax): 4B own-Lux on LP Choice .635 / Noul .759 / Score .423
  (base .838 / .803 / .527); 2B own-Sol on LP .660 / .633 / .372 (base .721 / .726 / .491). Long rows are harder
  for the teachers too.
- **2B note:** S7C's replay rows are second copies of base rows (ids suffixed `~r2`) with the base row's own-Sol
  target.

## Release safety at lock time

Both round-2 datasets are cleared for released models at the revisions used here (HS1 rows ⊂ `@27b1d2f1`; PN1-r2
`@c41d65d4`). The C1 content recheck the custodian's ledger asks for HS1 / PN1 / `m6-xl-full-59m` rows still
applies before any item-8 collection.
