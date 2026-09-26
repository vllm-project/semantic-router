# Three-level Score SELECT r2: frozen gold-blind review candidate

**Status: frozen, unreviewed, blocked from model inference and checkpoint
selection.** R1 remains `HOLD_R1_MATERIAL_GROUP_SHORTCUT` and its files
and seals were not changed. The prospective r2 amendment was signed
before this construction: SHA-256
`96cd806cd2bfcfba8c56dc750fb225b044b44f5873bb0bf394e413d7bdfeed61`,
source commit `3ac61f5f0`. The exact r2 builder commit is `9fb160e31`.
Fresh private 32-byte author seed and independent HMAC review salt were
generated for this version; their bytes remain outside source control.

## Mechanical admission

| Check | Frozen result |
| --- | ---: |
| Independent source groups / related native Score rows | 80 / 240 |
| Operations | four, with 20 groups each |
| Language per operation | 16 English + 4 Chinese groups |
| Labels per operation | 20 at each of 0, 1 and 2 |
| Gold-free reviewer packet | 80 opaque groups / 240 rows |
| Protected references | 30 roles / 43,685 rows |
| Exact ID, group, input, raw and normalized state matches | 0 |
| Bounded near-state / full-prompt matches | 0 / 0 |
| Maximum native Qwen3.8-27B token length / cap | 273 / 1,024 |
| Best one-field group-held-out per operation | 40 / 60; cap 40 / 60 |
| Independent structured / visible-text oracle disagreements | 0 / 0 |

The 30 protected roles include the frozen parent TRAIN/SELECT/CAL, all
three earlier serialized Score curriculum attempts, the entire v6 TRAIN
candidate, all 240 r1 gold-free prompts, and current available gold-free
DEV, CSS, authored, multilingual and public rosters. The v6 TRAIN file is
bound to SHA-256
`6aee966cc5499a87d2a77241676586c9c9f801b3c662a078daf025f001169f54`;
the r1 reviewer packet is bound to
`d38702cee5ef1b50458a4ee11d4370a7fda44013321b2dac43b40d01600ea88a`.
The protected-inventory SHA-256 is
`21f7249fac3e31518221772ab10a66991549b1f05d95c7881cc476e0a00df2ce`.
The overlap screen is approximate and cannot establish semantic
independence; r2 retains the same four abstract operations as r1.

The allocation ladder holds each pool to exactly two raw margins and two
margin signs per group; the first-changing pool is balanced 8/8 in
English and 2/2 in Chinese. Inclusive coverage has the same first-change
balance. Quorum has exactly three signed reports at every level, with
0/2/2 qualifying reports and 0/1/2 independent qualifying origins.
Waiver and all other single-field and one-subrecord projections take no
more than two values in a triplet. The group-held-out cap of two-thirds
holds for every predeclared shallow feature. Three focused tests passed
in the pinned CPU container, including a synthetic recreation of r1's
three-margin defect that the new gate rejects. No model or GPU ran.

## Immutable custody and review handoff

The author casebook, target key and gold-free reviewer packet are in
separate private directories. Only the latter is for the reviewer. The
source builder, structured oracle and visible-text oracle SHA-256 values
are respectively:

- `23e9c055a802a7b90f7b31245f193af370c8cb1158ec48d297b2138b0b0ff7a7`
- `1f7dce517eb1717f131810e54b987738157a00d23cff9ba09fb7ebfc2fac9bb4`
- `10d10ae3dd942e0b4703ec37915e108ce0c13d7f9ba99f159e6cf51dd0fea5e6`

The private freeze manifest SHA-256 is
`2bb632616f42e170438980b30ca05634b782e34ba0f7c072e90b110b5ba0fefd`;
the author casebook SHA-256 is
`2c90ef89661e47d2fdae3effd821ab35b7aad43c66d59f2ab4e55eb2fb42743d`;
the private oracle key SHA-256 is
`8e9d1232c1d0db75d7fc1e583d07753e5db6716c85aca39040744a2ec834eda1`.
The reviewer packet SHA-256 is
`091e3023b84d64131a72b23b90b3eacf837027ed23d58045de801e92a331f683`;
its reviewer manifest SHA-256 is
`ee3045f852478b743d24af9c076590b4cbec0bc60999e6dd8b44f308e55fbf9b`.
The packet has only HMAC-opaque row/group aliases, operation, language,
state, instructions and options; no source ID or answer field.

A fresh independent gold-blind reviewer must solve every row, inspect
complete triplets, and seal judgments, ambiguity, naturalness, language
fidelity and shortcut findings before any key comparison. One material
finding blocks r2. Qualified bilingual review is still required for a
Chinese-transfer claim. This panel is only for checkpoint selection and
cannot support a JevArena release score by itself.
