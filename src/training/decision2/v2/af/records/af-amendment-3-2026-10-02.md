# Arm factory — amendment 3: the comparison base moves to the current releases (2026-10-02 ≈15:55Z)

Written before any arm-factory Index result exists. COORDINATION 23:50 UTC+8: Nox-4B `d55528d1` = M17
`4b-SDMLxALL` and Lux-9B `f3122c7c` = M10 `KIB4-a40` are released; the 9B successor gate is vs `M10-KIB4-a40-bf16`,
and a fresh 9B owner (M10 continuation) receives the factory's 9B candidates.

- **References for every factory family delta and paired bootstrap** (the owners' successor gates):
  - 4B: `DEV2.0-4B-SDMLxALL-bf16` (merged results SHA-256 `54389c5a…`), instead of `IS-4b-LHA10SDML-bf16`;
  - 9B: `M10-KIB4-a40-bf16` (merged results SHA-256 `4d564435…`), instead of `K-a13IB-bf16`.
  Copies on node A / F are made from node C's runs with the results hash pinned (`af-ix.sh ref`).
- The Index value of each candidate does not depend on the reference; only the deltas and bootstraps do. Candidates,
  members, order and budget are unchanged (amendment 2).
- **Hand-off:** 4B → M17 (7cee4275, Nox publisher); 9B → the new Lux-9B owner (M10 continuation).
