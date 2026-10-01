# Decoder M15 — data lock (2026-10-01)

Built 16:42Z by `ops/m15/m15-prep.sh` (mirror `6abb5358e`) on node E and node F independently, CPU only, seed
20261002; every TRAIN and teacher file is **byte-identical across the two nodes**. Inputs were hash-checked against
M12's / M13's locks (M12 arm TRAIN `4b-LHA10` `d41cdd1a…`, `2b-RA` `08140409…`, `08b-RA` `12bd63d8…`; M13 SD targets
4B `7639fab1…`, 2B `2b9858d8…`, 0.8B `18827abc…`, copied to both nodes by node A and verified; MLX-DEV panel `100ae4e7…`
/ index `6ffa4b84…`, relayed from node B and verified).

## TRAIN (multilingual = non-`en` rows; tokens from M12's per-row counts, each tier's tokenizer)

| Arm | Rows | Tokens | IB dose (I / T) | IB rows by copy level | ML copies `~m2`: rows / groups / tokens | Built ML share (released s) | TRAIN SHA-256 | Teacher rows / SHA-256 |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `4b-LHA10SDML` | 77,099 | 34,393,331 | .1000 | 14,108 | 4,252 / 2,802 / 2,049,189 | .41418 (.41423) | `fef6b036…` | 62,991 / `b95c5e63…` |
| `2b-RASDML` | 102,402 | 42,457,770 | .2500 | 35,005 | 11,256 / 6,387 / 6,006,324 | .45466 (.45468) | `97157068…` | 67,397 / `69d309b2…` |
| `2b-RA10SDML` | 74,721 | 34,476,192 | .1000 | 13,991 | 4,589 / 2,598 / 2,398,709 | .45464 (.45468) | `164cb06d…` | 60,730 / `8df53abe…` |
| `08b-RASDML` | 332,778 | 191,766,202 | .2208 | 48,843 × 3 | 23,553 / 22,422 / 23,001,263 | .43269 (.43270) | `4a69853e…` | 186,249 / `a639c382…` |
| `08b-RA10SDML` | 239,706 | 162,485,151 | .1000 | 48,843 + 17,535 (`~c2`) | 10,632 / 10,134 / 10,416,622 | .43269 (.43270) | `bc84da49…` | 173,328 / `f7f79ce1…` |

- The full-dose arms keep their M12 arm file whole; the copies are by language in proportion to the released
  multilingual tokens (largest: zh — 4B 1.07M, 2B 3.63M, 0.8B 22.5M tokens; then ko / ja / es / de). Every copy is a
  typed (released) row, so every copy has an SD target (teacher rows = released rows + copies).
- The 10% arms: `2b-RA10SDML` keeps 13,991 of `2b-RA`'s 35,005 IB rows (whole groups, stratified by family);
  `08b-RA10SDML` keeps the pool once plus 17,535 `~c2` rows.

## MLX-DEV-M15 panels (`ops/m15/m15_mlxpanel.py`)

| Tier | Rows / groups | Dropped groups (id / input hash / group id) | Index SHA-256 |
| --- | --- | --- | --- |
| 4B | 9,386 / 3,802 (the whole panel) | 0 | `6ffa4b84…` |
| 0.8B | 9,386 / 3,802 (the whole panel) | 0 | `6ffa4b84…` |
| 2B | 7,742 / 3,395 | 407 (407 / 407 / 407: the 2B released TRAIN holds MLX-DEV `h5` groups) | `99283f94…` |

2B per cell (kept / before): `h5-noul` 4,539 / 6,077, `h5-choice` 295 / 401; every other cell whole (`h8-miracl-noul`
801, `h8-choice` 400, `a7q` 607, `a7k` 400, `a7s` 400, `h8-score` 300).

`m15/data/READY-m15.json` (arms → TRAIN hash, teachers → teacher hash) is written on both nodes after this record is
pushed; every seed re-hashes against it.
