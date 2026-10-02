# Decoder M18 data lock (prereg dec-m18-prereg-2026-10-02.md, "Part B")

Built on node F by `ops/m18/m18-prep.sh` from mirror `a9847b580` (decoder image `dbe5f32b`, CPU), 2026-10-02T04:59Z.
Every seed re-hashes its TRAIN, weights and teacher against these values (`m18/data/READY-m18.json`).

Inputs (each checked against its hash): M12 `2b-RA` TRAIN `08140409…` / ids `22854cfc…`; Sol's released TRAIN =
its first 56,141 lines, `1527b38b…`; M13's 2B SD targets `2b9858d8…`; the own-Sol teacher `3f92e8c1…`; IB3-r2 TRAIN
`9d92d92a…` (dataset revision `1c8452da`, `m6/ib3/ib3.train.jsonl`, 8,752 `mqa` rows).

| Arm | TRAIN | weights | teacher | rows | released rows | IB rows | released weight share |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `2b-RS17UP` | `b726d1c9d129bb892374678c244bb7b1a27b9867c42a4deb0aa8270b7414067f` | `5898cce43f7a125e200b74a2b827fe44c6c21392713ed411c3789906c73b012a` | `464b07edd0cf6bc019a002180e129543ee6af56fb336af5756675e4c80c207a0` | 68,401 | 46,347 | 22,054 | .7725 |
| `2b-RAUPM` | `a747cbc15a549e68b263a9d5d6f2f4617730c99d3b54525062f09fa03a822ede` | `799b3bf574308459b33c1ce90713ff133693df6b1a5be7e4269fa329a3216e18` | `3f92e8c108c5a9949fda4b08638907fcc80175f68efec18bece912d0edc8eea1` | 99,898 | 56,141 | 43,757 | .6743 |

- `2b-RS17UP`: 29,162,108 TRAIN tokens (T + 1); IB share .17; English-only released rows removed .3117 of their
  tokens' target; teacher = the SD targets of the 46,347 kept released rows (KL 1.0).
- `2b-RAUPM`: `2b-RA` TRAIN byte for byte, then the IB3-r2 TRAIN byte for byte (35,005 IB1 / IB2 rows + 8,752
  `mqa` rows, weight 1); teacher = the own-Sol teacher on the released rows (KL 0.5).
- Weights: released Choice / Noul 1.5, Score 2.0, IB 1.0 (`ops/m14/m14_weights.py`).

## Amendment 1 arms (node A, `ops/m18/m18-prep-a.sh` from mirror `c9616d873`, 2026-10-02T05:56Z)

Inputs checked: M14's staged M12 `2b-RA` (`08140409…`, ids `22854cfc…`) and `08b-RA` (`12bd63d8…`) TRAIN files,
the own-Sol teacher `3f92e8c1…` and IB3-r2 TRAIN `9d92d92a…` (both pulled node F -> node A).

| Arm | TRAIN | weights | teacher | rows |
| --- | --- | --- | --- | ---: |
| `2b-RAM` | `a747cbc15a549e68b263a9d5d6f2f4617730c99d3b54525062f09fa03a822ede` (= `2b-RAUPM`'s) | none | `3f92e8c108c5a9949fda4b08638907fcc80175f68efec18bece912d0edc8eea1` | 99,898 |
| `08b-RAM` | `e414cb2d2e7f492f96e674e4216ded44b52f32f029bda03f779b19ba25b250e3` | `e52dec07de0a15f9071eff3756d3baf0c31e37cbf3f2d089b7be04effbd1ab4f` | none | 317,977 |

`08b-RAM` weights: 1.0 for the 309,225 `08b-RA` rows, 3.0 for the 8,752 IB3-r2 rows.
