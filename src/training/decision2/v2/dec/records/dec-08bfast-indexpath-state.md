# 0.8B / 2B Index path — state (continuation of 1afc17e8; polled ≤ 30 min; newest first)

Assignment: COORDINATION 2026-10-02 02:05 (Index path), 04:25, 04:40 and the 09:40 watchdog note: apply the Index path
to M16's 0.8B / 2B finalists `08b-RA-a75`, `08b-RASD-a75` and `2b-RASD-a25` and release the best qualifier per tier
to `Decision-2.0-Eos-0.8B` / `Decision-2.0-Sol-2B`. **Superseded at 09:55 by the user's Index-first rule** (below).
Branch `xunzhuo/decision-2-training-dec-08bfast`. Index values stay private (node C
`/data/dev2/private/eval/index021/ix1/runs/<name>/` and `decision2-program/private/m16-indexpath/`); this file holds
verdicts, counts and hashes only.

## 2026-10-02 02:10Z — Index runs found complete; bootstraps done; rule now Index-first

- **Found finished (predecessor, node C GPU1–7, mirror `00401c9b5`, 20:22–21:04Z):** for each candidate the
  restaged package (onto DEV2.0-0.8B `bede7938` / DEV2.0-2B `a53cf66a`, FP32 checkpoint of the M16 formal package,
  files lists `bca5ec1d…` / `a57e236b…` / `b5248c80…`), the 86-request parity gate (86 / 86 ok, max |Δp| 0.0), the
  full panel-7 run (120,226 rows: 120,224 ok, 2 unsupported `max_length_exceeded`, 0 errors), dual scoring (port vs
  kit `87d4650b`) PASS:

  | Candidate | Results | IX1 identity | GPU-h |
  | --- | --- | --- | --- |
  | `M16-08b-RA-a75` | `23b90341…` | `fee82408…` | 1.28 |
  | `M16-08b-RASD-a75` | `3697e0f8…` | `690e5b97…` | 1.22 |
  | `M16-2b-RASD-a25` | `aabe4111…` | `7afafef9…` | 1.35 |

  Only `08b-RASD-a75`'s transfer-only bootstrap had run. The node C GPU1–7 owner files still named the finished 2B
  run; they were rewritten to `status=released` at 01:42Z (GPUs idle, no IX1 container).
- **Bootstraps (node C CPU, `paired_boot.py` of mirror `42960045f`, 2,000 paired replicates, seed `20261002`):** for
  each candidate vs its tier's DEV2.0 IX1 run, the full panel (no exclusion; the item-1′ evidence) and the
  transfer-only variant (HoVer, When2Call, iSarcasmEval, GSM8K, BPoMP excluded); family deltas. **All three have a
  95% Index lower bound > 0.** Files (private): `paired-boot-full-vs-ref.json` `e50577de…` / `dc2634d3…` /
  `982430b8…`, `paired-boot-vs-ref.json` (transfer) `544c3fb9…` / `0aa5acd0…` / `9611e6bd…`.
- **Item 1′ and 6(b)′ (from M16's two-bar successor files on node A):** v3 upper bound > 0 against both bars on the
  full panel (item 1′, v3 part) and with the flagged items removed (6(b)′) for all three; the rule-5 parts of 6(b)
  pass; 6 / 6 stored pairs reproduce. **All three qualify on the Index path.**
- **Rule change (COORDINATION 09:55, user decision):** the release gate is now the Index delta alone (95% lower
  bound > 0) plus integrity checks (exact package parity, Hub / `trust_remote_code`, the row-level contamination
  audit, no collapsed type); v3, human transfer, mlx-diag, public 231 and C1 are references; C1 item 8 is not run
  unless in progress (it was not); selection = the largest Index-gain lower bound. The coordinator compares M12's
  frozen `08b-RA` (IX1 run `DEV2.0-0.8B-08bRA`, results `9c28a430…`) with the M16 a75 points: its results were
  copied node D → node C through node A (hash equal) and its two bootstraps are running (node C CPU, 02:04Z).
- **Contamination audits (IX1 method, planted 200 / 200) cover every candidate:** the RASD arms' TRAIN files are hard
  links of M12's RA files (M13 data lock). 0.8B RA TRAIN `12bd63d8…`: 18 duplicate-class rows, 1 item row (the same
  public-split BANKING77 item as the DEV2.0-0.8B TRAIN); 2B RA TRAIN `08140409…`: 76 duplicate-class rows, 0 item
  rows (DEV2.0-2B TRAIN: 80 / 0).
- **Next:** pick the 0.8B winner; BF16 copies of the winners, an Index run of each on exactly the released weights
  (card Index and gate evidence), then the releases on the then-current `main` with the default card generator
  (banner A, round-3 Index conventions).
