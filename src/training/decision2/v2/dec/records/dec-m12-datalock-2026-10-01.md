# Decoder M12 — data lock (2026-10-01)

Prereg [`dec-m12-prereg-2026-10-01.md`](dec-m12-prereg-2026-10-01.md) and code `1f7fc94be` (`ops/m12/m12_data.py`,
`m12-prep.sh`). Built at 11:03Z on node E and node F in the decoder image (CPU, `--network none`); the two builds are
**byte-identical** (every SHA-256 below, and each tier's `report.json`, on both nodes). Written before any M12 GPU job.

## Inputs (hash-checked on both nodes)

IB1-r3 `1e1b08f3…`, IB2 `ee137efa…`, 4B released TRAIN `c385406e…` (58,739 rows), 4B own-Lux teacher `e2ff27ce…`, 2B
released TRAIN `1527b38b…` (56,141 rows), 2B own-Sol teacher `3f92e8c1…`, 0.8B released TRAIN `f9f3c022…` (162,696
rows). Tokenizers: each tier's base (`Qwen3.5-4B-Base@1001bb4d`, `-2B-Base@b1485b2f`, `-0.8B-Base@dc7cdfe2`); the
IB pool counts the same under all three (10,173,355 tokens; transfer families 7,504,930). Longest row 7,754 tokens.

## TRAIN files (`m12/data/<tier>/<ARM>/train.jsonl`)

| Arm | `train.jsonl` SHA-256 | Rows | Tokens | T (released) | IB rows / tokens | IB vs T |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `4b-LHA` | `e9d30c8f89ce215ff4e1b5f57e5eafd194c1822d1108c63ba5c52e7d119fd810` | 94,042 | 36,754,403 | 29,404,539 | 35,303 / 7,349,864 | +25.0% |
| `4b-LHA10` | `d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5` | 72,847 | 32,344,142 | 29,404,539 | 14,108 / 2,939,603 | +10.0% |
| `4b-LHAx` (optional) | `4af218307fd614e2c7efdd1487c7bc44b671bd16cfca9a57568514ec4798d40c` | 95,847 | 36,754,993 | 29,404,539 | 37,108 / 7,350,454 | +25.0% |
| `4b-LHA10x` (optional) | `89042192cab744daddf24db09cff3e0ab5bc4afc7b4dd9f1d94353d5e07810bc` | 73,594 | 32,344,343 | 29,404,539 | 14,855 / 2,939,804 | +10.0% |
| `2b-RA` | `08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592` | 91,146 | 36,451,446 | 29,162,107 | 35,005 / 7,289,339 | +25.0% |
| `08b-RA` | `12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3` | 309,225 | 168,764,939 | 138,244,874 | 3 × 48,843 / 3 × 10,173,355 | +22.1% |

- Every arm starts with its tier's released TRAIN, byte for byte (rows and tokens equal T); the IB rows follow.
- `4b-LHA`'s IB rows are exactly M11 stage 2's `4b-LHB` IB sample (same seed, same quota: 35,303 rows / 7,349,864
  tokens), and `4b-LHAx`'s are `4b-LHBx`'s; M11 traded those tokens against the released rows, M12 adds them. This
  makes `4b-LHA` vs M11's `4b-LHB` a clean additive-vs-displacing contrast (report only).
- `08b-RA` holds the whole pool three times; copies 2 and 3 carry ids suffixed `~c2` / `~c3` (309,225 unique ids).
- Reports (`report.json`, SHA-256 prefix): 4B `adbb8ebaa9f76b53`, 2B `2410918eac24879b`, 0.8B `c18ce5cc1cb6ce5e`;
  `train.ids.jsonl` prefixes: `4b-LHA` `c7193bd82c56ba6d`, `4b-LHA10` `e429ce9b7506e3e3`, `4b-LHAx` `11353a6da2e1aaf4`,
  `4b-LHA10x` `17bf9d3fd51a03cc`, `2b-RA` `22854cfca61cb61f`, `08b-RA` `5b83116183c9cecd`.

## Retention-probe overlap (prereg rule; node E, 13-grams)

| Source | Probe-panel items hit |
| --- | --- |
| IB1-r3 TRAIN (whole) | 0 |
| IB2 TRAIN (whole; `gsm2` is GSM8K train) | 886 (all GSM8K) |
| 4B released TRAIN | 0 (823 candidates hit, none in the panel: M10 built the panel around them) |
| 2B released TRAIN (M11 `hits-2b.json`) | 100 (GSM8K) |
| 0.8B released TRAIN (M11 `hits-08b.json`) | 0 |

Tier hit lists (`m12/probes/hits-<tier>.json`): 4B **886** (`0083e8b9…`), 2B **900** (`1e77745e…`), 0.8B **886**
(`31b366ab…`); all GSM8K. The tier golds therefore keep 2,203 (4B, 0.8B) and 2,189 (2B) of 3,089 items: MMLU 1,265 and
ARC 824 whole, **GSM8K only 114 (100 at 2B) of 1,000** (disclosed: the GSM8K part of the retention macro is thin in
M12, and M12 retention numbers are not comparable with M10's or M11's).

## READY

`m12/data/READY-m12.json` (`arms` = the six `train.jsonl` SHA-256 above; `teachers` = 4b `e2ff27ce…`, 2b `3f92e8c1…`;
this record's name) is written on both nodes after this record is pushed; every seed re-hashes its TRAIN and its tier's
teacher against it.
