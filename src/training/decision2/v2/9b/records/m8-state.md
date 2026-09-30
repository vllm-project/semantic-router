# 9B M8 state (resume file)

Updated: 2026-09-30 20:15 UTC+8 (12:15Z) — **M8 is finished: no finalist, so no formal run and no successor.**
Result `records/lux9b-m8-result-2026-09-30.md`. Nothing needs relaunching. Branch
`xunzhuo/decision-2-training-9b-m8` (worktree `vllm-sr-dev2-9b-m8`), merged into `xunzhuo/decision-2-training`.
Gist file `05b-decision-2-9b-m8.md`.

- Prereg `records/lux9b-m8-prereg-2026-09-30.md` (`0d5534325`) + amendment 1 (`2641b484e`: node A GPU3–4 only,
  KDX dropped), both before any GPU job. Mirror used for every step: `2641b484ec0526a996aa49dbfca2b953e40c1f40`.

## Outcome (node A `/data/dev2/runs/9b/m8/`)

| Item | State |
| --- | --- |
| Teacher parity / targets | PASS (80 / 80, max \|Δp\| 0.0); 12,443 rows; D1 teacher `cd531c3e…`, D2 teacher `b7985a44…` (build manifest `8455dbb0…`) |
| Recipe checks vs M7's C | PASS for both zero-step preflights and all six D continuations |
| D2 | stopped at the early rule (`rules/early-D2.json` `9f354342…`): RP 245 < 286 − 4 |
| D1 | continued (`rules/early-D1.json` `5e3ecd26…`); members 1–5 done; KD1 line read |
| Rules (`rules/rules-lines/`) | `alpha-KD1.json` `6209c493…`: no eligible α (PN1 clean gold-no above K-a13 at every α; ⅔ also HT-DEV v2 FLAG); `finalists.json` `2a150bbf…`: none; `alpha-C5.report.json` `e2436bae…` (control: none) |
| GPU-hours | 3.21 of 24 |
| Leases | `owner.9b-m8` released on GPU3–4 at 12:06Z; ~27B owner entries untouched |

## If someone revisits

- Everything stays on node A: the A20r predictions (`teacher-*`), the teacher files, D1 / D2 members, `KD1-*`.
- The KD1 line reads like K5-a12 at ½. A new preregistration would be needed for anything else (α < ⅓, a
  yes-bias-balanced Noul block on the D1 recipe, HR2).
- `m8/screens.sh` needs a lock before two arms share a control's screens concurrently (the 10:38Z incident).
