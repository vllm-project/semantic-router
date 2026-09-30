# Decoder Milestone 7 — preregistration (round-2 data for DEV2.0-4B, then DEV2.0-2B; 2026-09-30)

Written 2026-09-30 ≈03:20 UTC+8, before any M7 GPU job, after the M6b result
([`dec-m6b-results-2026-09-30.md`](dec-m6b-results-2026-09-30.md): no successor). Assignment: coordinator note
2026-09-30 01:50, decision 2. Development readouts are never release scores; v3 / public 231 / C1 are post-key
same-panel comparisons; `hs1-dev`, PN1 dev and `mlx-diag` are diagnostics. A data lock with every SHA-256 follows
before any training launch. Nothing goes to HF; C1 is never opened by this track.

## Goal and bars

Per tier, a successor that passes successor-rule items 1–8 against the current revision, with the gain coming from
**human transfer and generalization** (CSS15 H, public-231 hard, `mlx-diag`, the C1 guard), not typed gains alone:
M6 / M6b showed that development typed gains reverse on typed FINAL (4B N6D +.025 → −.041; 2B +.013 → +.003).
Priority 4B, then 2B.

| Tier | Current revision | Weights | Scored T = 1 run (bar) | v3 | Best peer | C1 baseline |
| --- | --- | --- | --- | ---: | --- | ---: |
| 4B | `DEV2.0-4B@452f1332` | N4XF soup `11b5ca1c…` | `runs/release/dev2-4b-t1-derived` | 63.151 | Decider 4B 61.882 | 48.38 |
| 2B | `DEV2.0-2B@5ad3e9a3` | S2T soup `073bd1f2…` | `runs/release/dev2-2b-t1-derived` | 53.437 | Decider 2B 49.499 | 45.70 |

## Evidence behind the design

1. **Hard skills.** The JevBench analysis traced DEV2.0-4B's gap to Decider 4B (−21 items) to quoted-conclusion
   verification, long policy packets with amendments, and "yes" when a condition is not met. HS1 builds exactly these
   (F1 / F2 / F3); on `hs1-dev` DEV2.0-4B adopts the quoted answer .775 of the time (Decider .641; ideal .50) and
   says yes to unmet conditions .262 (Decider .128).
2. **Long inputs.** Our fine-tunes moved long-document / multi-hop items down (JevBench note 4); C1 is 19% long.
3. **Multilingual Noul yes-bias.** N4XF says yes .762 on PAWS-X Noul (Nox 1.0 .663); PN1's dev slice sees the
   same gap (.708 vs .653).
4. **Dose.** M6b: more of the same XL recipe (2×) lost typed FINAL Noul / Choice at level human transfer. A
   matched-token control therefore separates "new data" from "more tokens".

## Arms and data

Common to every arm: full fine-tuning from own 1.0, one epoch, CE + 0.5 Brier + KL with the released recipe's
teacher and weight, backbone / head LR 5e-6 / 5e-5, max length 8,192, token batching ≤ 32,768 tokens / 64 rows,
`even8` checkpoints with SELECT700 `matrix-v1` selection per seed, `--teacher-partial` (gold-only rows below),
`drive_arm.sh` preflights (zero-step, one-step, `preflight_dec` load parity + one-step reload), seeds 20260926 /
27 / 28 (the same triple in every arm, so arm contrasts share seeds), and the uniform 3-seed FP32 soup as the arm
artifact. SELECT / CAL: `m3/data-sel700-cal698` (SELECT700 `32a4352d…`, CAL698 `19cc1a8c…`) on both nodes.

**4B** — start Nox 1.0 `cde2a68d`; own-Lux 1.0 KL 1.0 exactly as N4XF (N4XF's composed targets `e2ff27ce…` on the
base, `lux-all-59m` `a1bafad5…` on added recipe rows; H7 / H8 rows gold-only, as in N4XF). Base = N4XF's mixture
`m4-xl-full-29m` (`c7d51219…`) minus the quarantined 4B near-match group (3 HotpotQA rows, 17:15 note).

| Arm | Mixture |
| --- | --- |
| **N7H** | base + HS1 block + LP block |
| **N7P** | base + PN1 block + filler of (&#124;HS1&#124; + &#124;LP&#124; − &#124;PN1&#124;) tokens |
| **N7C** (matched-token control) | base + filler of (&#124;HS1&#124; + &#124;LP&#124;) tokens |

4B filler: whole groups of `m6-xl-full-59m` (`160812e2…`) not in the base — the released XL r2 recipe's next rows.

**2B** — start Sol 1.0 `ce0c018a`; own-Sol 1.0 KL 0.5 on every recipe row exactly as S2T (S2T's targets
`947bc65b…` on the base, `sol-59m` `53e4adc8…` on LP rows). Base = `m4-v2m-ret-r2` (`1527b38b…`: S2T's recipe
minus the 57 r2-exposed rows, so a successor's new training file has an empty exposure receipt).

| Arm | Mixture |
| --- | --- |
| **S7H** | base + HS1 block + LP block (LP rows not in the 2B base) |
| **S7P** | base + PN1 block + replay of (&#124;HS1&#124; + &#124;LP&#124; − &#124;PN1&#124;) tokens |
| **S7C** (matched-token control) | base + replay of (&#124;HS1&#124; + &#124;LP&#124;) tokens |

2B filler: whole groups of the base repeated once (the recipe has no remainder; the copy's id is suffixed `~r2`
and carries the base row's own-Sol target).

**Blocks** (built by `ops/m7/m7_compose.py` in one deterministic pass per tier; whole groups; sampled blocks use the
builder's stratified sampler `build_template_s.select_groups`, strata source × task type × language):

- **HS1** — `m4/hs1/train.jsonl@171e6f0c` (`c90ef316…`): F2 (`hs1_policy_packet`) and F3 (`hs1_unmet_condition`)
  whole plus half of F1 (`hs1_quote_check`) by tokens (sampler seed `dec-m7-hs1-half-v1`) — the coordinator's
  "F2 + F3 + ½ F1" dose. The 75 groups (150 rows) of the confirmed F2 rendering defect (data fix `05a0d391d`) are
  dropped. Gold-only. By the HS1 record's counts this is ≈ 11.2M native tokens (its "about 9M" under-adds the
  three families); the lock records the exact count.
- **LP (long prose)** — groups of `m6-xl-full-59m` not in the tier's base whose longest state exceeds 3,000
  characters, sampled to 3.0M tokens (seed `dec-m7-lp-v1`). They are human-labeled multi-hop / long-document rows
  (MuSiQue, HotpotQA, HoVer, NQ, 2Wiki, MIRACL, TyDi QA, …) that the recipe already admits, with the tier's own
  teacher (4B H7 / H8 rows gold-only).
- **PN1** — `m4/pn1/arms/pn1.train.jsonl@5ad36287` (`f6f6a531…`, 4,888 rows ≈ 0.55M tokens) ×3; copies 2–3 carry
  ids suffixed `~r2` / `~r3`. Gold-only.
- **Filler / replay** — sampler seed `dec-m7-fill-<tier>-v1` / `dec-m7-replay-<tier>-v1`; its budget is bisected
  until the arm's tokens are closest to the H arm's, so P's filler is a per-stratum prefix of C's. Lock check:
  |C − H| / H and |P − H| / H ≤ 0.5%.

**PN1 revision rule (fixed now).** The data track reported that PN1 `@5ad36287` fails its §3 blind label review
(12 / 224 gold errors, all false "no", in `pn-near` es / fr / ar / ru / ko and Russian `pn-name`) and is
certifying a PN1-r2 rebuild (data amendment `46702e2fc`). If PN1-r2 passes that certification and is published
with hashes before a tier's first P seed starts, that tier's P mixture is rebuilt with PN1-r2 ×3 by the same
composer, seeds and matching rule and locked in a lock part before launch. Otherwise the P arm uses `@5ad36287`
and is **research-only**: it can be a finalist and is reported, but is never handed off as a successor.

**Release safety carried into any hand-off:** an HS1-trained model is release-safe only after the data track closes
the HS1 §4 spot check (template fix and republish); a PN1-trained model only after PN1's audits (embedding scan,
H7 / H8 isolation, blind review) close on the revision it trained on.

**Data lock** (before any training): `ops/m7/m7-prep.sh` (node B, CPU) builds the six TRAIN files, one composed
teacher per file (`v2.dec.compose_teacher`, gold-only pools allowed), an exposure receipt per file (r2 payload
`2194716a…`), and `ops/m7/m7_lock.py` checks hashes, unique ids, quarantine and defect absence, token matching, empty
exposure, teacher coverage outside the gold-only pools and the C1 registry source guard. The lock record with every
SHA-256 is committed, then READY files release the chains. The 2B files are relayed to node A hash-checked.

## Candidates and development readouts

Lines anchored on the incumbent `I`: β = 1 (the arm soup), ½ (`[I, A]`) and ⅓ (`[I, I, A]`), built with
`v2.dec.soup`. Typed DEV (1,600) + CSS pilot (1,430) at **16,384 tokens** via `v2.dec.infer_dec`, then
`v2.dec.dev_readout` with paired comparisons against `I`, on the tier's formal kernel path:

- 4B on node B (`dbe5f32b`). `I` = M6's 16K node-B readout of the same weights (`files.sha256` `df602ca9…`),
  reused after a hash check.
- 2B on node A (`f83b1d10`). `I` is read there from a copy of the S2T soup relayed from node B and checked against
  its per-file list (`f2ea6ddf…`).

## Development gates and finalists (fixed now; `ops/m7/m7_rules.py`)

A point passes when the 9B rule module's `eligibility` (`v2/9b/lux9b/m4_rules.py`, unchanged) accepts it against `I`:

1. every type keeps c_t ≥ c_t,I − 0.03·n_t;
2. **CSS-pilot three-task mean H3 ≥ H3_I** (the human-transfer non-decrease gate);
3. no typed-DEV family below F_f,I − 0.10.

There is no typed-gain requirement. A line's pick is its largest passing step (β 1, then ½, then ⅓); a pick ≥ 8 P
below the tier's best pick is dropped. Slots: 1 = the H line, 2 = the P line, 3 = the C line; a line without a pick
leaves its slot empty. At most three finalists per tier.

## Diagnostics (read and reported, never selected on)

- **`hs1-dev`** (2,396 rows): per-family accuracy (dev-id / dev-ood), the F1 adopt rate (ideal .50) and the F3
  false-yes rate (ideal 0), for `I`, every arm soup and every finalist, scored against `I` with
  `v2.data.hs1.validity`.
- **PN1 dev** (1,974 rows): yes-rates pooled over the six PAWS-X languages and all eight, by family, against `I`
  with `v2.data.m4.pn1_validity_score`. Its es / fr / ar / ru / ko near rows and Russian name swaps may carry
  wrong "no" labels (data amendment 1 §B); this is disclosed with every reading.
- **Data effect at matched tokens:** H − C and P − C at β = 1 on typed DEV, CSS pilot, `hs1-dev` and PN1 dev
  (development, report only).
- **`mlx-diag`** (formal only; item 4): PAWS-X Noul yes-rates per language from the aggregates.

## Formal (finalists only)

Post-key same-panel v3 (typed FINAL + CSS15) and public 231 at 16,384 tokens with the frozen runner, gold-free
seal, report and paired CIs on node A; `mlx-diag`; a 16K CAL698 fit with the 23:15 rule; an 8-item smoke first. The
M6 formal scripts are reused with `M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m7`, `M6_SELECT=/data/dev2/runs/dec/m7/
select`, `M6_PREFIX=m7`:

- **4B:** node B (`dbe5f32b`, copy of `formal/m5/cache-frozen` `f6d0f920…`), relayed, scored on node A;
  `mlx-diag` on `cache-frozen-mlx` vs the node-B N4XF reference (the M6b procedure).
- **2B:** node A GPU5 (`f83b1d10`, `HIP_FORCE_DEV_KERNARG=1`, copy of S2T's persisted node-A cache — DEV2.0-2B's
  own scored settings), staged on node A (`M6_2B_NODE=A M6_2B_STAGE_A=1`); `mlx-diag` vs `formal/m3/m3-S2T-soup-mlx`.

## Successor rule, choice and item 8

Items 1–8 exactly as in the M6b preregistration (bars above; item 5: 4B vs the adopted Nox 1.0, v3 ≥ 55.7, H not
significantly below Decider 4B / Jet v6.2; 2B vs Sol 1.0 16K, v3 ≥ 44.5, H not significantly below Decider 2B;
item 6(a) on the arm's new TRAIN file; points containing `I` inherit its disclosed exposure: 4B none, 2B S2T's
24 groups / 57 rows; item 7 `gates public231` gating; item 8 through the eval custodian). Among finalists passing
items 1–7 and eligible for hand-off (not research-only): the highest paired lower bound vs the best same-size
peer, then vs the bar, then slot. Only that C1 candidate goes to the custodian (frozen package on node A, a
`dev2-c1-postkey-spec/1` spec, and the content recheck its ledger needs for HS1 / PN1 / `m6-xl-full-59m` rows). A
candidate passing items 1–8 is handed to the coordinator with the release-safety conditions above.

## Budget, GPUs, stop rules

- **Cap: 29.7 GPU-h** (30 minus M6b's 0.284), counted as wall-clock × GPUs including preflights, postruns, soups,
  readouts, diagnostics, CAL fits, smokes and formal collections.

  | Attempt | Cap (GPU-h) |
  | --- | ---: |
  | N7H, N7P, N7C (4B, 3 seeds each, ≈ 43.6M tokens) | 4.5 each |
  | S7H, S7P, S7C (2B, 3 seeds each, ≈ 43.4M tokens) | 2.8 each |
  | Lines, references, diagnostics (both tiers) | 2.5 |
  | Formal, ≤ 6 finalists incl. `mlx-diag`, CAL fits, smokes | 2.5 |
  | Contingency (never for reruns) | 2.8 |

- **At 27 GPU-h cumulative (M6b + M7):** no new training seed starts; formal work finishes within the cap. The 2B P
  arm runs last and is the first to go if the budget is short.
- **Stop rules:** a failed preflight stops that arm (no rerun, no replacement seed); an arm stops at its cap; an arm
  with fewer than two finished seeds is dropped (two-seed soups are disclosed); a failed smoke, calibration or
  collection stops that finalist. Failed arms are never rerun.
- **GPUs:** node B GPU3 (N7H → N7P s1, s3) and GPU4 (N7C → N7P s2); node A GPU5 (S7H → S7C → S7P). Readouts and
  formal jobs are recorded co-tenants.
- **Chain rule:** every script runs from an exact mirror (uploaded and verified by `mirror_to_node.sh`), launched
  in a separate step; liveness by PID and container name, never `pgrep -f`; the first log line is confirmed.

## Code (committed before use)

`v2/dec/ops/m7/`: `m7_compose.py`, `m7_lock.py`, `m7-prep.sh`, `m7-chains.sh`, `m7-soup.sh`, `m7-gpuh.py`,
`m7-lines.sh`, `m7_rules.py`, `m7-relay.sh`, `specs/m7-quarantine-groups.json`; tests
`v2/dec/tests/test_m7_compose.py`, `test_m7_rules.py`. The M6 formal scripts gain `M6_PREFIX` and `M6_2B_STAGE_A`
(small, decoder-only). No shared module changes.
