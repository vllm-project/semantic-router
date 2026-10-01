# Decoder Milestone 9 — results (HR2 efficacy pilot at 4B; PILOT, NOT RELEASABLE; 2026-10-01)

Preregistration [`dec-m9-prereg-2026-10-01.md`](dec-m9-prereg-2026-10-01.md) (`f22a5797c`, before any GPU job); data
lock [`dec-m9-datalock-2026-10-01.md`](dec-m9-datalock-2026-10-01.md) (`2f20c2223`, PASS); amendments
[1](dec-m9-amendment-1-2026-10-01.md) (`e7f8e9858`: the control becomes M7's N7C) and
[2](dec-m9-amendment-2-2026-10-01.md) (`29057a4de`: H9 arm cap), both before any H9 readout. Development readouts are
never release scores. HR2 is `release_safe: false`, so nothing here is handed off, uploaded or sent to the C1
custodian. Aggregates (summary, pick, parity, GPU-hours): [`dec-m9-results-2026-10-01/`](dec-m9-results-2026-10-01/).

## Bottom line

**HR2 did not move human transfer at 4B. Under the preregistered rule it is not a lever, and no formal run was made.**

- **Primary (HT-DEV v2, α 1):** H9 soup − control soup = **−0.004 [−0.018, +0.010]**, TIE (H_dev2 .527 vs .531).
  The upper bound excludes a GAIN-sized effect (+0.02).
- **vs DEV2.0-4B:** H9 soup −0.012 [−0.025, +0.001] TIE; H9 α ½ −0.003 [−0.013, +0.008] TIE. Matched contrast at α ½:
  −0.004 [−0.014, +0.006].
- **Gates:** no H9 point passed. Both fail the typed **Noul floor** (`rule_precedence` 241 at α 1 and 258 at α ½,
  floor 260; at α 1 the Noul type also fails, 241 < 252). Score5-typed-DEV is clean everywhere. With no development
  passer there is no formal run (prereg), so there is no formal human-transfer delta.
- **What HR2 does do:** H9 learns HR2's own tasks (HR2 DEV family macro .828 vs .688, +0.140 [+0.112, +0.167]; every
  family up). It raises typed Choice (536 vs 501) and lowers typed Noul (241 vs 264), but this does not carry to the
  held-out human-transfer panel.
- **Recommendation:** no HR2-r2 release-candidate milestone at 4B on this evidence. The 2B, 0.8B and 9B priors are
  negative too (see below). HR2-r2 remains useful as a release-safe data asset, not as the human-transfer lever.

## What ran

- **H9** = the N4XF base (29.40M tokens) plus the whole HR2 TRAIN block (16.75M tokens; gold labels only on HR2 rows).
  - Three full seeds from Nox 1.0 (20260926 / 27 / 28) on node A GPU6–7, image `dbe5f32b`.
  - BEST checkpoints 846 / 847 / 1121 of 1,128; SELECT700 family macro .8971 / .8965 / .8992.
  - The arm artifact is the uniform FP32 soup.
- **Control.** The preregistered C9 (the same base + 16.75M recipe-filler tokens) stopped at its seed-1 preflight.
  - It failed one gate: zero-step cross-process agreement 698 / 700 at drift 0.019, caused by a cold autotune cache
    filled by both chains at once. It was not rerun.
  - By amendment 1 the control is **M7's N7C**: the same base, recipe, seeds and image, with filler = the first 14.10M
    tokens of C9's filler. That is 43.5M tokens (−5.7% vs H9), trained on node B in M7.
  - The early rule E1 did not apply because one seed 1 failed.
- **Amendment 2** raised H9's cap from 4.8 to 5.8 GPU-h, using C9's unused budget.
  - The GPU6 chain was replaced in flight, and the new chain adopted the running H9-s1.
  - The seeds then cost 1.38–1.55 GPU-h each, so H9-s3 would also have started under the original cap. The amendment
    changed the cap, not the outcome.
- **Paths verified exactly.**
  - On the M9 node-A readout path, DEV2.0-4B's weights reproduce the stored node-B readouts: 0 answer differences and
    drift 0.0 on typed DEV, CSS pilot, HT-DEV v2, `hs1-dev` and Score5-typed-DEV. HT-DEV v2 equals the eval
    reference.
  - On the node-A formal path they reproduce the bar run `dev2-4b-t1-derived`: 0 category changes on typed FINAL,
    CSS15 and public 231; v3 63.151.
  - So every M9 number is comparable with the M6–M8 history.

## Development (16K, T = 1, node A; paired against `4b-I` = DEV2.0-4B's weights)

| Point | HT-DEV v2 vs `4b-I` | Typed DEV T (C / N / S) | `rule_precedence` | Score5t check | CSS pilot H3 / median | P | HR2 DEV | Gates |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| `4b-I` (DEV2.0-4B) | H_dev2 .539 | .704 (501 / 264 / 362) | 264 | no flag, top .39 | .5625 / .5364 | 61.47 | .688 | — |
| `4b-H9-a1` (H9 soup) | −.012 [−.025, +.001] TIE | .705 (536 / **241** / 351) | **241** | no flag, top .35 | .5553 / .5678 | 63.27 | .828 | fail: Noul type 241 < 252, `rule_precedence` 241 < 260 |
| `4b-H9-a1_2` ([I, H9]) | −.003 [−.013, +.008] TIE | .698 (496 / 258 / 362) | **258** | no flag, top .40 | .5701 / .5805 | 63.63 | .794 | fail: `rule_precedence` 258 < 260 |
| `4b-N7C-a1` (control soup) | −.008 [−.022, +.006] TIE | .648 (440 / 258 / 338) | 258 | no flag, top .41 | .5634 / .5535 | 59.86 | .693 | fail: Choice, Score, `transition_table`, Noul floors |
| `4b-N7C-a1_2` ([I, N7C]) | +.001 [−.009, +.011] TIE | .703 (491 / 266 / 368) | 266 | no flag, top .42 | .5663 / .5628 | 62.91 | .689 | pass (control: never a formal candidate) |
| `4b-H9-s1` (seed 1 BEST, report only) | −.018 [−.033, −.002] TIE | — | — | — | — | — | — | — |

- **Matched contrasts (HT-DEV v2, H9 − N7C).**
  - α 1: −.0039 [−.0184, +.0099], P(Δ ≤ 0) .69.
  - α ½: −.0041 [−.0142, +.0061], P(Δ ≤ 0) .79.
- **Per task, α 1, H9 vs N7C.**
  - H9 higher: reddit_humor .593 vs .542, tempowic .729 vs .701, wiki_corpus .447 vs .436, media_ideology .416 vs .408.
  - H9 lower: ibc .525 vs .583, conv_go_awry .502 vs .556, talklife .229 vs .245, mrf .766 vs .771.
  - Level: emotion .540 vs .540.
  - The pattern is mixed; no task family that HR2 resembles moves consistently.
- The control reproduces M7's readings of the same weights exactly: N7C α 1 −.008 TIE with the typed floors failing,
  and α ½ +.001 TIE passing.

## HR2 DEV slice (diagnostic; in distribution for H9)

| Family (n) | `4b-I` | H9 α 1 | H9 α ½ | N7C α 1 |
| --- | ---: | ---: | ---: | ---: |
| `hs3_pref` Choice (263) | .749 | .867 | .825 | .776 |
| `eth_util` Choice (354) | .907 | .975 | .955 | .924 |
| `eth_cs` Noul (94) | .904 | .915 | .904 | .883 |
| `eth_deon` Noul (100) | .670 | .960 | .900 | .680 |
| `eth_just` Noul (106) | .811 | .906 | .896 | .792 |
| `prm_step` Noul (60) | .433 | .750 | .633 | .533 |
| `vitc` Noul (262) | .844 | .943 | .935 | .824 |
| `hs3_help` Score-5 (126) | .317 | .429 | .429 | .317 |
| `allegro` Score-5 (250) | .560 | .712 | .672 | .508 |
| **Family macro** | **.688** | **.828** (+.140 [+.112, +.167]) | .794 (+.106) | .693 (+.005 [−.013, +.023]) |

The block trained as intended: large in-distribution gains, with the control flat. The gains did not reach the
held-out human-transfer panel.

## hs1-dev (diagnostic)

| Point | Quote adoption (ideal .50) | False yes on unmet conditions (ideal 0) | F2 policy accuracy |
| --- | ---: | ---: | ---: |
| `4b-I` | .775 | .262 | .609 |
| H9 α 1 | .758 | **.316** | .623 |
| N7C α 1 | .779 | .205 | .628 |

HR2's Noul rows (ETHICS, PRM800K, VitaminC; 50 / 50 balanced) push the unmet-condition yes-rate up. The Noul head
shift that fails the `rule_precedence` floor shows up here too.

## Formal

Not run: no H9 point passed the development gates, and the preregistration sends only an H9 development passer to
the formal runner. The formal-path parity run was made beforehand (`m9-ref-N4XF`, DEV2.0-4B's weights, T = 1, node A,
`dbe5f32b`, copy of `cache-frozen` `f6d0f920…`). It is exact against the bar (0 category changes; v3 63.151, T .688,
H .580, public 231 171), so a future node-A 4B formal run can pair with the stored bar directly.

## Recommendation (preregistered rule)

- **Verdict: not a lever.** The rule says "not a lever" if ΔH_dev2(H9 − control) ≤ 0; it is −0.004.
  - The "weak / conditional" branch needs a positive matched contrast, which this is not.
  - The CI rules out the +0.02 that the "lever" branch requires.
  - The typed Noul floor failure is real but secondary: removing it would not create a human-transfer signal.
- **4B:** do not commission an HR2-r2 release-candidate milestone with this design (a whole-block addition with full
  fine-tuning from Nox). A Noul-protected or smaller HR2 dose could probably clear the floor, but there is no
  human-transfer effect on the screen to protect.
- **2B (prior: no).** Its recorded failure mode is a Score-head shift under human Score rows (M8-small), and HR2 is
  26% Score rows. At 4B the Score head held (Score5t clean), but 2B is more fragile.
- **0.8B (prior: no).** Its development gains did not carry to formal in M6 / M8-small. HR2 produced no development
  human-transfer gain to carry.
- **9B (prior: no).** Its binding problem is the multilingual PAWS-X yes-bias. HR2's Noul rows are English-heavy, and
  at 4B they raised the unmet-condition yes-rate and broke `rule_precedence`, the same Noul fragility that sank 9B M5 /
  M6 arms.
- **Program reading.** At 4B, more human-labelled rows under full fine-tuning teach their own tasks but do not
  transfer: the same pattern as typed / A7 data, HS1, PN1 and teacher soft targets.
  - The 27B M5 attribution (11:35: the adapter kept the human transfer that full FT lost) points to the training
    method rather than the data as the next thing to test. 4B M10 is already probing that.
  - If HR2-r2 is used in 27B M6 as planned, it needs a matched control and a Noul guard, and it should not be expected
    to raise human transfer on its own.
- **Limits.**
  - One dose: the whole block, 36% of H9's tokens.
  - One recipe: full fine-tuning from Nox.
  - A near-matched control: −5.7% tokens, trained on node B in M7, with otherwise the same recipe, seeds and image.
  - The screen is HT-DEV v2, not formal CSS15.
  - Three seeds per arm; the seed-1 read (−.018) agrees in sign with the soup.

## GPU-hours: 5.94 of 16

| Item | GPU-h |
| --- | ---: |
| H9 training, 3 seeds incl. preflights and postruns (1.551 / 1.526 / 1.383) | 4.460 |
| C9-s1 (zero-step, one-step, failed gate) | 0.100 |
| References (`4b-I`, 6 panels), control line (N7C, 11 reads), H9 line (11 reads), seed-1 early read | 1.031 |
| Formal-path parity (`m9-ref-N4XF` v3 + mlx-diag, co-tenant) | 0.348 |
| **Total** | **5.939** |

Co-tenant reads are counted in full. Node A only; node B was read-only (inputs over the direct link). Node-A run
directories: `/data/dev2/runs/dec/m9/` (`data`, `teacher`, `exposure`, `arms`, `soup`, `lines/4b`, `formal`, `select`,
`post`, `results`). The H9 soup: `m9/soup/H9/build/H9-soup` (per-file list `49783cc3…`); the N7C copy:
`m9/control/N7C-soup`.

## Tooling and incidents

- **New tooling** in `v2/dec/ops/m9/`: compose, lock, node-A chains with the early rule and an adopt mode, soup, lines
  with the readout-path parity check, HR2 DEV prompts and scoring, rules, formal wrapper, post-soup chain and results
  summary. 11 tests are in `v2/dec/tests/test_m9.py`; 139 decoder + guard tests pass in the image.
- **Small decoder-only changes:** `drive_arm.sh` (node-A GPU6–7, image override), `launch.sh` (opt-in
  `HIP_FORCE_DEV_KERNARG=1`) and the `ops/m6` formal library (`M6_4B_NODE=A`). No shared module changed.
- **Moved node B → node A over the direct link**, each checked by content manifest: the 4B recipe inputs, the node-B
  readout autotune cache, the 4B frozen formal caches and M7's N7C soup.
- **C9 preflight failure:** a cold training autotune cache was shared by two simultaneous starts.
  - The program's preflight caught it as designed. No rerun followed.
  - **Lesson for every track:** start the first job on a fresh node / image cache alone (or pre-warm it) before
    launching parallel seeds.
- **GPU-hour counting:** the first copy of `m9_gpuh.py` also counted the copied M7 parity receipts (+0.226 GPU-h,
  conservative for the stop rules). It was fixed in `eccf0a232`, before the totals above.
- **Time stamps:** the prereg's "written ≈11:10" and amendment 1's "≈11:35" are about 30 min late, and amendment 2's
  "≈04:05Z" means ≈03:57Z. The commit times (10:37, 11:10:50 and 11:55 UTC+8) are authoritative. They precede, in
  order, every GPU job, every arm readout and every H9 readout.
