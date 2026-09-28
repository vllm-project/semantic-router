# Eval M3.1 — option-key / position leak audit of every eval panel (2026-09-28)

Eval & peers track. Preregistration: [`m3-prereg-2026-09-28.md`](m3-prereg-2026-09-28.md)
and [amendment 1](m3-prereg-amendment-1-2026-09-28.md). Code: `v2/eval/leak_audit.py`
(run v1 at `3bea72d93`, v2 at `5631d362c`) and `v2/eval/leak_effects.py`. CPU only, node A,
all eight panels hash-verified before reading. Count-only per-group tables:
[`m3-leak-audit-tables-2026-09-28.md`](m3-leak-audit-tables-2026-09-28.md).

## Result

**No panel carries the A7 option-key leak.** No panel has a `result_<n>` key, no panel
has numbered keys out of display order, and no item id contains its gold key. Seven of
eight panels are CLEAN on every option-surface and state-surface cue. Two design
properties of other kinds were found and quantified. Neither changes any score or
ranking, so **no panel is corrected, re-scored or re-issued, and the development proxy
and tie rule stay as calibrated** (P = 100·√(T_dev·H_pilot), v3 ≈ 19.12 + 0.629·P,
|ΔP| < 4 is a tie).

| Panel | Role | Questions | Key schemes | Verdict (option surface / with state) | Finding |
| --- | --- | ---: | --- | --- | --- |
| SELECT | development | 700 | words, true/false, `K1..K4` in display order (120 Choice rows), levels | CLEAN / CLEAN | Numbered keys follow display order; gold positions are within sampling noise (the SELECT and CAL skews do not agree) |
| CAL | calibration | 700 | same as SELECT | CLEAN / CLEAN | same |
| typed DEV | development | 1,600 | opaque `X?????` + `none`, true/false, levels | CLEAN at panel level / LEAK in one family | `transition_table`: the correct state is never the current state (design property; see below) |
| CSS pilot | development | 1,430 | words | CLEAN / CLEAN | One fixed option set and order per task: no item-level surface exists |
| typed FINAL (v3) | formal | 2,000 slots | opaque + `none`, semantic words, true/false, levels | CLEAN / CLEAN | Largest single cue: lexicographic key rank in `constraint_competition`, +4.5 [−0.3, +9.3], not significant |
| CSS15 (v3 human transfer) | formal | 6,547 | letters, words, digits | CLEAN / CLEAN | One fixed option set and order per task (15/15) |
| JevBench public 231 | formal (public subset) | 231 | words, a few numbered words, true/false, levels | CLEAN combined; one single cue | The correct Choice option is the unique longest description more often than chance (see below) |
| mlx-diag | development diagnostic | 2,275 | words, true/false, levels | CLEAN / CLEAN | Fixed option sets per source |

The A7 hypothesis that 1.0-lineage models exploit construction-order keys on development
panels, and that this explains the dev-vs-v3 misalignment, is **not supported**: the
development panels contain no such keys.

## Finding 1 — typed DEV `transition_table`: the current state is never correct

The generator always includes a transition row that matches, so "if no row matches,
remain in the current state" never applies. The option equal to the state's `current`
field is never the gold (0/400). This makes the question effectively two-way for a model
that knows the rule, and the state-role cue alone reaches 51.8% against 33.3% chance
(+18.4 [+15.4, +21.4]). It is a property of one development family (a quarter of
T_dev's family macro), not of the option keys, and it does not reach typed FINAL (other
families).

From stored development predictions, weak models are **trapped** by this distractor
rather than helped: Kai1 picks the current state on 88% of the questions, GLiNER2.5
90.5%, Lex 71%, DEV2.0-0.6B 68%, Bosun 0.6B 28%, Sol1 24%, Nox1 20.5%, Intern 20%,
Kev 19%, Eos1 13%; Decider 4B, JPT-4B/9B, Lux1, AutoJev, Eikos, Jebadiah and This-That
never do.

Proxy sensitivity (T_dev without `transition_table`, same models and statistics):

| View | Proxy | Spearman | LOO v3 error MAE / max | Pairs in v3 order | Same-tier pairs |
| --- | --- | ---: | --- | ---: | ---: |
| 16-model calibration set | frozen P | 0.941 | 3.13 / 7.35 | 109/120 | 14/18 |
| | P without the family | 0.935 | 3.13 / 7.58 | 110/120 | 14/18 |
| all 24 models | frozen P | 0.948 | 2.67 / 7.01 | 252/276 | 28/38 |
| | P without the family | 0.940 | 2.86 / 6.62 | 251/276 | 28/38 |

The variant is not better on leave-one-out error and same-tier order together, so by the
preregistered rule **the frozen proxy is kept**. The finding is recorded so that a future
typed-DEV v2 adds no-match rows.

## Finding 2 — public 231: the correct Choice option is often the longest

In 131 public Choice items with a unique longest description, that option is the gold in
44 (33.6%) against a chance rate of 23.1%. The shortest description carries no cue
(21.7% against 22.9%). Description length alone scores +9.4 [+1.1, +18.1] over chance on
pooled public Choice; the combined surface model does not (+0.3 [−6.5, +8.0]). This is
an answer-length artifact of the hand-written public JevBench items, not a key or
position leak.

No measured model exploits it. Accuracy when the gold is the longest option is about the
same as when it is not. The pick rate of the longest option is 18–37% for every model,
against a gold rate of 33.6%. Reweighting each model's Choice results to the chance rate
(0.231 × accuracy when the gold is longest + 0.769 × accuracy otherwise) changes Choice
accuracy by −0.6 to +1.8 points: at most 2.3 items of 231, mostly in favour of weaker
models (Kai1 +2.3, DEV2.0-0.6B +1.8, Lex +1.8). No same-tier order changes.

**Proposed disclosure (formal panel unchanged; for the coordinator).** Add one line where
public-231 counts appear internally: "In the public 231, the correct Choice option has the
unique longest description in 44 of 131 items (chance 23%). A length-balanced reweighting
moves every measured model by at most 2.3 items and changes no same-tier order, so counts
are reported unchanged." Cards need no change; they already label public 231 as a
public-subset reproduction.

## Method notes

- v1 → v2 (amendment 1): the v1 CUE on pooled CSS15 (+1.9 [+0.7, +3.1]) and the v1 LEAK
  on pooled public Choice (+8.1 [+1.7, +14.9]) came from the reference, not from the
  panels. Cross-validated majority priors fall below chance in class-balanced groups, and
  the pooled prior was not conditioned on task. v2 uses the better of chance and the
  prior as the reference, and a group-conditioned prior in pooled rows. Every per-group
  verdict is the same in v1 and v2.
- Single CUE verdicts that do not replicate on the sibling panel are not counted (typed
  DEV `transition_table` option surface +4.2 [+0.4, +8.4]; pooled typed DEV Choice −0.3).
- CSS panels: trailing whitespace on each task's last option ("Surprise ", "None ") is a
  formatting quirk that is constant within the task. It carries no item-level
  information.

## Receipts (node A `/data/dev2/runs/eval/m3/leak-audit/`; private HF eval-artifacts `m3/leak-audit/`)

| File | SHA-256 |
| --- | --- |
| `audit-v1.json` (code `3bea72d93`) | `480b61f55fbcc0e1adc65440b8c68c47c5ba5c6da0da5e4ba65bff1bf3474d80` |
| `audit-v2.json` (code `5631d362c`) | `781c2591a65f02e47045d0772dc731e8dbae8d3c64c586c88d0908ab09e774cb` |
| `follow-public231-longest.json` | `7ede7b8661b16de31d0f3b14cd56bfb6ee86cdd9f108cfbf42df1529580780a7` |
| `follow-public231-shortest.json` | `54f7c15bc8be613c31533d5b44b3558bdf59cea4f1d8965f910ffdd6f5ed8eaf` |
| `follow-typed-dev-current.json` | `7aa4b09a94481df05029192a8f9ca28a4a87b3ac7d90081cc1c3f7397317cb85` |
| `proxy-excl-transition-table.json` | `a5c6670dae3fc703999ddda78b6df9ed2c262b288e5f9223914631eb4932ff91` |

GPU-hours: 0 (CPU only, about 3 minutes of node A CPU in total).
