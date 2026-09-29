# JevBench public 231: provenance, composition and reproduction fidelity

Scope: aggregates only. Upstream = `fstandhartinger/jevbench` clone at HEAD `9ec6f15a`
(pin `1bcc55eb`). Ours = worktree head `28f05ace7`, stored runs on node A (and node B
rows adopted there). Drivers: `drivers/` (stdlib, run on node A via stdin).

## Verdict

**Faithful.** The 231 items and our scorer reproduce upstream's public items and
accuracy rule. Re-scoring all 86 stored public-231 runs with an upstream-faithful rule
changes **0 item outcomes**, and the panel rebuilt from upstream HEAD is byte-identical
to node A's. What it is **not** is the official JevBench score: public 231 covers
about 36% of the inputs to the official Intelligence axis and none of the sealed set.
On the sealed set, every one-pass decision model upstream collapses to near chance.

## 1. Provenance and composition

**Who.** JevBench belongs to Benchmark Heaven (README.md:9-10). It is a one-person
hobby project (README.md:426) and is not affiliated with TypeSafe. Code and the 72
original items are MIT (README.md:420).

**How the public items were authored and labelled.**

- Standard 72 (`original.jsonl`): 36 hand-written paraphrase pairs, written inline
  in `scripts/build_public.py:6-58`. Every row has `label_basis` "Explicit rubric,
  reviewed before inference". README.md:394-395 calls them "short and hand-written".
  No LLM author and no reviewer are named.
- Easy 48: "authored and reviewed before inference" (item provenance); no author is
  named. It was frozen before v1.1 inference (RESULTS-v1.1.md:60).
- Hard 111: **LLM-generated**. 54 items are by Claude Opus 5 and 57 by GPT-5.6 Sol,
  in batches opus-a/b/c = 17/23/14 and sol-a/b/c = 22/21/14. Each model reviewed the
  other's items: a blind answer, then a gold verdict, then one discussion round
  (datasets/HARD-TIER.md:14-45). Items were frozen 2026-09-19T11:43Z before any
  entrant saw them. A non-entrant pilot model (Gemma 4 31B) was used once, only to
  trigger a harder round 2 (HARD-TIER.md:27-29, 146-159). Gold is a single label; 10
  probability items also carry exact `gold_probs`, which upstream uses only for
  calibration. There was no human review of gold.
- The 146-item judge tier is imported: 78 routing requests with human-assigned
  categories and 68 math answers judged by a deterministic grader (README.md:291-293).

**Official suite by version** (datasets/manifest.json; README.md:278-292,
83-84, 256-259; CHANGELOG.md:30-38):

| Tier | Total | Public (ours) | Held out / private | Weight in I13 |
|---|---:|---:|---:|---:|
| easy (v1.1) | 72 | **48** | 24 easy-heldout | 0.14 |
| standard (v1.0) | 96 | **72** | 24 heldout-private | 0.28 |
| judge (v1.0 imported) | 146 | 0 | 146 (router 78 + judge 68; not redistributable) | 0.28 |
| hard (v1.2) | 220 | **111** | 109 held out | 0.30 |
| sealed (v1.4.0) | 308 | 0 | 308 fresh sealed | 20% of I |

v1.0 had 242 decisions. v1.1 had 314. v1.2 to v1.3 had 534. v1.4 through
v1.4.2.2 have 842. The sealed families (n) are temporal_numeric 56, judge_hard 41,
long_policy 40, multi_hop 38, ambiguous_abstain 37, probability 28, tradeoff 26,
safety_judge 16, paraphrase_robustness 14 and trap_adversarial 12. Upstream also
reports sealed results by "panel stratum" (0/3 = 109, 1/3 = 97, 2/3 = 102) without
defining the strata. **Ours contains exactly the 231 public items. It lacks the 24
held-out easy items, the 24 heldout-private standard items, all 146 judge items, the
109 held-out hard items and all 308 sealed items.**

**Official score** (docs/METHOD-v1.4.md:7-32; composite_v14.py):

- I13 is chance-corrected per tier, `(acc − chance)/(1 − chance)` clipped at 0, with
  weights easy .14, standard .28, judge .28 and hard .30.
- `Isealed = 100·max(0,(a−0.293)/(1−0.293))` and `Ibase = 0.8·I13 + 0.2·Isealed`.
- Rule quoted exactly: "`gap = 100 × (p − a)` # percentage points;
  `I = Ibase × (1 − max(0, gap − 25) / 100)`", where "`p` accuracy on the 231 public
  items" and `a` is sealed accuracy. Prose version: "A public-to-sealed accuracy gap
  above 25 percentage points reduces Intelligence" (README.md:36).
- Calibration: `C = C13 + (C14 − C13)·min(1, 0.2/0.35)`.
- Composite: equal-weight harmonic mean (p = −1) of Intelligence, Calibration, Speed
  and Cost. Each of I, Speed and Cost below 50 multiplies the composite by (x/50)².

Public items feed the official Intelligence inputs at weight
0.8·(.14·48/72 + .28·72/96 + .30·111/220) ≈ **0.36**. We measure no judge, sealed,
calibration-axis, speed or cost component. The one exact correspondence is that **our
pooled public-231 accuracy is literally `p` in the official gap rule.**

**Sealed JevBench exists** (308 items). Only aggregates are published, and nothing we
measure corresponds to it. Across 87 non-LLM rows it is near chance: median sealed
accuracy is 0.289, 47 rows fall below the 0.293 chance baseline, and the maximum is
0.60 (djev-thinking). The median public-minus-sealed gap is **47.1 pp**, and **74/87**
rows exceed 25 pp, including upstream #1 Imajev-4B (86.1% vs 37.0%). Reasoning LLMs
keep small gaps: GPT-6 Luna scores 99.6 → 95.5 and DeepSeek V4.1 Flash 97.8 → 94.8.
Spearman(public 231, sealed) over the non-LLM rows = 0.52. Public-hard and
held-out-hard are about equally difficult: upstream reports a field mean gap of
−0.7 pt (README.md:201). The sealed collapse therefore reflects a harder set, not
public-item leakage in general. Contamination is another helper's topic.

**Is the public set identical at HEAD?** Yes. `git diff --stat 1bcc55eb HEAD --
datasets jevbench` is empty; the 23 changed files are results, docs and build scripts.
The three public file SHA-256 hashes match `manifest.json`. Rebuilding locally from
HEAD with `jevbench_public.build` reproduces node A's prompts (`642d3fac…`), targets
(`abc17b97…`) and manifest (`e0e7c677…`) exactly.

## 2. Upstream numbers vs ours

Full table: `upstream-vs-ours.json`. Public counts are easy/standard/hard (total).

| Ours | Ours | Upstream row | Upstream public | Sealed | Gap pp | Score (rank) |
|---|---|---|---|---:|---:|---|
| Decider 4B (eb5fbdfc) | 48/71/73 (192) | decider-4b-v2 (7ab294cb) | 48/71/74 (193) | .347 | 48.8 | 64.13 (#3) |
| Decider 2B (533964da) | 48/64/63 (175) | decider-2b (b37f7e1, "v10") | 48/61/55 (164) | .247 | 46.3 | 30.74 (#41) |
| Hopper-G 1.2 | 48/70/76 (194) | Hopper (round 4) | 48/?/? (190) | .341 | 48.2 | 59.43 (#7) |
| Nimble 9B v2 | 48/68/69 (185) | Nimble 9B v1, 8K limit | 48/67/69 (184) | .289 | 50.8 | 18.66 (#62) |
| GLiNER2.5-Decide | 48/46/22 (116) | gliner2.5-base / multi / small | 134 / 113 / 106 | .29/.33/.29 | 29/16/17 | 11.8/9.8/7.2 |
| Kev 0.8B | 48/58/41 (147) | kev 0.6B / 4B / 8B | 154 / 153 / 165 | .24/.22/.22 | 43/44/50 | 24.8/36.1/25.6 |

- **Agreement.** Where the same model family was run, totals agree within 1 to 4
  items (Decider 4B −1, Nimble +1, Hopper +4 on a different checkpoint).
- **Decider 2B (+11)** is a weight-revision difference, not a scorer difference.
- Upstream's Nimble notes are a token-limit precedent: hard accuracy was 43.6% at
  2,048 tokens and 65.5% at 8,192 (README.md:186-188).
- **No upstream rows** exist for Decision 1.0 (Kai, Lex, Eos, Sol, Nox, Lux), DEV2.0,
  Bosun, Jebadiah, This-That, Jet, Intern-Decision, JPT-0.8B or JPT-9B.
- JPT-4B, Eikos 4B/27B and AutoJev-27B are "moved to v1.4.3" with no numbers
  (docs/RELEASE-v1.4.2.md:49-51).
- **Contamination hints upstream.** Every matched peer shows a 43 to 51 pp gap, the
  same as the field. Hopper's row records heavy public-directed development (26
  configs, 20+ calibration maps tuned against the public half). decider-4b-v2
  discloses 8,000 LoRA rows generated from the published names of the sealed families.

## 3. Fidelity of the reproduction

**(a) Scoring semantics.** Ours is `jev_arena/jevbench_public.py`: `_valid_probs`
169-192 and `_evaluate` 195-238. Upstream is `jevbench/scoring.py:27-124`,
`adapters/typesafe.py:81-107` and `runner.py:43-44,59-60`.

| Aspect | Upstream | Ours | Can flip an outcome? | Observed |
|---|---|---|---|---:|
| Option keys | exact label set | exact label set (172) | no | 0 |
| Values | non-bool number, finite, in [0,1] | same (177-181) | no | 0 |
| Sum tolerance | strict 1e-3, else renormalize inside 0.02 | 0.02 band, always divide by total (185-191) | no (argmax is scale-invariant) | 0 |
| Argmax ties | lexicographically smallest label | `min(-p, label)` (221) | no | 0 |
| Noul | `{yes: p, no: 1−p}`, so 0.5 → "no" | same (201-210) | no | 1–2 exact-0.5 answers in 7 peer runs, both "no" |
| Score | argmax over "0".."k" keys | same | no | 0 |
| Missing `type` | invalid (typesafe.py:82-84) | accepted (`get("type", qtype)`, 199) | yes, in principle | 0 answers lack `type` |
| Choice point field | must be one of the labels (typesafe.py:92) | not required; reports point≠argmax (235-237) | yes, in principle | 0 on valid answers |
| Missing / failed / over-limit (422) | wrong | wrong (424-435; over-budget invalid) | no | — |
| Chance correction, tier weights | per tier; .14/.28/.28/.30 | raw accuracy; tier-macro for the arena axis | metric only | — |
| Brier / ECE | sum; 10 bins | sum/2; 15 bins | metric only | — |

Driver `rescore_upstream_faithful.py` rescored all 86 runs that have a public-231
REPORT on node A. Our rule and the upstream-faithful rule agree on every item:
**0 flips**, and every REPORT total is reproduced.

**(b) Renderer.**

- The model-visible object is `{state, questions: {decision: question}}`, where each
  upstream `question` holds exactly {type, instructions, criteria} (all 231 items).
  This is identical to upstream `base.build_question` / `typesafe.build_request`
  (base.py:72-81, typesafe.py:36-40), and the input digests bind it
  (jevbench_public.py:62-71).
- Decision 1.0 and Decider receive this object unchanged through their published
  `decide` / `system_one` (inference/run.py:431-436). This is the same wire format
  upstream sends to TypeSafe-compatible servers, including decider-4b-v2 and kev.
- DEV2.0 (`training/model/infer.py:83-146`; `decision_model.py:48-60`) renders
  "Context / Task type / Question / Options". Dict states are canonical JSON with
  sorted keys (data.py:37-43). Options keep criteria insertion order. Noul is
  rendered as keys `false` and `true`, false-first in all 74 items. Score levels are
  keys "0".."k" with their descriptions, and 2–10 levels are accepted. There is no
  truncation: over-length input is invalid (decision_model.py:77-80).
- Upstream's own adapters for other System-One rebuilds re-map Noul differently:
  so1 uses options no/yes plus a rubric, and sg_system_one uses yes-first. Upstream
  documents order sensitivity of 21% vs 72% (README.md:96-98). We follow each
  model's own package, which is upstream's stated policy (README.md:296-299).
- Slice audit (`render_slice_audit.py`). On 4-level Score, DEV2.0 gets 14–15/18 and
  peers 14–16/18. On dict states, DEV2.0-4B gets 23/35, the same as Decider 4B
  (Nox 19, JPT-4B 26). All standard Score items are 12/12 for every model.
- The DEV2.0 deficit is in hard Choice (4B: 35/67, vs Decider 4B 49, JPT-4B 53) and
  hard Noul (4B: 18/38, 8B: 23/38, vs JPT-4B 30, Eikos 36). Nox 1.0 (23/38), which
  uses the same wire format, is equally low. **No rendering choice specifically
  disadvantages DEV2.0.**

**(c) Invalid answers and budgets.**

- Invalid means a missing or ill-typed answer, a bad key set or value, a sum outside
  0.02, or a native over-budget rejection.
- Public-231 state length: easy max 81 chars, standard max 158, hard median 1,583 and
  max 14,986. 37 hard items exceed 4,000 chars and 36 exceed 8,000.
- The maximum native input on public 231 is about **3,980 tokens** (DEV2.0 0.6B–8B
  3,980; Decider 4B 3,911; Kev 3,897). Limits of 8,192 / 16,384 / 32,768 tokens are
  never reached, so **DEV2.0 has 0 over-budget answers.**
- All stored invalids are hard-tier context overflows of short-limit peers:
  - Kai and Lex (1,024 tokens): 44 each. Kai's 8K run has 0 invalid and scores 127
    (hard 30 → 43).
  - GLiNER2.5-Decide: 56.
  - This-That 1.2 (1,536-token state): 37.
  - Jebadiah (2,048-token render): 36.
- Upstream also counts context refusals (422) as wrong, so this matches.

## 4. Relation to JevArena v3

- **No v3 component is derived from or modelled on JevBench.**
- Typed FINAL is 1,600 programmatic-oracle items, all with dict states, in four
  families: constraint_competition, exception_stack, evidence_join and
  resource_ledger (benchmark/README.md:13-25). Its question mix is Choice 800 /
  Noul 800 / Score 400, with 5-level Score. State median is 403 chars and max 462.
- CSS15 is 6,547 human-labelled items over 15 computational-social-science
  classification tasks from the CSS replication sources (transfer/build.py:21-42).
  All are Choice; state median is 208 chars, max 57,627, and 420 items exceed
  4,000 chars.
- JevBench was an axis in v1/v2 (arena_v2.py:131,263). v3 excludes it and keeps
  public 231 as a separate cross-check (arena_v3.py:602; jev_arena/README.md:13-17).
- **What JevBench measures that v3 does not:**
  - long policy or contract documents with interacting clauses and amendments
    (2k–6k tokens);
  - multi-hop lookups across tables, footnotes and aliases;
  - date and number arithmetic (business days, pro-rating, thresholds);
  - evidence-derived probability items;
  - adversarial embedded instructions and traps;
  - subtle answer-adequacy judging;
  - overlapping routing;
  - explicit "insufficient information" abstention;
  - paraphrase pairs (standard tier).
  These are LLM-authored English scenarios of mixed string/JSON form.
- **What v3 measures that JevBench does not:**
  - programmatically verified reasoning with counterfactual, order and relabel
    variants;
  - 5-level Score at scale (JevBench has 18 Score items);
  - human-labelled real-world social-language tasks (stance, hate, politeness,
    humour, persuasion, and others);
  - scale: 8,147 originals vs 231, so much tighter intervals.
- The v3 typed states are short (≤ 462 chars) and exercise none of JevBench's
  long-document hard families. That is the most likely structural reason models can
  look good on v3 but weak on JevBench-hard. The gap analysis belongs to another
  helper.

## Files

- `findings.md` (this file), `upstream-vs-ours.json`
- `drivers/rescore_upstream_faithful.py`, `drivers/invalid_and_length_audit.py`,
  `drivers/panel_length_profile.py`, `drivers/render_slice_audit.py`
