# HT-DEV v2: human-transfer development panel — preregistration (2026-09-30)

Owner: eval track (development-panel worker; branch `xunzhuo/decision-2-training-eval-devpanel`). Commissioned by the
coordinator on 2026-09-30 01:50 UTC+8. This file is committed and pushed before any HT-DEV v2 item is converted and
before any model is run on it. HT-DEV v2 is a DEVELOPMENT panel: never a release score, never in the v3 composite,
the rank charts or a model card; its numbers are labelled "development readout".

## 1. Target, comparator, question

- **Target:** within-tier deltas of formal human transfer H (post-key same-panel JevArena v3 CSS15: median over the 15
  evaluation tasks of task macro-F1; stored `REPORT.json` → `panels.css15.H`).
- **Bar:** the CSS-pilot three-task mean `H_mean3` (mean macro-F1 of semeval_stance, implicit_hate, discourse), the
  current human-transfer screen (coordinator 09:25). HT-DEV v1 (median 0.539) did not beat it (0.607).
- **Question:** do within-tier differences on the new panel agree with formal ΔH clearly better than `H_mean3`?

## 2. Candidate designs and the choice (before building)

| Design | What it is | Evidence before building |
| --- | --- | --- |
| (a) | CSS15 parallel form: more items from held-out portions of the CSS15 task sources, same tasks, templates and option maps, disjoint from the formal items at item and group level | Design check below |
| (b) | Fresh-source, dataset-isolated human-labelled panel | This is HT-DEV v1 (13 isolated tasks): 0.539 vs 0.607, construct mismatch (v1 record) |
| (c) | Combination | Informational only (§5) |

**Design check** (`v2/eval/htdev2/ceiling.py`, commits `fe5ba61bf` / `4b6917838`; aggregates in
`records/htdev2/design/`). Stored formal CSS15 predictions of the 58 validation models, no inference, no development
item. Each of 200 replicates splits every CSS15 task's items (stratified by gold) into a "development" part of fraction
f and a disjoint target part; `H_mean3` is scored against the same targets (52 models have a stored pilot readout).
This is the most favourable parallel form possible, so it bounds design (a):

| Dev tasks | f | Dev median | Dev mean | `H_mean3` | P(better), mean [min of 20] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 0.10 | 0.827 | 0.857 | 0.654 | 0.987 [0.959] |
| 15 | 0.20 | 0.872 | 0.890 | 0.653 | 0.998 [0.961] |
| 15 | 0.35 | 0.901 | 0.907 | 0.649 | 0.998 [0.986] |
| 13 (no dialect, tropes) | 0.20 | 0.864 | 0.891 | 0.653 | 0.999 [0.975] |
| 9 | 0.35 | 0.815 | 0.819 | 0.649 | 0.973 [0.781] |

Observed on the full formal target: `H_mean3` agrees on 0.663 of 181 decidable pairs (within-tier r 0.513). Formal
within-tier ΔH is therefore highly reproducible across disjoint items of the same tasks; the v1 failure was the
construct, not target noise. **Choice: build design (a), named `ht-dev2`.** (b) is not rebuilt (HT-DEV v1's stored
predictions stand in, §5). The task mean is chosen as the primary functional because it is the more robust estimator
at the affordable size and task coverage (table); the median is secondary. This choice used the design check only.

## 3. Panel design

Sources are the datasets behind the formal CSS15 tasks, processed as the SALT preparation did (pinned
`SALT-NLP/LLMs_for_CSS@55183a64`, `mappings.py` templates and column maps, `data_loader.py`
sha256 `2d6a668c…` logic reimplemented without ConvoKit or pandas), rendered to typed choice with
`transfer.build.criteria_for` and the authors' maps (`decision-models-css@311956c2`, `pilot_jev.py`). States and
instructions are therefore formed exactly like the formal items.

| Task | Held-out pool | Group key (disjointness unit) |
| --- | --- | --- |
| emotion | `dair-ai/emotion` unsplit rows not in the formal sample (SALT label letters by the same mapping) | normalised text |
| ibc | SALT `ibc.csv` rows not in the formal sample | normalised sentence |
| media_ideology | SALT `media_ideology.csv` rows not in the formal sample | article (url, title) |
| raop | SALT `raop.csv` rows not in the formal sample | normalised sentence |
| talklife | SALT `talklife.csv` rows not in the formal sample | seeker post (`sp_id`) and response (`rp_id`) |
| wiki_politeness | ConvoKit `wikipedia-politeness-corpus` | conversation |
| tempowic | TempoWiC train and test splits plus unused validation rows (cardiffnlp GitHub) | tweet pair and each tweet |
| flute | SALT `flute-classification.json` rows not in the formal sample | premise, hypothesis |
| mrf | SALT `mrf-classification.csv` rows not in the formal sample | normalised headline |
| conv_go_awry | ConvoKit `conversations-gone-awry-corpus` | conversation and its page |
| persuasion | ConvoKit `winning-args-corpus` | conversation (thread) |
| reddit_humor | RedditHumorDetection `reddit_full/test.tsv` rows after the first 500 (the formal sample), then `dev.tsv` | normalised joke |
| wiki_corpus | ConvoKit `wiki-corpus` (power: speaker is admin) | conversation |

Excluded a priori: `indian_english_dialect` (the formal sample is the whole 266-row annotated set, so no held-out
portion exists) and `tropes` (the formal task scores 114 singleton labels including 42 composites; the 320 held-out
characters carry only the 72 single tropes, so the task cannot be reproduced). **13 tasks.**

**Reconstruction check.** For each task the builder maps the formal CSS15 items to its reconstructed pool by normalised
context hash (`normalized_context_sha256` from the gold file; ids and hashes only). A task is admitted only if ≥ 95% of
its formal items map; otherwise it is dropped (parallelism and group disjointness cannot be shown).

**Exclusions** (applied before sampling; counts recorded, never text):

1. Formal CSS15 and the pilot: every pool item whose normalised context equals a CSS15 or pilot item's, and every pool
   item sharing a group with a mapped formal item.
2. Training: `v2.eval.sealed.overlap scan` of the pool against every training corpus (the C1 v1.3 rescan's node-A
   training manifest plus data landed since: PN1, HS1, round-1 mixtures, listed in the build record); drop items at
   REVIEW or worse (containment ≥ 0.2 or an exact span of ≥ 8 tokens).
3. Protected panels (item level, same thresholds): typed-final, typed-dev, public231 (JevBench), mlx-diag, ht-dev,
   score5-dev, score5t-dev, hs1-dev, SELECT, CAL.
4. JevArena-C1: dataset level only; no source is a C1 registry dataset (`cardiffnlp/GrainWiC`, a rejected candidate,
   is a different dataset from TempoWiC). The sealed directory is never read.
5. Duplicate normalised contexts within a pool keep one item.

**Sampling.** Per task, a class-balanced sample like the formal one: per-class quota ⌊150 / k⌋ for k gold classes
(reddit_humor, whose formal sample is unbalanced 252/248, is also balanced); at most 2 items per group; order by
`sha256("ht-dev2/1:<task>:<source key>")`. Floor 60 items per task, else the task is dropped. At most 1,950 items.
Leak audit (`v2.eval.leak_audit audit`) and length-only baselines are reported, not gated.

## 4. Scoring

- Per task: macro-F1 over the task's labels, `transfer.score` semantics (invalid or missing counts as wrong).
- **Primary: `H_dev2` = mean over admitted tasks of task macro-F1.** Secondary: `H_dev2,median`.
- Noise: item bootstrap within tasks (2,000 draws, seed 20260930).

## 5. Validation

- **Models:** the 58 of `records/htdev2/spec.json` (sha256 `27a53b82…`): every unique-weight 0.6B–9B model with a
  stored node-A formal run, siblings included; 27B excluded (weights and kernels on node B).
- **Collection:** one job per model with its formal runtime (adapter spec, input limit, image, autotune cache from its
  formal `COLLECT.json`), `--panels ht-dev2`, plus `css-pilot` for the 6 models without a stored node-A pilot readout.
- **Comparator:** `H_mean3` from the pilot predictions named in the spec.
- **Primary metric:** within-tier sign agreement of ΔH_dev2 with ΔH_formal on decidable pairs (|ΔH_formal| ≥ 0.02),
  against the same statistic for `H_mean3`. Paired model bootstrap: 5,000 draws, seed 20260930, models resampled
  within tiers, two draws of one model form no pair.
- **Pass rule:** (i) primary agreement higher with bootstrap P(better) ≥ 0.90, and (ii) within-tier Pearson r of
  tier-demeaned scores with formal H higher (point estimate). Otherwise it fails and nothing is recommended.
- **Secondary (informational, no decision):** thresholds all / 0.01 / 0.03; the median functional; Pearson r of
  within-tier pair deltas; per tier; sibling pairs (same lineage, both 2.0 candidates); the formal CSS15 task-mean
  target; the pilot median; HT-DEV v1 on its 37 models; combinations (c) = ht-dev2 ∪ pilot tasks (mean of 16) and
  ht-dev2 + HT-DEV v1 (37 models). **JevArena-C1 post-key:** within-tier sign agreement of ΔH_dev2 with the stored C1
  aggregate deltas on the C1-scored models, reported only; nothing is ever tuned to C1.
- **Screen band (only if it passes):** formal H on H_dev2 with tier fixed effects; the normal-residual 10%-risk gap
  1.2816·√2·σ/b rounded up to 0.005, and empirical agreement by |ΔH_dev2| bin.

## 6. Use

- **Pass:** an `htdev2` block in `v2.eval.dev_readout` and "Eval runners" usage. Tracks use it as the human-transfer
  screen in development gates in place of the CSS-pilot three-task mean (non-decrease against the reference within the
  band). The formal paired CSS15 CI still decides human transfer.
- **Fail:** state why and give the best available alternative.

## 7. Compute, storage, rules

- GPU ≤ 1.5 GPU-h: shared-lease jobs `--shared-lease eval-htdev2` on node A GPU0–1; build, scans, scoring and
  validation on CPU in `decision20-train-fast:host2` with `--network none` (downloads on the host).
- Files: `/data/dev2/private/panels/{goldfree,gold}/ht-dev2.*` (gold mode 600), registered in `v2/eval/panels.py`;
  private HF backup in the eval-artifacts dataset. No items, gold or source text in git or the gist.
- Panel items are never training data; the build record lists their source ids by hash so builders can exclude them.
- Any deviation is a dated amendment committed before the affected step runs.
