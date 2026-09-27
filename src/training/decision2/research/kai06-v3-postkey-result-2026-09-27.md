# 0.6B prospective v3 same-panel result

**Disposition: HOLD for the tested Kai2 package.** Its composite score is lower
than its same-size Kai1 parent. This is a post-key prospective comparison: the
project had already used the same FINAL labels for an earlier size. The 0.6B
roster, adapter amendment and all prediction hashes were fixed before any
0.6B scoring; these numbers are not a never-unsealed blind test.

## Frozen panel and method

- JevArena v3: typed FINAL 1,600 items / 2,000 scored answers plus 15 human-label
  tasks / 6,547 items. `T` is the four-family mean accuracy; `H` is the median
  task macro-F1; composite is `100 × sqrt(T × H)`.
- Public JevBench: 231 available questions, separate from the v3 composite.
  This is an independent public-subset comparison, not an official sealed rank.
- Same prompts, canonical native adapters, full-denominator invalid policy, and
  same scorer versions were used for all three models. The 0.6B peer roster
  includes our Kai1, its direct Kai2 continuation, and pinned Bosun v3.1.
- Roster SHA-256: `e7c1599a43f7a7a09391934f3a8a2c4ddd1bebf0ca78ce6ba59353e7c56e7c99`.
  Prediction seal SHA-256: `2e6333049ff51ceeb79463702dad0ecc88d389bc0daf2970565c7f399704fcc6`.
  Aggregate rank report SHA-256: `5fa9ff67eb8a3e29c118ba4dda18828362b487b974a7576bdc96c35cf7c1aeb9`.
  Paired CI report SHA-256: `fa2727e3dca664e78db78c10bddfd0f2d4e54f247c3703369576e7688784ee74`.

| Model | Loaded parameters | T | H | v3 score | Public 231 | Public valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bosun v3.1 | 606,131,200 | 0.43375 | 0.34216 | **38.5243** | 133/231 | 231/231 |
| Kai1 | 571,909,635 | 0.36188 | 0.35691 | 35.9383 | 114/231 | 187/231 |
| Kai2 clean-v2 1 epoch | 571,909,635 | 0.34938 | 0.35243 | 35.0897 | 117/231 | 187/231 |

The three-model v3 rank is Bosun, Kai1, Kai2. Kai2 is dominated by Kai1 at
the same loaded parameter count. Its composite change against Kai1 is
**−0.8486 points**, with a paired 95% bootstrap interval of **[−2.8734,
+1.6564]** (5,000 draws; independent typed groups and CSS task/item hierarchy).
The interval includes zero, and the point estimate fails the first-release
improvement target. A 3-question public-subset gain does not reverse the v3
finding.

## Capacity and source sensitivities

- Kai2 improves typed Score accuracy from 0.2450 to 0.3200, while Choice falls
  from 0.3463 to 0.3163 and Noul from 0.5050 to 0.4725. Bosun reaches 0.4563,
  0.6763 and 0.2075 respectively, so it also shows a Score tradeoff.
- Both Kai models return 404 invalid answers on the 6,547 CSS inputs and 44 on
  the 231 public inputs due to their native context capacity. Those remain in
  the denominator. The Bosun package returns valid answers for all these
  items. No input was shortened or candidates removed.
- Kai2 training contains 120 official FLUTE-train examples; the 15-task panel
  contains 500 distinct FLUTE-test items. This measures supervised same-dataset
  transfer for FLUTE. Median task macro-F1 without FLUTE is 0.35354 for Kai2,
  0.35721 for Kai1 and 0.33536 for Bosun; removing FLUTE does not change the
  Kai2-versus-Kai1 direction.
- A gold-free audit of Kai2's 6,262 actual training records against the three
  panels found zero shared IDs, raw or normalized exact states, or matches under
  the frozen near-duplicate rule. Kai1 has an inherited legacy-corpus audit
  flagging 42 near CSS rows; this remains a transfer-evidence limitation for
  both Kai-derived models. Approximate near matching does not prove complete
  semantic disjointness.

## Run incident and next test

The first Kai1/Kai2 collectors stopped when a complete question and candidates
left no room for state. No 0.6B scores were computed from these partial files.
The adapter was patched to count only that exact native exception as an invalid
answer, an amended lock was committed, and both Kai models were rerun from new
empty output files. Bosun's complete, unchanged predictions were carried
forward. All nine final files were hashed and sealed before scoring. Total GPU
time, including both failed collections, was approximately 0.264 GPU-hours.

The most discriminating next controlled experiment is an official Qwen3-0.6B
initialization with the same audited training rows and token budget, after a
zero-step three-type native-contract check. The comparator suggests that the
present Kai capacity and Choice/Noul behavior are limiting, but cannot isolate
architecture from Bosun's data. Existing Kai human-source arms have lower typed
DEV accuracy than clean-v2 and do not yet justify full-panel reuse. Select any
new candidate only with SELECT/DEV and seek fresh independent confirmation
before a release claim.
