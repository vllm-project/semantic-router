# 9B Milestone 4 preregistration, amendment 3: rule implementation, two-seed fallback, line builds

Frozen 2026-09-29 ~00:35 UTC, before any K, P or KN development result was read. K-s1 and P-s1
predictions exist on node A but have not been scored. Nothing here depends on a result.

## Rule implementation

`lux9b/m4_rules.py` (tests `tests/test_m4_rules.py`) implements the preregistered rules on
`v2.dec.dev_readout` JSON (`lux9b/m4/score.sh`):

- **α rule.** The floors and gains are computed exactly, in rationals from the correct / n
  counts. An α that sits exactly on a floor is eligible ("below" means strictly below).
- **Proxy drop rule.** P is the readout's `proxy` = 100·√(T·H), where H is the median CSS-pilot
  task macro-F1 (the preregistration's "H_pilot median"). H3 is `H_mean`. A line pick is dropped
  if its P is ≥ 8 below the best line pick's P.
- **Check.** On the Milestone 4 D-line readout (`m4/readout-dline/readout.json`) the code
  reproduces α\*_D = ½. The per-α results: α = 1 fails the Choice floor (759 < 775), and its
  transition-table family is exactly on its floor. G\* = .0819, so 0.75·G\* = .0614, which ⅓
  (.0587) misses.

## Seed rule for two-seed arms

The coordinator rule is: soup if its P is ≥ the seed mean, otherwise the median seed. For the
two-seed arms P and KN the fallback is the **primary seed (`-s1`, seed 20260926)**, following
Milestone 3 amendment 3's two-seed convention. K (three seeds) uses the median seed by P.

## Artifacts and lines

All builds run on node A: soups on CPU, readouts on GPU6 / GPU7. The Lux end point is the
checkpoint-form Lux of the D line (`m3/pf-D-s1-zero/run/checkpoint-0000000`).

- **Seed artifacts.** Each seed artifact is its SELECT-chosen checkpoint (`run/BEST.json`). The
  arm soups are `soup.sh` over the seed checkpoints (K: K-s1..s3; P: P-s1, P-s2; KN: KN-s1,
  KN-s2), each followed by its readout. That soup readout is α = 1 of the arm's line when the
  soup is the artifact.
- **K / KN / P lines.** `interp.sh` from the arm artifact: K ¼, ⅓, ½, ⅔; KN ½; P ½. α = 1 is the
  artifact's own readout.
- **U line.** θ_U(α) = α·(½·D soup + ½·K artifact) + (1 − α)·Lux, built by `soup.sh` with
  repeated members:

  | α | D soup | K artifact | Lux |
  | --- | ---: | ---: | ---: |
  | ¼ | 1 | 1 | 6 |
  | ⅓ | 1 | 1 | 4 |
  | ½ | 1 | 1 | 2 |
  | ⅔ | 1 | 1 | 1 |

- **Node-B seeds (amendment 2).** They are re-read on node A with `readout.sh` unchanged, as
  `<run>-A` (`-A-cal`, `-A-dev`, `-A-css-pilot`). This refits the CAL698 temperatures on node A,
  which supersedes amendment 2's "with the node-B temperatures". The argmax metrics that the
  rules use are unaffected either way.
- **Readout reference.** All lines read against the M4 Lux reference `lux-16k`.

## Contrasts and HT-DEV clause

- **Seed-paired contrasts (reported only; never selection).** K − P uses the pairs (K-s1, P-s1)
  and (K-s2, P-s2). K − KN uses (K-s1, KN-s1) and (K-s2, KN-s2). The arm-level versions
  compare artifacts.
- **HT-DEV clause.** The preregistration's HT-DEV clause is unchanged. COORDINATION
  "Eval runners" is checked immediately before the α readouts are scored. If HT-DEV v1 is
  published there by then, a further amendment fixes its aggregate and read procedure before
  any HT-DEV readout is made.
