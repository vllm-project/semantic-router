# Amendment 1 to the PN1 preregistration (research & data Milestone 4, 2026-09-29)

Amends [m4-pn1-prereg-2026-09-29.md](m4-pn1-prereg-2026-09-29.md). Committed and pushed after the first
generation pass and before anything else runs: no second generation pass, no judge, no selection and no rows yet.

## 1. Twin-edit generation in two passes (§8, "a timed preflight on 256 items per model sets the volumes")

- **Pass 1** (commit `b9af24ea1`, node B; GPU3 ja zh ru ar, GPU4 ko es de fr; 0.069 + 0.065 GPU-h including
  loading) scaled itself to 47.8% and 46.0% of the planned seeds (2,864 + 2,756 of 12,000).
  - Its preflight rate (3.2 seeds/s) was taken from a one-seed batch at a first-time padded width. The
    128-seed batches that followed took 5–9 s each (about 20 seeds/s).
  - Pass 1 therefore finished after 247 s and 232 s of its 1,020 s budgets.
- **Pass 2** generates exactly the remaining planned seeds, disjoint by seed id. Model, revision, prompt,
  greedy decoding, the 192-token limit and the edit filters are unchanged.
  - The rate now comes from full batches at an already-seen width, plus a fixed allowance per first-time width.
  - Its budget is what generation has left (0.6 GPU-h minus pass 1).
- Both passes' receipts go into the build manifest. The rows' `editor` tag is the same for both passes.
- The cumulative stop is stricter than §8: no PN1 builder GPU job starts once the recorded jobs reach
  1.05 GPU-h. Validity and the embedding scan keep 0.30 GPU-h.

## 2. Clarifications of §2–4 as implemented (no rule change)

- **`pn-near`, "no shared translation within two link hops":** the two sentences' two-hop neighbourhoods are
  disjoint. A neighbourhood is the sentence plus every sentence within two links, in any language.
- **`pn-hop`:** each pivot gives at most one pair.
  - A pair below bigram Jaccard 0.3 is not kept. No `pn-near` pair can balance it, so the §2 per-stratum
    balance would drop it anyway.
  - Each bin keeps at most as many hop pairs as the language needs natural units (TRAIN plus dev).
- **`pn-near`, spread across the bins:** quotas are per cell (0.05 Jaccard step × length-ratio class ×
  length-quartile class) and equal the hop pairs' counts. The near pairs therefore follow the hop pairs'
  overlap and length distribution.
- **Judge volume:** per language × stratum × label, the first ⌈1.6 × planned units⌉ + 2 rows in seed-hash
  order are judged. Only judged rows can be selected.
- **Balance:** the group targets are the §3 minimum swap shares (for example ja 2,200 swap and 2,200 natural).
  - A group that is short after the filters is not filled from the other group.
  - Within a group, units are split over the strata in proportion to min(yes, no).
- **Dev:** per stratum and label, dev rows are drawn in `sha256("pn1-dev:" + key)` order. Stratum units and
  family quotas are proportional to the planned TRAIN selection.
  - If any family's share then differs from TRAIN by more than 10 points, the draw is repeated with the
    realized TRAIN allocation (at most three passes).
