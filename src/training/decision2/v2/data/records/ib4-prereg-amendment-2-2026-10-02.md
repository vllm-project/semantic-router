# IB4 prereg amendment 2 (2026-10-02): G4 outcome of phase-1 run a1; two families rebuilt, two removed

Committed and pushed before the rebuilt run's first audit. Seen before this amendment: run a1's (`ae9eeeeb`) audit
receipts up to G4 (counts and G4 accuracies only; no row was read). Run a1 stays on node A as `p1-a1`, unpublished.

**G4 on run a1** (pass-1 TRAIN; gate: `state_removed`, `option_only`, `hypothesis_only` ≤ majority + 0.05):

| Family | Result | Failing view |
| --- | --- | --- |
| `sqa2` | pass | — |
| `isarc2` | pass | — |
| `sentfin3` | **fail** | `hypothesis_only` (entity alone predicts the class) |
| `fc_pick` | **fail** | `hypothesis_only` (candidate name alone predicts the label; amendment 1's overall balance) |
| `w2c_act` | **fail** | `hypothesis_only` (the tool list alone predicts the action) |
| `smish` | **fail** | `state_removed` (constant input; balanced 250 / 250 TRAIN rows, so the excess is fold noise at n = 500, but the rule binds) |

**Decisions.**

1. `smish` and `w2c_act` are removed from phase 1 (prereg §3: a failing family is removed). They are not rebuilt in
   phase 1. `smish` may return in a later phase only with a larger, new construction under its own amendment, never by
   re-running the same rows. `w2c_act` is not retried: G0 already removed most of it (shared tool schemas with Index tool
   rows) and the coordination note says tool data as built did not transfer to When2Call-like decisions.
2. **`sentfin3` is rebuilt with per-entity balance**: cell = normalized entity, equal rows per class inside every entity,
   so the entity alone is at chance by construction (cap 3,000 per class unchanged). The rule of amendment 1 (it replaces
   IB1 `sentfin`) stands.
3. **`fc_pick` returns to the original prereg rule**: balanced per candidate name (amendment 1's overall balance is
   withdrawn). It is smaller but name-neutral by construction.
4. Phase 1 builds `sqa2 isarc2 sentfin3 fc_pick` in a fresh run (`p1`), and every audit (G0, G0u, overlap, names, C1
   names, pass 1, **G4 again for every family**, final, rescan, IX1, leak guard) runs from scratch. A family that fails
   G4 again is removed, with no further rebuild in phase 1.
