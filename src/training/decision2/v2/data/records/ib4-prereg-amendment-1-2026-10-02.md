# IB4 prereg amendment 1 (2026-10-02): replacement dedupe, `fc_pick` balance, Arabic column fix

Committed and pushed **before any IB4 audit** (no G0, overlap, G4 or IX1 step had run) and before any IB4 row was read.
Seen before this amendment: the first phase-1 build's aggregate per-family counts only (`cand/build.json` of build
`ba7c08af0`, moved aside on node A as `p1-superseded-a0`; never published).

1. **`sentfin3` replaces IB1 `sentfin`, so it is not deduplicated against it.** The §2.0 dedupe removed nearly every
   SEntFiN (headline, entity) pair, since IB1 `sentfin` used the same pairs with two classes. The coordination rule asks
   future arms to drop IB1 `sentfin` or rebuild it entity-level; `sentfin3` is that rebuild. Rule now: a family listed
   in `REPLACES` (`sentfin3` → IB1 `sentfin`) skips the dedupe against the family it replaces only; the dedupe against
   every other IB1–IB3 row stays. The state overlap with the replaced family is counted
   (`replaced_family_state_overlap`) and **arms that mix `sentfin3` must drop IB1 `sentfin`** (announced with phase 1).
2. **`fc_pick` is balanced overall, not per candidate name.** Per-name cells kept about one row in eight, because most
   function names occur once. The name-shortcut check moves to G4: the `candidate`-only view must stay within majority +
   0.05, or the family is removed (as for every family). Each row records `candidate_name` in its audit metadata.
3. **Bug fix, no rule change:** the Arabic iSarcasmEval train file names its text column `text`, not `tweet`; the first
   build dropped every Arabic row. Both languages are read as §1.2 says.
4. **Clarification, no rule change:** the When2Call train SFT file holds no tool-call targets (its targets are ask /
   decline responses); `w2c_act`'s call class comes from the preference split's chosen responses, as §2.1 allows.
   `isarc2` tweets that IB1 `isarc` used in its different pairwise task (same label) stay, disclosed under §2.0's
   leaf-overlap count.
