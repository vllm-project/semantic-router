# HR2 preregistration — amendment 1: repeated upstream samples (2026-09-30)

Committed before any HR2 candidate file exists. The first build attempt (`25ec87a8d`) stopped at its duplicate-id
check and wrote nothing. Cause: upstream samples repeat — HelpSteer3 lists some samples more than once (sometimes
with the two responses in the other order) and PRM800K phase 2 has the same pre-generated solution rated by several
labelers. §2's tie / low-agreement rule is extended to these repeats; nothing else changes.

1. **One key per upstream item.** HelpSteer3 sample key = hash of the context plus the two responses in sorted order
   (independent of which is listed first). `hs3_help` picks its response by text
   (`min` of `hash("hr2-help-v1:" + key + response)`), not by position.
2. **All copies must agree.** An item is kept only if every copy passes the family's rules with the same verdict:
   `hs3_pref` — every copy strict with the same preferred response; `hs3_help` — every copy parses with max − min ≤ 1
   and the same median; `prm_step` — every rating any labeler gave that exact state (problem, previous steps, step,
   including 0; the chosen and the alternative completions at every walked step) is the same; other families — same
   label for the same content key. Any disagreement drops **every** copy (reported as conflicting duplicates).
   Agreeing copies keep one row.
3. The build applies the same rule once more across families (same `id` or same `input_sha256` with different labels
   → all copies dropped).
4. IndoNLI's content key is the hash of premise and hypothesis (instead of `pair_id`), so repeated pairs are caught.
