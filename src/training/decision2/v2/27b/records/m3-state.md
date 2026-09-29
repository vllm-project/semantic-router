# ~27B M3 state (resume file)

Updated: 2026-09-29 16:05 UTC+8 (F2-completion worker; **M3 complete, F1 stays**)
Branch: `xunzhuo/decision-2-training-27b` (merge-only into `xunzhuo/decision-2-training`)
Records: `m3-results-2026-09-29.md` (F1, F2, the rule, mlx-diag, incident), `m3-prereg-2026-09-29.md` (amendment 4),
`m4-proposal-2026-09-29.md` (not approved, not run).

## Outcome

- **F2 = M3-S soup.** Post-key v3 64.465. SEAL `cb7609ed…`; scored run node B `/data/dev2/runs/27b/M3-S-soup/formal`.
- **F2 − F1 = −2.74 (−5.86, +0.38)**, and F2 is below the 64.92 bar, so under the coordinator's rule **F1 stays** the ~27B
  release candidate. F2 is attribution only: nothing staged, nothing on HF.
- **A7 effect (F1 − F2):**
  - T +.099 (+.081, +.118), so the claim holds.
  - Constraint competition +.188, exception stack +.200.
  - v3 +2.74 (n.s.); H −.030 (n.s.).
- **F2 gates:**
  - H is not below any peer.
  - Types OK: Choice .691, Noul .794, Score .783 (leans on level 4: 169 vs 128 gold).
  - Overlap exposure: none.
  - Fails only v3 ≥ 64.92.
- **mlx-diag:** F1 .828 and F2 .826 (type-macro); per language in the results record.
- **GPU-h:** this worker 1.26; Milestone 3 total 21.08 of 32.

## Chain-rule check (14:25 rule)

- Every uploaded driver was checked against its local SHA-256 and size. The running copies in `m3-logs/` equal the
  mirrored commit `97b64aa65` (`v2/27b/m3f2/`).
- Every job was confirmed started (process or container, and its first log line) and finished with exit 0.
- Remote `pgrep` uses the `[f]2-…` pattern so that it does not match its own ssh shell.
- No automatic chain is left running. Node B GPU5 and GPU7 leases were set back to reserved-idle at 07:50Z; GPU6 was
  never used.

## Next (for the coordinator)

- F1 release engineering continues as planned; C1 event 3 uses F1.
- The M4 proposal (≈ 70 GPU-h) needs approval and an eval leave-family-out development check.

## Paths (node B)

- Runs: `/data/dev2/runs/27b/{M3-S-s2,M3-S-soup,m3-contrast,m3-f2/{f1-scored-cache,gates,overlap,mlx-diag}}`.
- Driver logs: `/data/dev2/runs/27b/m3-logs/f2-*`, `M3-S-soup.driver.log`.
- mlx-diag scores: node A, the same `m3-f2/mlx-diag/<NAME>/mlx-diag.score.json`.
