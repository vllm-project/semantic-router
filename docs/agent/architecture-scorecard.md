# Architecture Scorecard

This scorecard is a maintainer dashboard, not a historical log.

## Current Score

| Area | Score | Owner |
|---|---:|---|
| Release readiness | 70 | PL0033 |
| Session-aware agentic routing | 65 | PL0033 |
| Non-release architecture debt | 70 | PL0032 |

Overall posture: `70/100`.

## Current Release Risks

- The v0.3.0 release contract currently fails because a published Docker image
  is not represented in release notes and upgrade or rollback docs.
- Supported install, docs-site, Helm, and `vllm-sr serve` smoke paths still need
  release-blocker triage against current open bug reports.
- Session-aware agentic routing is the only explicitly retained feature
  direction for v0.3 and still needs a scoped ship-or-defer decision.

## Current Debt Risks

- TD006 remains the broad structural ratchet for current structural exceptions.
- TD016 and TD017 keep fleet-sim from fully sharing the repo lint and structure
  contract.
- TD020 keeps classifier boundaries from being fully clean.
- TD027 keeps fleet-sim optimizer ownership broader than desired.

## Score Movement Rule

Move scores only when current-source evidence changes:

- code landed
- tests or gates passed
- a TD was retired or narrowed
- a release risk was accepted explicitly in PL0033
- a non-release debt item was moved out of current scope with rationale

Do not move scores because a historical plan says work was intended.

## Current Sources

- [PL0033 v0.3 Themis Release Closure](plans/pl-0033-v0-3-themis-release-closure.md)
- [PL0032 Architecture Debt Consolidation](plans/pl-0032-architecture-scorecard-ratchet.md)
- [Tech Debt README](tech-debt/README.md)
- [Maintainer Ops](maintainer-ops.md)
