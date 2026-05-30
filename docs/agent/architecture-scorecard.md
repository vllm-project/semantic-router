# Architecture Scorecard

This scorecard is a maintainer dashboard, not a historical log.

## Current Score

| Area | Score | Owner |
|---|---:|---|
| Release contract coherence | 72 | PL0033 |
| Runtime state and lifecycle | 74 | PL0033 |
| Control-plane boundaries | 70 | PL0033 |
| Product hardening and dashboard seams | 68 | PL0033 |
| Native/runtime parity | 66 | PL0033 |
| Non-release architecture debt | 70 | PL0032 |

Overall posture: `70/100`.

## Current Release Risks

- The v0.3 release scope is broad and must be governed through one milestone
  plan instead of scattered historical workstreams.
- Runtime/control-plane boundaries still carry release risk through TD031,
  TD034, TD037, and TD039.
- Dashboard and eval hardening still need tighter release issue ownership
  through TD030 and TD032.
- Native/runtime parity is still release-relevant through TD033.

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
