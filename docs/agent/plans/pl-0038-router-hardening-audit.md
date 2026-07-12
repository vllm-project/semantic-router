# PL-0038 Router Hardening Audit

## Goal

Complete issue #2375 as a bounded audit and hardening track: publish a
maintainer-readable inventory of security, resource-lifecycle, API-contract,
module-boundary, and performance risks; map every high-priority risk class to
durable follow-up ownership; and land one high-confidence proof-of-fix for the
Looper internal-request trust boundary. Close the audit-time dashboard password
lifecycle and local feature-gate integrity defects that otherwise prevent the
proof from being operated and validated safely.

## Scope

- Router config, extproc, runtime/service composition, selection/model runtime,
  Looper, memory/learning, vector/cache/replay stores, API/header contracts,
  CLI, dashboard, operator/CRDs, native bindings, training interfaces, and E2E.
- Issue and pull-request lineage under #2375, #2357, and #1443.
- Static, race, fault, fuzz, dependency, performance, and deployment checks
  needed to make the findings enforceable.
- CPU-local validation, AMD deployment regression, and normal pull-request CI.
- Dashboard compromised-password recovery, browser password-manager semantics,
  and NIST-aligned password establishment and verification controls.

Out of scope:

- implementing every remediation in one pull request;
- broad rewrites of router core or control-plane modules;
- publishing unnecessary reproduction detail for security-sensitive findings;
- unrelated cleanup discovered during the audit.

## Exit Criteria

- The public audit covers every subsystem and inventory required by #2375.
- Every high-priority risk class is fixed here, linked to an existing issue
  with current evidence, or assigned to a new child issue.
- Indexed technical debt records architecture gaps that remain after this
  change.
- The #1443 proof-of-fix authenticates internal Looper calls, rejects forged
  markers, preserves the credential across hot reload, and has unit plus E2E
  regression coverage.
- Repo-native lint, CI, feature, PR, and affected E2E gates pass.
- AMD deployment regression passes with Looper positive/negative coverage and
  non-sensitive evidence.
- The pull request is green and #2375 receives a final closure summary.

## Task List

- [x] `RHA001` Establish a clean latest-main worktree and gitignored evidence
  ledger without disturbing the pre-existing checkout.
- [x] `RHA002` Complete security/API, resource/runtime, control-contract, and
  cross-repository audit lanes.
- [x] `RHA003` Prepare, index, and publish the audit, closure map, and gate
  recommendations as repository-tracked public artifacts.
- [x] `RHA004` Create or update durable child issues and technical debt for
  every high-priority risk class.
- [ ] `RHA005` Complete the #1443 Looper authentication fix and close final
  deployment-secret, malformed-envelope, and no-leak review findings.
- [ ] `RHA006` Close the dashboard password lifecycle and local feature-gate
  integrity defects, then pass the local validation ladder and targeted
  regression tests.
- [ ] `RHA007` Pass AMD deployment and affected E2E regression.
- [ ] `RHA008` Submit a signed-off pull request and keep CI green.
- [ ] `RHA009` Post the closure summary and close #2375 when its acceptance
  criteria are satisfied.

## Next Action

Close the final review findings, then pass CPU feature, affected-E2E, local PR,
and AMD gates. Submit the signed-off proof pull request, keep CI green, and
attach the final non-sensitive closure receipts.

## Operating Rules

- Keep local commands, raw logs, and changing GitHub state under
  `.agent-harness/issue-2375/`.
- Public artifacts must not contain credentials, private host details, local
  paths, or tool attribution.
- Treat issue creation and comments as reviewed apply steps.
- Keep this pull request focused on the audit and one proof-of-fix; follow-up
  implementations use their own affected-file reports and gates.
- A failed gate is an iteration point, not a handoff point.

## Related Docs

- [Issue #2375](https://github.com/vllm-project/semantic-router/issues/2375)
- [Router quality and security audit](../../architecture/router-quality-security-audit.md)
- [Security hardening guide](../../architecture/security-hardening.md)
- [Architecture guardrails](../architecture-guardrails.md)
- [Testing strategy](../testing-strategy.md)
- [Technical debt inventory](../tech-debt/README.md)
