# v0.3 Themis Release Closure

## Goal

- Track only the work that can actually block a v0.3 release.
- Keep the release plan grounded in current source, release gates, reproducible
  bugs, and maintainer-approved product direction.
- Remove roadmap wishes that do not map to a concrete release blocker.

The latest published release is v0.2.0. The current source already carries
v0.3.0 package and crate versions, so this plan is about making v0.3.0
publishable rather than preserving every old roadmap item.

## Scope

Release work:

- v0.3.0 release contract and artifact documentation.
- Current install, docs-site, Helm, and supported `vllm-sr serve` blockers.
- Agentic routing through the session-aware model-switching workstream.
- Triage of current crash or user-visible regressions that could make v0.3
  unsafe to publish.
- Final release validation and release notes.

Current issue anchors:

- [#1956](https://github.com/vllm-project/semantic-router/issues/1956) docs
  site paused.
- [#1962](https://github.com/vllm-project/semantic-router/issues/1962)
  `vllm-sr serve` model factory or embedding initialization failure.
- [#1957](https://github.com/vllm-project/semantic-router/issues/1957) large
  input crash or hang.
- [#1910](https://github.com/vllm-project/semantic-router/issues/1910) Helm
  chart renders invalid deployment resources.
- [#1908](https://github.com/vllm-project/semantic-router/issues/1908)
  operator cannot validate multiple `IntelligentRoute` resources independently.
- [#1897](https://github.com/vllm-project/semantic-router/issues/1897)
  OpenRouter usage parsing breaks dashboard insights.
- [#1753](https://github.com/vllm-project/semantic-router/issues/1753)
  session-aware model switching.
- [#1751](https://github.com/vllm-project/semantic-router/issues/1751)
  multi-turn follow-up routing context.

Out of scope:

- Non-release hotspot and fleet-sim debt owned by
  [PL0032](pl-0032-architecture-scorecard-ratchet.md).
- Security and RBAC hardening. Security bugs can still be triaged as normal
  bugfixes, but new security/RBAC closure is not a v0.3 release track.
- Broad API/config/control-plane contract redesign, dashboard hardening,
  training/eval artifact redesign, runtime-state redesign, and native backend
  parity work unless a current reproducible release blocker proves they are
  needed.
- Daily maintainer sync, report generation, and seed issue creation; these
  belong to [Maintainer Ops](../maintainer-ops.md).
- Plan archaeology.
- Daily issue and PR status snapshots, which are generated under
  `.agent-harness/maintainer/`.

## Exit Criteria

- `make release-check RELEASE_VERSION=0.3.0` passes.
- Supported install, docs-site, Helm, and `vllm-sr serve` smoke paths have no
  open release-blocking regressions.
- Session-aware routing is either shipped in a scoped v0.3 slice or explicitly
  deferred with stale attempts closed or superseded.
- Current crash and user-visible regression reports are fixed, rejected as not
  reproducible, moved out of scope, or accepted as release risks.
- Release notes describe the shipped v0.3.0 artifacts and accepted risks.

## Task List

- [ ] `V030001` Land the harness refresh that defines this release plan.
- [ ] `V030002` Make the v0.3.0 release contract pass.
- [ ] `V030003` Clear supported install and deployment blockers.
- [ ] `V030004` Decide and land the agentic session-aware routing slice.
- [ ] `V030005` Triage current crash and user-visible regressions.
- [ ] `V030006` Produce the final release readiness result.

## Task Details

### V030001: Land Harness Refresh

Do:

- merge or supersede
  [#1972](https://github.com/vllm-project/semantic-router/pull/1972)
- ensure the repo-native maintainer ops and current plan model are on `main`

Done when: this plan and maintainer ops run from the default branch.

### V030002: Release Contract

Do:

- run `make release-check RELEASE_VERSION=0.3.0`
- fix the current release-contract failure for the `anthropic-shim` image in
  release notes and upgrade or rollback docs
- rerun the release check after the documentation and workflow references are
  aligned

Done when: the release contract check passes for `0.3.0`.

### V030003: Install And Deployment Blockers

Do:

- resolve the docs-site availability issue in #1956 or accept it as an
  external release risk
- fix the Helm values rendering bug in #1910
- reproduce #1962 on a supported Linux path and fix it, or close it as
  unreproducible or environment-specific
- decide whether Windows `vllm-sr serve` in #1949 is supported in v0.3; fix it
  only if it is supported, otherwise document the support boundary

Done when: supported install and deployment smoke paths are not blocked by
those reports.

### V030004: Session-Aware Agentic Routing

Do:

- choose the v0.3 scope for #1753 and #1751
- ship the smallest useful multi-turn routing behavior, or defer the feature
  explicitly
- close, rebase, or supersede stale session-routing PRs so the PR queue reflects
  the decision

Done when: session-aware routing is either in v0.3 with tests or explicitly out
of v0.3.

### V030005: Crash And Regression Triage

Do:

- reproduce or reject #1957 before release
- decide whether #1908 blocks the supported v0.3 operator story
- decide whether #1897 blocks the supported v0.3 dashboard insights story
- move non-blocking items out of the release milestone instead of keeping them
  as vague release work

Done when: each item is fixed, scoped out, or recorded as an accepted risk.

### V030006: Final Release Readiness

Do:

- run `make release-check RELEASE_VERSION=0.3.0`
- run the repo validation gates required for the final release candidate
- update scorecard evidence and maintainer release readiness output
- assemble release notes from merged PRs, closed issues, and accepted risks

Done when: v0.3.0 can be tagged and published without unresolved release
blockers.

## Next Action

- Start with `V030002`, because the release contract check already fails on the
  current source and has a concrete fix.

## Operating Rules

- One release means one release plan, but the plan is not a roadmap archive.
- Do not keep a task because a previous roadmap mentioned it.
- A task belongs here only when it blocks publishing v0.3.0 or when the
  maintainer explicitly chooses it as the release direction.
- Technical debt enters the release only when current source or a reproducible
  issue proves it blocks the release.
- Daily GitHub state is generated under `.agent-harness/maintainer/`, not stored
  as canonical repo documentation.

## Related Docs

- [Tech Debt README](../tech-debt/README.md)
- [Maintainer Ops](../maintainer-ops.md)
- [Themis roadmap blog](../../../website/blog/2026-03-12-v0-3-themis-roadmap.md)
