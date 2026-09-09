# Testing strategy

Use the cheapest deterministic evidence that can falsify the change, then add
integration evidence only where a boundary is actually crossed.

```bash
make impact CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
make verify DOMAIN=<domain>
make verify PROFILE=<profile>
make ci-full
```

`make check` checks formatting and lints changed files without editing source,
and enforces dependency/generated contracts. Use `make fmt` to apply formatting.
Choose focused unit or contract targets based on the behavior being changed;
`make impact` lists related commands from `tools/agent/domains.yaml`.

Use `make verify` for runtime, deployment, provider, storage, or other
integration behavior. It requires an explicit domain/profile so a path
classifier cannot silently choose an expensive or wrong environment. Behavior
changes need tests that prove them. Use integration or E2E when correctness
depends on interaction across components. A pure refactor does not require
new E2E coverage.

Use `make ci-full` for the containerized quality and local core test/build
baseline. It is intentionally outside the daily edit loop and does not cover
hardware-specific jobs or release qualification.

CI path classification is deliberately coarser than developer reasoning:

- a source unit-test-only change runs its owning unit job, not E2E or images;
- an E2E file may select its named profile;
- a reusable workflow change runs that workflow's job;
- Docker validation runs only for image definitions;
- `ci/full` selects the maintained full safety net;
- nightly and release workflows remain independent safety nets.

The aggregate required check is `PR Gate`. Individual jobs may be skipped when
their domain is unaffected; the gate fails if any selected job is missing,
skipped, cancelled, or failed.

Harness and workflow changes use `make harness-check`, which validates the
registry, reusable workflow contracts, harness/resource tests, and action syntax.

## CI ownership

Quality runs static checks and repository contracts. Component jobs own their
unit tests, generated API/CRD checks, compilation, and website builds. Python CLI,
fleet, and training contracts share a lightweight workflow; CLI integration
already includes its unit suite. Integration jobs own live stacks and profiles.
The path registry selects these jobs; it does not prescribe a local work loop.

Nightly publishers wait for all scheduled validation jobs. A release tag must
point to the exact commit of a successful Main push run. Merge a candidate into
main and finish Main validation before tagging; if its Main run failed, rerun
that run successfully before retrying Release. Manual Release dispatch remains
a version-contract dry run and does not publish. Shared image tags advance by
source ancestry so an older run cannot overwrite a newer qualified source.
