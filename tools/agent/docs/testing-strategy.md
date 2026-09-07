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

`make check` is the daily default. It formats and lints changed files, enforces
dependency/generated contracts, and runs the matching domains' unit or static
contract checks from `tools/agent/domains.yaml`.

Use `make verify` for runtime, deployment, provider, storage, or other
integration behavior. It requires an explicit domain/profile so a path
classifier cannot silently choose an expensive or wrong environment. Add or
update E2E when user-visible routing, startup, config, Docker, CLI, API, or
protocol behavior changes; a pure refactor needs focused unit/contract proof.

Use `make ci-full` for high-risk changes or reviewer-requested full parity. It
runs the complete pre-commit and test/build baseline and is intentionally not
part of every local edit loop.

CI path classification is deliberately coarser than developer reasoning:

- a source unit-test-only change runs its owning unit job, not E2E or images;
- an E2E file may select its named profile;
- a reusable workflow change runs that workflow's job;
- Docker validation runs only for image definitions;
- `ci/full` selects the maintained full safety net;
- nightly and release workflows remain independent safety nets.

The aggregate required check is `PR Gate`. Individual jobs may be skipped when
their domain is unaffected; the gate fails if any selected job fails or is
cancelled.

Harness and workflow changes use `make harness-check`, which validates the
registry, reusable workflow contracts, unit tests, and action syntax.
