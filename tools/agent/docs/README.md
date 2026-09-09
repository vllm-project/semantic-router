# Development harness

The harness supplies deterministic repository facts and checks. It does not
classify developer intent, choose a skill, manage a work loop, or define when
an agent is finished.

## Daily flow

```bash
make impact ENV=cpu CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
make fmt CHANGED_FILES="path/one path/two"  # explicitly apply formatting
make verify DOMAIN=<domain>       # only when integration evidence is needed
make verify PROFILE=<profile>     # explicit E2E
make ci-full                      # local quality and core test/build baseline
```

`impact` reports changed paths, matching ownership domains, related checks,
candidate CI jobs/images/profiles, and available host tools. The single source
for those mappings is [domains.yaml](../domains.yaml).

`check` checks changed-file formatting/lint and real dependency/generated
contracts without editing source. `fmt` applies formatting explicitly. Choose
the tests that prove the change using the existing domain targets; CI invokes
those tests in their owning jobs. `verify` requires a configured integration
domain or E2E profile. `ci-full` runs containerized quality and the local core
test/build baseline; hardware jobs and release qualification remain separate.

Use `make harness-check` for changes to harness code, workflows, the domain
registry, or agent instructions.

## Durable references

- [change-surfaces.md](change-surfaces.md): cross-layer product contracts
- [environments.md](environments.md): supported local and CI environments
- [architecture-guardrails.md](architecture-guardrails.md): dependency and generated-artifact constraints
- [architecture-risks.md](architecture-risks.md): compact repository-only risk index
- [maintainer-ops.md](maintainer-ops.md): reviewed GitHub and release operations
- [openai-api-contracts.md](openai-api-contracts.md): protocol translation contracts
- nearest local `AGENTS.md`: non-obvious subtree constraints

The executable pieces are `domains.yaml`, `structure-rules.yaml`,
`maintainer-policy.yaml`, `tools/agent/scripts/`, `tools/ci/`, and
`tools/make/agent.mk`.
