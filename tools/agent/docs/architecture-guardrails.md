# Architecture guardrails

Architecture checks enforce dependency and generated-artifact constraints.

## Blocking invariants

- declared production dependency boundaries
- new dependency cycles in configured module graphs
- generated/schema/source consistency contracts
- unowned files added at the repository root
- public protocol, configuration, and deployment contracts proven by tests

Existing forbidden edges may be ratcheted against `BASE_REF`; adding to them
fails. Rules live in `tools/agent/structure-rules.yaml` and remain declarative.

## Module design

Module cohesion and interface quality are review judgments. The harness does
not impose file length, function length, nesting, or interface-size thresholds.

Prefer a cohesive deep module with a narrow API over several shallow wrappers.
Extract only when ownership, dependency direction, independent testing, or
review clarity improves. Interfaces belong at external, package, or genuine
multi-implementation seams.

## Router ownership

- signals extract request/response facts
- decisions compose signals into control policy
- algorithms choose among models after a decision
- plugins perform decision-driven processing
- global config owns intentionally cross-cutting behavior

Transport translation, semantic evaluation, storage lifecycle, and deployment
translation remain independently testable. The nearest local `AGENTS.md`
records concrete exceptions and owners for hotspot subtrees.

Open architecture gaps that do not yet have a clear GitHub issue are listed in
[architecture-risks.md](architecture-risks.md).
