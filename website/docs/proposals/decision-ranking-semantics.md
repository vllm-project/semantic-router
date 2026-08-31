---
title: Decision Ranking Semantics
description: Separates rule eligibility, policy ordering, and evidence strength so decision ranking does not vary per request.
created: 2026-09-01
status: Proposal
---

> **Status:** Proposal · **Created:** 2026-09-01

## Problem

Decision selection blends three concepts into one number: whether a decision matched,
where policy places it, and how strong its evidence is. #3080 stopped unscored rules
from claiming synthetic certainty, but ordering still depends on which rules a given
request happened to score, so the same configuration can rank differently per request.

## Current ordering

Which branch runs depends on whether any matched decision sets a tier.

| Case | Keys, in order |
| --- | --- |
| any `tier > 0` | tier asc, catch-all last, confidence desc, priority desc, name asc |
| no tier, `strategy: confidence` | catch-all last, confidence desc, priority desc, name asc |
| no tier, `strategy: priority` | priority desc, confidence desc, name asc |

Confidence applies only to a comparable pool: #3080 falls the pool back to priority
when any non-catch-all member is unscored. Pools are tiers under tiered selection and
the whole result set otherwise.

Five properties of that table need an explicit decision.

1. The tiered branch never reads `routing.strategy`, so one setting means two things.
2. The `priority` branch has no catch-all check, unlike the other two, so a
   high-priority catch-all can outrank a real match there.
3. `AND` averages its matched children, so an aggregate falls as a decision gains
   evidence: `mean(.90, .90)` beats `mean(.90, .90, .88)`.
4. `evalOR` keeps only the winning branch's scored flag. An extra matching keyword
   branch can therefore demote a decision from scored to unscored and flip the whole
   pool to priority ordering, even though that match is additional support for it.
5. `signalConfidence` treats an absent key as `1.0` unscored, but a matched
   `conversation` rule writes `1.0` explicitly. A boolean predicate then reports
   maximal evidence marked scored, which the #3080 gate does not catch because that
   gate only reacts to unscored members.

Scores are also not the same quantity. `SignalConfidences` is a flat
`map[string]float64` holding all of:

| Key | Value written |
| --- | --- |
| `domain:` | classifier probability |
| `embedding:` | embedding similarity |
| `reask:` | minimum similarity across a matched streak of turns |
| `conversation:` | `1.0` when the predicate holds, `0` otherwise |
| `projection:` | a projection output, calibrated through a slope |
| generic `type: llm` | model-reported label distribution since #3152 |

`evalAND` means a probability, a similarity, and a constant together.

## Proposal

- Tier stays the hard precedence boundary.
- `routing.strategy` applies inside the selected tier, so one setting means one thing.
- `priority` is deterministic policy ordering, not overridden by evidence the contract
  has not declared comparable.
- An absent score is absent, not `1.0`. #3106 already models a failed signal as
  unknown; a missing one needs the same treatment for a different cause.
- Every leaf is policy or evidence. Keyword rules, `NOT`, predicates, conversation
  predicates, and projection outputs are policy: they gate eligibility and carry no
  weight in the aggregate. A projection restates a decision the operator already made.
- Confidence ranks only evidence that was reported, whose kind is declared comparable,
  and whose aggregate does not move with leaf count or with which `OR` branch matched.
- Otherwise ordering falls back to priority, and the trace records the ranking mode,
  the provenance of the scores considered, why confidence did not apply, and the final
  tie-break.

## Compatibility

`decision.tier` is set by 52 decisions across six shipped configs, but only four tiers
hold more than one decision.

| Pool | Ranks by | Why |
| --- | --- | --- |
| `agent` tier 1 | priority | `local_privacy_policy` carries a `NOT` |
| `agent` tier 3 | priority | every member carries `keyword` or `NOT` |
| `privacy` tier 2 | priority | `local_privacy_policy` carries a `NOT` |
| `multi-objective` / `privacy-first` tier 2 | confidence | both members are scored |

Only the last pool is affected. `omni` matches on a boolean conversation predicate and
reports a scored `1.0`, which outranks the projection score of
`unified_privacy_sensitive_route`. The shipped priorities happen to agree, so nothing
misroutes today, but priority is not what decides: raising the route to priority `900`
against `omni`'s `250` still selects `omni` under both strategies. Treating the
conversation predicate as policy restores the operator's ordering.

The other three pools already fall back to priority and are unaffected.

One caveat on that table. A decision is treated as unscored here when its tree
contains a `keyword`, `NOT`, or predicate leaf. That is exact for `AND`, where
`scored` is the conjunction of its children, but approximate for `OR`, where only the
winning branch's flag propagates. Decisions in `agent` tier 3 contain `OR` branches,
so priority fallback there is the ordinary case rather than a guarantee.

## Delivery

| Slice | Scope | Runtime change |
| --- | --- | --- |
| DR-01 | Document the ordering table and `decision.tier` in the decision tutorial | No |
| DR-02 | Declare a policy or evidence role per leaf and exclude policy leaves from the aggregate | Yes |
| DR-03 | Stop `evalOR` from propagating a policy branch's unscored flag over a scored one | Yes |
| DR-04 | Apply `routing.strategy` inside the selected tier | Yes |
| DR-05 | Sort catch-alls last under `strategy: priority` | Yes |
| DR-06 | Declare score kind and enforce the comparability check in configuration validation | Yes |
| DR-07 | Report ranking mode, provenance, fallback reason, and tie-break in traces | Yes |
| DR-08 | Recipe-conformance probes for a cross-route collision | No |

DR-01 and DR-08 hold against the current behavior regardless of which contract is
adopted. DR-02 carries the only observable ranking change in a shipped config.

## Open questions

- Is score kind declared per signal type or per signal instance?
- Should an absent score reuse the unknown state from #3106, or a weaker marker that
  distinguishes "never reported" from "failed"?
- Should catch-alls sort last under `strategy: priority` as well?

## References

- [Prompt Classification Routing](./prompt-classification-routing)
- [Router Learning](./router-learning-memory-and-adaptations)
- [Decision overview](../tutorials/decision/overview)
