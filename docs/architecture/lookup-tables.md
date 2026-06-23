# Legacy Lookup Tables and Router Learning Experience

This document records the old lookup-table design as migration material. Lookup
tables are no longer a public router configuration concept. New work should use
the Router Learning vocabulary:

| Layer | Role |
| --- | --- |
| Router Replay | Durable event log for route, response, feedback, and diagnostics records. |
| Router Learning states | Low-latency online state owned by enabled adaptations. |
| Router Learning experience | Materialized historical evidence derived from replay, evals, or operator overrides. |
| Router Learning adaptations | Policies that adjust a base route using states and experience. |

The old public shape:

```yaml
global:
  router:
    model_selection:
      lookup_tables: ...
```

must not be used in new configs. Runtime loading rejects it with an actionable
error. Experience materializers may reuse the implementation ideas below, but
they should not expose `lookup_tables`, `storage_path`, `auto_save_interval`, or
similar selector-local storage knobs as public API.

## Useful Migration Material

The old implementation modeled three kinds of materialized evidence:

| Legacy table | Future experience view | Purpose |
| --- | --- | --- |
| `quality_gap` | `quality_gap` | Estimated quality delta between models for a task, domain, or decision scope. |
| `handoff_penalty` | `handoff_penalty` | Estimated cost or quality loss when switching models. |
| `remaining_turn_prior` | `remaining_turn_estimate` | Expected remaining turns in a session or conversation. |

These remain useful facts, but they belong in Router Learning experience:

```text
Router Replay records
  -> async materializer or eval import
  -> versioned experience snapshot
  -> local in-memory request-time read
  -> adaptation diagnostics: used | missing | stale | ignored
```

Request routing should not synchronously query durable storage for these facts.
If a snapshot is missing or stale, the adaptation must fail open and record the
condition in Router Replay diagnostics.

## Migration Rules

- Keep Router Replay as the event source.
- Keep request-time reads local and bounded.
- Keep manual overrides as future experience overrides, not selector-local
  tables.
- Keep sample count, source, version, freshness, and applicable scope on each
  experience entry.
- Do not reintroduce `model_selection.lookup_tables` or
  `router.learning.experience.enabled` as broad user-facing toggles.

## Related Design

- [Router Learning Memory and Adaptations](../../website/docs/proposals/router-learning-memory-and-adaptations.md)
- [Learning: Memory and Replay](../../website/docs/tutorials/learning/memory-and-replay.md)
- [Learning: Experience](../../website/docs/tutorials/learning/experience.md)
