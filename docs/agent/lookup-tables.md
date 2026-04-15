# Lookup Tables for Session-Aware Routing

Lookup tables replace hardcoded constants in the model-selection pipeline with
**data-driven values** derived from router replay records and operator
configuration. They are the mechanism through which empirical observations
(e.g. "how much better is model-large than model-small for coding tasks?")
feed back into real-time routing decisions.

This document covers:

1. [Concepts](#concepts) — what each table stores and why
2. [Configuration](#configuration) — YAML reference
3. [Lifecycle](#lifecycle) — initialization, population, persistence
4. [Consumption](#consumption) — how selectors read the tables at request time
5. [Maintenance](#maintenance) — overrides, re-derivation, manual entries
6. [Examples](#examples) — end-to-end walkthroughs

---

## Concepts

Three logical table types live inside one unified key-value store:

| Table | Key format | Value | Purpose |
|---|---|---|---|
| `quality_gap` | `quality_gap::<task_family>::<current>::<candidate>` | float64 | Quality difference between two models for a task family |
| `handoff_penalty` | `handoff_penalty::<from>::<to>` | float64 | Cost of switching models mid-session |
| `remaining_turn_prior` | `remaining_turn_prior::<intent_or_domain>` | float64 | Expected remaining turns after the first turn |

### quality_gap

Stores the average confidence-score difference `avgScore(candidate) - avgScore(current)` observed in replay records for a given task family.

**Example:** `quality_gap::coding::small::large = 0.20` means model-large
historically scores 0.20 higher than model-small on coding tasks. The
HybridSelector uses this to decide whether upgrading from the current model to a
candidate is worthwhile: if the gap exceeds the configured threshold, escalation
is triggered.

### handoff_penalty

Captures the cost delta when the router switches from one model to another
within a single session. The value is an Exponential Moving Average (EMA, alpha
= 0.2) of `cost(new_turn) - cost(previous_turn)` across all observed switches.

**Example:** `handoff_penalty::small::large = 0.02` means switching from
model-small to model-large mid-session typically costs an extra 0.02 per turn.
This penalizes unnecessary context switching within active sessions.

### remaining_turn_prior

Estimates how many turns remain after the first turn of a session in a given
domain. Stored as `average_session_length - 1`. A policy reading this at turn N
should subtract (N - 1) to get the remaining-turns estimate for that turn.

**Example:** `remaining_turn_prior::support = 3.5` means support sessions
average 4.5 turns total, so at the first turn the router expects 3.5 more turns.
This helps the router balance cost efficiency vs. quality over the expected
session lifetime.

---

## Configuration

All lookup table settings live under `model_selection.lookup_tables` in the
router YAML configuration:

```yaml
model_selection:
  lookup_tables:
    # Master switch — set to true to activate lookup table resolution.
    enabled: true

    # Path to the JSON persistence file.
    # When empty, an in-memory backend is used (data lost on restart).
    storage_path: "/var/lib/semantic-router/lookup_tables.json"

    # Background flush interval for dirty entries (e.g. "5m").
    # Only meaningful when storage_path is set.
    auto_save_interval: "5m"

    # Derive entries automatically from router replay records.
    # Requires a replay store to be configured.
    populate_from_replay: true

    # Re-derive entries on this interval (e.g. "15m").
    # When empty, derivation runs only once at startup.
    populate_interval: "15m"

    # ── Manual overrides ──────────────────────────────────────────────
    # Override values always take precedence over replay-derived values.

    quality_gaps:
      - task_family: coding
        current_model: small
        candidate_model: large
        value: 0.15

    handoff_penalties:
      - from_model: small
        to_model: large
        value: 0.02

    remaining_turn_priors:
      - intent_or_domain: support
        value: 3.5
```

---

## Lifecycle

### Initialization

On startup, `buildLookupTable()` in
[router_selection.go](src/semantic-router/pkg/extproc/router_selection.go)
runs the following steps:

```
1. Create storage backend
   ├── storage_path set?  → FileStorage  (atomic JSON writes)
   └── empty?             → MemoryStorage (in-process only)

2. Load persisted state
   └── FileStorage.Load() reads existing JSON file into memory

3. Start auto-save (if configured)
   └── Background goroutine flushes dirty entries on interval

4. Populate from replay (if enabled)
   └── Async: fetch all records → Builder.PopulateFromRecords()
   └── If populate_interval set: start periodic re-derivation goroutine

5. Apply config overrides
   └── Manual entries written with Source = "config_override"
   └── These are never overwritten by replay derivation
```

### Shutdown

The returned `cancel` function:

- Stops the periodic populator goroutine
- Calls `FileStorage.Close()`, which stops auto-save and performs a final flush

### Persistence format

FileStorage writes atomically (`write .tmp` → `rename`) with this JSON schema:

```json
{
  "version": 1,
  "last_updated": "2025-04-14T10:30:00Z",
  "entries": {
    "quality_gap::coding::small::large": {
      "value": 0.2,
      "source": "replay_derived",
      "updated_at": "2025-04-14T10:30:00Z",
      "sample_count": 50,
      "aggregation_window": "7d"
    },
    "handoff_penalty::small::large": {
      "value": 0.02,
      "source": "config_override",
      "updated_at": "2025-04-14T10:00:00Z"
    }
  }
}
```

If the file is corrupted on load, a `.corrupted` backup is created and the
error is surfaced in logs.

---

## Consumption

### HybridSelector integration

The `HybridSelector` is the primary consumer. The factory attaches the lookup
table during creation:

```go
factory := selection.NewFactory(cfg).
    WithLookupTable(lt)
```

At request time, `resolveQualityGap` checks the table first, then falls back to
the static config threshold:

```go
func (h *HybridSelector) resolveQualityGap(taskFamily, currentModel, candidateModel string) float64 {
    if h.lookupTable != nil && taskFamily != "" {
        if gap, ok := h.lookupTable.QualityGap(taskFamily, currentModel, candidateModel); ok {
            return gap
        }
    }
    return h.config.QualityGapThreshold // fallback
}
```

This pattern — **table lookup with config fallback** — is the recommended way
for any selector to consume lookup table values. It provides graceful
degradation: if the table has no data for a key (cold start, new model, etc.),
the system falls back to operator-configured defaults.

### Read API

The `LookupTable` interface exposes three convenience accessors plus a generic
`Get`:

```go
type LookupTable interface {
    Get(key Key) (Entry, bool)
    QualityGap(taskFamily, currentModel, candidateModel string) (float64, bool)
    HandoffPenalty(fromModel, toModel string) (float64, bool)
    RemainingTurnPrior(intentOrDomain string) (float64, bool)
}
```

All methods are safe for concurrent reads and return `(value, found)` tuples.

---

## Maintenance

### Automatic derivation from replay records

When `populate_from_replay: true`, the `Builder` scans all records in the
replay store and derives entries using three algorithms:

| Table | Algorithm |
|---|---|
| `quality_gap` | Group by category + model → average confidence score → pairwise difference |
| `handoff_penalty` | Detect model switches in pseudo-sessions (same Decision, <30 min gap) → EMA of cost deltas |
| `remaining_turn_prior` | Count turns per pseudo-session per category → average length minus 1 |

Re-derivation runs asynchronously. If `populate_interval` is set, a background
goroutine repeats the derivation on that cadence (e.g. every 15 minutes).

### Override precedence

Config overrides (`source: "config_override"`) are **never overwritten** by
replay-derived entries. The precedence order is:

```
config_override  >  replay_derived  >  (absent → use config fallback)
```

To "un-override" a value and let replay derivation take over, remove the entry
from the YAML and restart the router.

### Manual entries via code

For testing or one-off adjustments, use the `LookupTableStorage` write API
directly:

```go
storage.Set(
    lookuptable.QualityGapKey("coding", "small", "large"),
    lookuptable.Entry{
        Value:     0.25,
        Source:    lookuptable.SourceManual,
        UpdatedAt: time.Now(),
    },
)
```

Manual entries behave like replay-derived entries for override purposes — a
`config_override` entry for the same key still takes precedence.

### Inspecting the table at runtime

```go
// Snapshot of all entries (returns a copy, safe to iterate).
for key, entry := range storage.All() {
    fmt.Printf("%-50s  value=%.4f  source=%-16s  samples=%d\n",
        key, entry.Value, entry.Source, entry.SampleCount)
}
```

---

## Examples

### Example 1: Cold start with config overrides only

A fresh deployment with no replay data. The operator seeds quality-gap values
based on internal benchmarks:

```yaml
model_selection:
  lookup_tables:
    enabled: true
    quality_gaps:
      - task_family: coding
        current_model: gpt-4o-mini
        candidate_model: gpt-4o
        value: 0.18
      - task_family: general
        current_model: gpt-4o-mini
        candidate_model: gpt-4o
        value: 0.05
```

When the HybridSelector evaluates whether to escalate from `gpt-4o-mini` to
`gpt-4o` for a coding task, it reads `0.18` from the table and compares it
against its threshold. For general tasks the gap is only `0.05`, so escalation
is unlikely.

### Example 2: Replay-driven derivation

After accumulating a week of replay records, enable automatic derivation:

```yaml
model_selection:
  lookup_tables:
    enabled: true
    storage_path: "/data/lookup_tables.json"
    auto_save_interval: "5m"
    populate_from_replay: true
    populate_interval: "15m"
```

The system:

1. Loads any previously persisted entries from `/data/lookup_tables.json`
2. Fetches all replay records and runs the Builder
3. Writes derived entries (quality gaps, handoff penalties, remaining turn priors)
4. Every 15 minutes, re-runs derivation to incorporate new replay data
5. Every 5 minutes, flushes dirty entries to disk

Over time, the table builds up entries like:

```
quality_gap::coding::small::large              value=0.2000  source=replay_derived  samples=150
quality_gap::general::small::large             value=0.0400  source=replay_derived  samples=320
handoff_penalty::small::large                  value=0.0180  source=replay_derived  samples=42
handoff_penalty::large::small                  value=-0.0120 source=replay_derived  samples=15
remaining_turn_prior::support                  value=3.5000  source=replay_derived  samples=89
remaining_turn_prior::coding                   value=6.2000  source=replay_derived  samples=55
```

### Example 3: Replay derivation with selective overrides

An operator observes that the replay-derived handoff penalty for `small→large`
is too low because early replay data was noisy. They pin it via config:

```yaml
model_selection:
  lookup_tables:
    enabled: true
    storage_path: "/data/lookup_tables.json"
    populate_from_replay: true
    populate_interval: "15m"

    handoff_penalties:
      - from_model: small
        to_model: large
        value: 0.05  # pinned higher than replay-derived 0.018
```

The config override is applied **after** replay derivation and is tagged with
`source: config_override`. Future re-derivation runs skip this key entirely,
preserving the operator's intent.

### Example 4: Session-aware routing decision flow

Consider a support chat session currently on turn 3, routed to model-small:

```
1. HybridSelector receives a new turn in the "support" domain.

2. It reads remaining_turn_prior::support = 3.5
   → At turn 3, estimated remaining = 3.5 - (3-1) = 1.5 turns.

3. It reads quality_gap::support::small::large = 0.12
   → Upgrading to model-large would gain 0.12 in quality.

4. It reads handoff_penalty::small::large = 0.02
   → Switching mid-session costs 0.02.

5. Decision logic:
   - Only ~1.5 turns remain, so the benefit window is short.
   - Net benefit per turn: 0.12 - 0.02 = 0.10
   - Total expected benefit: 0.10 × 1.5 = 0.15
   - If threshold > 0.15 → stay on model-small (avoid disruption).
   - If threshold ≤ 0.15 → escalate to model-large.
```

All three table types collaborate to produce a **cost-benefit-aware, session-
length-informed** routing decision — as opposed to stateless per-request
routing that ignores the ongoing session context.

### Example 5: Unit testing with in-memory storage

```go
func TestRoutingWithLookupTable(t *testing.T) {
    // Create an in-memory table — no files, no replay store.
    mem := lookuptable.NewMemoryStorage()

    // Seed known values for deterministic test behavior.
    _ = mem.Set(
        lookuptable.QualityGapKey("coding", "small", "large"),
        lookuptable.Entry{Value: 0.20, Source: lookuptable.SourceManual},
    )
    _ = mem.Set(
        lookuptable.HandoffPenaltyKey("small", "large"),
        lookuptable.Entry{Value: 0.02, Source: lookuptable.SourceManual},
    )
    _ = mem.Set(
        lookuptable.RemainingTurnPriorKey("coding"),
        lookuptable.Entry{Value: 4.0, Source: lookuptable.SourceManual},
    )

    // Attach to the selector.
    selector := selection.NewHybridSelector(hybridCfg)
    selector.SetLookupTable(mem)

    // Exercise the selector — it now uses data-driven constants
    // instead of the static config.QualityGapThreshold.
    result, err := selector.Select(ctx, selCtx)
    // ... assertions ...
}
```

### Example 6: Populating from replay records in tests

```go
func TestBuilderDerivesTables(t *testing.T) {
    mem := lookuptable.NewMemoryStorage()
    builder := lookuptable.NewBuilder(mem)

    now := time.Now()
    records := []store.Record{
        // Two models in "coding": small scores 0.7 avg, large scores 0.9 avg.
        {Category: "coding", SelectedModel: "small", ConfidenceScore: 0.6},
        {Category: "coding", SelectedModel: "small", ConfidenceScore: 0.8},
        {Category: "coding", SelectedModel: "large", ConfidenceScore: 0.9},

        // A model switch within the same session.
        {Decision: "default", SelectedModel: "small", ActualCost: ptr(0.01), Timestamp: now},
        {Decision: "default", SelectedModel: "large", ActualCost: ptr(0.03), Timestamp: now.Add(time.Second)},

        // Support sessions averaging 4 turns (stored as prior = 3.0).
        {Decision: "chat", Category: "support", Timestamp: now},
        {Decision: "chat", Category: "support", Timestamp: now.Add(time.Second)},
        {Decision: "chat", Category: "support", Timestamp: now.Add(2 * time.Second)},
        {Decision: "chat", Category: "support", Timestamp: now.Add(3 * time.Second)},
    }

    err := builder.PopulateFromRecords(records)
    require.NoError(t, err)

    // Verify derived values.
    gap, ok := mem.QualityGap("coding", "small", "large")
    assert.True(t, ok)
    assert.InDelta(t, 0.2, gap, 1e-9) // 0.9 - 0.7

    penalty, ok := mem.HandoffPenalty("small", "large")
    assert.True(t, ok)
    assert.InDelta(t, 0.02, penalty, 1e-9) // 0.03 - 0.01

    prior, ok := mem.RemainingTurnPrior("support")
    assert.True(t, ok)
    assert.InDelta(t, 3.0, prior, 1e-9) // 4 turns - 1
}
```

---

## Key source files

| File | Role |
|---|---|
| [table.go](src/semantic-router/pkg/selection/lookuptable/table.go) | Core types: `Key`, `Entry`, `LookupTable`, `LookupTableStorage` interfaces |
| [builder.go](src/semantic-router/pkg/selection/lookuptable/builder.go) | Derivation logic from replay records |
| [memory.go](src/semantic-router/pkg/selection/lookuptable/memory.go) | In-memory storage (for tests and ephemeral use) |
| [file.go](src/semantic-router/pkg/selection/lookuptable/file.go) | File-based persistent storage with atomic writes and auto-save |
| [hybrid.go](src/semantic-router/pkg/selection/lookuptable/../../hybrid.go) | HybridSelector: `resolveQualityGap()` consumes the table |
| [factory.go](src/semantic-router/pkg/selection/factory.go) | `WithLookupTable()` wires the table into the selector graph |
| [router_selection.go](src/semantic-router/pkg/extproc/router_selection.go) | Lifecycle: build, populate, override, teardown |
| [selection_config.go](src/semantic-router/pkg/config/selection_config.go) | YAML config structs |
