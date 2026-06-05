---
sidebar_position: 5
---

# Projection traces and replay

## Overview

When [router replay](../../installation/configuration) captures routing records, each record can include a structured **`projection_trace`** field (JSON) in addition to `projections` (matched output names) and `projection_scores` (aggregated numeric scores).

The trace explains *how* partition reduction, weighted scores, and mapping thresholds behaved for that request—so operators and dashboard users can debug routing without inferring internals from scalar scores alone.

## Key Advantages

- Replay records stay self-describing: the same persistence path carries both aggregate scores and structured explainability JSON.
- Partition contender lists, softmax winners, mapping boundary distance, and per-input score contributions surface in one object.
- Version `1` in the payload leaves room for additive fields without rewriting older consumers.

## What Problem Does It Solve?

Matched projection names (`projections`) and numeric summaries (`projection_scores`) answer **what was chosen**, but they do not preserve **why** a partition picked a winner or how close a mapping was to the next threshold band.

`projection_trace` closes that gap for audits, support, and insights views without extra query-time inference.

## When to Use

- You run **router replay** (memory, Redis, or PostgreSQL) and want explainability columns on each record.
- You use the **dashboard Insights** drill-down for replay-backed flows and need collapsible projection detail.
- You are building tooling that validates projection behavior against real traffic—not only against static config.

## Configuration

Explainability payloads are emitted when projections are evaluated; storage depends on replay backend configuration:

- Enable replay with the persistence settings described in **[Router replay configuration](../../installation/configuration)**.
- For PostgreSQL, ensure migrations include column **`projection_trace`** (JSONB) alongside **`projections`** and **`projection_scores`**.

There is no separate “trace on/off” switch—tracing is implicit whenever projections run and the recorder persists the enriched `SignalResults`/`Record`.

## Schema version `1`

The trace is versioned for forward-compatible consumers.

- **`partitions`**: one entry per `routing.projections.partitions` group that ran reduction on this request. Records **`contenders`** with **`raw_score`** (and **`normalized_score`** when semantics are `softmax_exclusive`), the chosen **`winner`**, **`winner_score`** (what the router stores on the signal afterward), **`raw_winner_score`**, **`margin`** (top minus second among comparison scores—normalized weights for softmax, raw confidences otherwise), and **`default_used`** when the partition synthesized its configured default member.
- **`scores`**: one entry per configured projection score with **`total`** and per-input **`contribution`** (`weight * value`), matching the weighted-sum runtime.
- **`mappings`**: one entry per projection mapping. For each threshold band in order, the trace records **`matched`**, **`boundary_distance`** (distance to the nearest active threshold), and—for the first matching band—the **`selected_output`**, sigmoid **`confidence`**, and **`boundary_distance`** used for that band.

## Where to inspect

- **Dashboard → Insights**: open a replay-backed record. The **Projection trace** section shows tables for partition winners (with contender breakdown when present), score inputs, and mapping decisions (including boundary distance and a per-output threshold step list), plus collapsible raw JSON.
- **Storage**: the same object is persisted on the replay `Record` (memory, Redis, or PostgreSQL column `projection_trace` JSONB).

Traces are derived only from the projection contract and evaluated signal results—there is no opaque sidecar model.
