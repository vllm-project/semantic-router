# Decision 2.0 research workspace

Local source of truth for Decision 2.0 benchmark, training and publication code. Model inference, evaluation and training run on the designated accelerator machine; this repository contains no credentials or private infrastructure details.

The [unified research and experiment gist](https://gist.github.com/Xunzhuo/cd90fce0fa548616d8a4f1b2d2398dea) tracks source evidence, frozen protocols, exact model revisions, run receipts and release decisions.

Run the module commands below from `src/training/decision2`; Python imports
such as `training.model.train` are relative to this directory. The repository
`make test-training-contracts` target includes all dependency-light contracts
in this tree.

## Work stages

1. Audit Decision 1.0 data, evaluation and released weights. Freeze a new held-out benchmark before using any results for 2.0 selection.
2. Run published 1.0 weights, open baselines and Jev through the same typed-decision requests. Record native validity and full probabilities.
3. Train controlled 2.0 pilots on disjoint data. Change one factor at a time where possible: data coverage, backbone/readout, loss, hard-case curriculum and calibration.
4. Select a family from development results. Evaluate frozen final candidates once on the held-out suite and preserve all attempts.
5. Publish weights, runtime, complete model cards, raw evaluation artifacts and score-generated rank/matrix charts.

## Benchmark item contract

Each JSONL record has a stable `id`, `family`, `split`, `state`, `questions`, `gold`, and `provenance`. `questions` uses the TypeSafe System One Choice/Noul/Score structure; `gold` is keyed by question ID and names the correct option or score level. A `group_id` binds related counterfactuals and a `pair_id` binds exact permutation pairs. The benchmark implementation owns the detailed schema and validates it before any model call.

Main metrics are family-macro accuracy and raw per-family counts. Secondary metrics include micro accuracy, Brier, log loss, calibration, selective risk, permutation consistency and native failure/coverage. Latency, throughput and cost are separate axes. Every score requires exact sample, model, adapter and code revisions.

## Evaluation panels

`jev_arena/` defines the frozen Decision 2.0 development and release ranking
contract. The independent public JevBench reproduction has 231 exposed items;
it must not be labelled an official sealed JevBench score. The
`decision_bench_v4/` adapter pins a separate external Choice-only corpus:
1,041 text-readable items from 34 tasks. It marks 30 image-only icon items
not evaluable for a text-only model and reports modality slices. These public
items are useful regressions, not blind release evidence. The sealed synthetic
and held-out human transfer panels remain distinct; new authored items require
an audited answer key and a frozen scoring protocol before joining a release
composite.

`training/data/build_kai06b_native_v1.py` converts audited train and selection
partitions to the published Kai 0.6B native fine-tuning contract. It records
whole-group input-length exclusions instead of truncating or silently dropping
examples. `scripts/build_length_probe.py` establishes long-context training
feasibility before a large-backbone run; its one-row result is not an accuracy
measurement.
