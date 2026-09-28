# Decision 2.0 evaluation plane (frozen runners)

Every formal number is **post-key same-panel** evidence: JevArena v3 typed FINAL
(1,600 items / 2,000 answer slots) plus CSS15 human transfer (6,547 items),
`v3 = 100 * sqrt(T * H)`, and the separate JevBench public 231 subset
(easy/standard/hard; not the official sealed rank). Missing, invalid and native
over-budget answers count as failures. Development readouts are for checkpoint
selection only and never release scores. Decision Index numbers never enter.

Frozen panels live on node A under `/data/dev2/private/panels` (gold-free prompts
in `goldfree/`, labels in `gold/`, hashes in `panels.py`; verify with
`python3 -m v2.eval.panels verify`). Pinned image: `decision20-train-fast:host2`
(`sha256:f83b1d10…`). All commands run on the node from an exact mirror:

```bash
# local worktree: mirror a pushed commit (subtree is enough for decision2 work)
src/training/decision2/v2/common/mirror_to_node.sh --path src/training/decision2 node-a <commit>
# on the node
SRC=<full-sha>-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
export PYTHONPATH=$S
```

## (b) Formal same-panel run for a candidate package

1. Native collection on one of your leased GPUs (inference never sees gold):

```bash
$S/v2/eval/run_same_panel.sh --gpu <N> --track <track> --src $SRC \
  --run-dir /data/dev2/runs/<track>/<run-id> --model-dir <package-dir> \
  --purpose "<why>" --expected-end <UTC> \
  -- --adapter-spec <adapter.json> --model-path <package-dir> --revision <immutable-id> \
     --extra KEY=VALUE ...
```

Registered adapters (`--adapter NAME`): `decision1-{lux,nox,sol,eos}`, `kai`, `lex`,
`decider`, `bosun06`, `jpt`, `autojev27`, `gliner25`, `this-that`, `decision2-typed`
(`training/model/infer.py` checkpoints; needs `--extra model_id=… calibration=… max_length=…`).
A new package brings a JSON spec with `name`, `module`, `args` (placeholders `{model}`
`{revision}` `{input}` `{output}` `{device}` plus `--extra` keys), optional `python`,
`image`, `batch_policy`. Its collector must write one row per prompt with `id`,
`answers`, `latency_ms`, `source_input_sha256`, `model_id`, `model_revision`, and mark
native over-budget inputs invalid instead of truncating. Add `--max-items 8` first for
a smoke run into `<run-dir>/smoke/`.

2. Seal (gold-free), report, and compare with the tier's own 1.0 model:

```bash
python3 -m v2.eval.same_panel seal --run-dir <run-dir>
python3 -m v2.eval.same_panel report --run-dir <run-dir> --label "<card label>" \
  --tier <0.6B|0.8B|2B|4B|9B|27B> --family decision2 --count-safetensors <package-dir>
python3 -m v2.eval.same_panel compare --run-dir <run-dir> \
  --comparator-run-dir /data/dev2/runs/eval/m1-adopt/<kai1|sol1|nox1> \
  --left-name <candidate> --right-name <1.0 model>
python3 -m v2.eval.same_panel table --report <run-dir>/REPORT.json --report ...
```

Qwen3.5/FLA-based packages (Decision 1.0 Eos/Sol/Nox/Lux and most 2.0 decoder
candidates) pick Triton kernel configurations by runtime autotuning. On one node the
choice is stable across processes (Lux1 node A runs are bit-identical), but a
different node chose differently and moved 43 of 8,778 Lux1 answers. Compare a
candidate with its comparator on the same node and runtime, and persist the autotune
cache with the run:
`--env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR=<run>/triton-cache --mount-rw <run>/triton-cache`
(create the directory first).

`REPORT.json` carries v3/T/H, Choice/Noul/Score, per-task macro-F1, JevBench tiers,
calibration, order/label/counterfactual pair consistency, long-input and language
screens, invalid counts, latency/throughput and loaded parameters.
1.0 comparator runs: Kai1, Bosun, DEV2.0-0.6B, Sol1, Decider 2B, Nox1, Decider 4B,
JPT-9B, AutoJev 27B under `/data/dev2/runs/eval/m1-adopt/<key>`; Lex, Eos1 and Lux1
under `/data/dev2/runs/eval/m1/`. For Lux1 use the frozen-runtime run named in the
current eval gist entry.

## (a) Development readout for checkpoint selection

Collect typed DEV and CSS pilot with the same native adapter, then score them
together with SELECT/CAL probabilities from your trainer:

```bash
$S/v2/eval/run_same_panel.sh ... -- --adapter-spec <adapter.json> --model-path <ckpt> \
  --revision <id> --panels typed-dev,css-pilot
python3 -m v2.eval.dev_readout --run-dir <run-dir> \
  [--select select.probs.jsonl] [--cal cal.probs.jsonl] --label <ckpt> --output <readout.json>
```

SELECT/CAL rows are `{"id": ..., "probabilities": [...]}` aligned with the partition's
`options`; the readout reports correct/700, family-macro accuracy and Brier (six
families), per-type accuracy, T_dev, H_pilot and the development proxy
`100 * sqrt(T_dev * H_pilot)`.

## (c) Product-card charts from same-panel reports

```bash
python3 -m v2.eval.charts --report <candidate>/REPORT.json \
  --report /data/dev2/runs/eval/m1-adopt/<1.0>/REPORT.json --report <peer>/REPORT.json \
  --output-dir <card-assets>
```

Writes `jevarena-v3-rank.svg`, `jevarena-v3-model-task.svg`,
`jevbench-public231-rank.svg` and `charts.json` with the existing publication
renderer. It refuses reports with different panel or scorer hashes; there is no
Pareto chart for cards.
