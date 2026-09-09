# Classifier Model Evaluation

`mom_collection_eval.py` evaluates the merged and LoRA variants registered in
`constants.py`:

- feedback;
- jailbreak;
- fact-check;
- intent;
- PII.

It reports classification metrics, latency summaries, and confusion matrices
where applicable. The registry defines the default model, dataset, label
mapping, text field, and split for each task.

## Install

```bash
cd src/training/model_eval
pip install -r requirements.txt
```

## Run

Evaluate one merged model:

```bash
python mom_collection_eval.py --model feedback --device cpu --limit 100
```

Evaluate several models or their LoRA variants:

```bash
python mom_collection_eval.py \
  --model feedback jailbreak fact-check intent pii \
  --use_lora \
  --device cuda
```

Useful options:

| Option | Purpose |
|---|---|
| `--model_id` | override the registered checkpoint for a single-model run |
| `--custom_dataset` | use a local JSON or CSV dataset |
| `--language` | filter rows when the dataset exposes a supported language field |
| `--batch_size` | control inference memory use |
| `--limit` | run a small smoke sample |
| `--parallel` | evaluate multiple models concurrently |
| `--output_dir` | choose the result directory |

Use underscores in option names, as shown by
`python mom_collection_eval.py --help`.

## Results

The default output directory is `src/training/model_eval/results/`. JSON files
contain the metrics and run metadata; text-classification tasks also produce a
confusion-matrix image.

Before comparing models, verify that they used the same dataset revision,
split, label mapping, sample limit, preprocessing, device policy, and batch
size. A small `--limit` run is a functional smoke test, not a quality result.

`result_to_config.py` can convert supported evaluation summaries into router
configuration fragments. Review the generated thresholds and model references
before deployment; generation does not prove that the fragment is suitable for
your workload.

## Quality baseline

`quality_baseline.py` measures the artifact a maintained configuration actually
loads, resolved from `config/config.yaml` rather than from `constants.py`. It
takes the class order from the artifact's own mapping, reports calibration and
threshold behaviour alongside accuracy, and writes provenance manifests next to
the result.

```
python src/training/model_eval/quality_baseline.py \
    --task jailbreak --device cuda --output-dir baseline/jailbreak

# From src/training/model_eval. The served artifacts predate the training-run
# manifests, so this reports one missing run_ref per artifact until a run
# publishes one. Everything else has to pass.
python -m provenance.cli validate baseline/jailbreak/manifests

python src/training/model_eval/gap_report.py \
    --baseline baseline/*/*_baseline.json --output baseline/gap-report.md
```

`--artifact-repo` measures a candidate instead of the served artifact, and
`--artifact-dir` with `--artifact-manifest` measures a locally trained artifact
before anything is published. Both are recorded in the result, so a candidate
number is never mistaken for the baseline.

The inventory covers every task a maintained configuration loads a classifier
artifact for. Complexity is not one of them: the signal scores embedding
prototypes against candidate phrases rather than loading a classifier, so there
is no artifact to measure until #2568 adds a trained-classifier mode.

Where a task declares `positive_labels`, the router thresholds the probability
mass on those classes rather than taking an argmax, so the baseline reports what
that gate does. `operating_points` gives recall and false-positive rate at each
threshold, and `discrimination` gives AUC and recall at a false-positive budget,
which fix no threshold and so compare two artifacts built to different threshold
conventions.

`gap_report.py` sorts findings by who has to act on them. `identity`, `runtime`
and `coverage` are fixed in the config, the registry or the harness.
`calibration` and `threshold` are fixed by recalibrating or by moving the
configured threshold, so they are integration gaps too. Only `quality` says the
artifact itself is the problem and asks for a retrain or a different checkpoint.
That is the split #3194 wants, and it means a gap can be routed without
rereading the numbers.

See `provenance/README.md` for the manifest contract and what fails validation.
