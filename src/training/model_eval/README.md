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

python -m provenance.cli validate baseline/jailbreak/manifests

python src/training/model_eval/gap_report.py \
    --baseline baseline/*/*_baseline.json --output baseline/gap-report.md
```

`--artifact-repo` measures a candidate instead of the served artifact, and
`--artifact-dir` with `--artifact-manifest` measures a locally trained artifact
before anything is published. Both are recorded in the result, so a candidate
number is never mistaken for the baseline.

See `provenance/README.md` for the manifest contract and what fails validation.
