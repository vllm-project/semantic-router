# Classifier Model Evaluation

`mom_collection_eval.py` defaults to the Router's served classifier collection:
Vela Feedback (five classes), FactCheck, Domain (`intent` CLI key), PII,
and Guard (`jailbreak` key). Vela native
snapshots are pinned to the immutable revisions in the Router registry.

`--collection legacy-mom` explicitly selects the previous five mmBERT-32K
models and their legacy adapter entries. The filename remains compatible.
Loading preserves each artifact's tokenizer, context configuration, pooling and
complete classification head. Wrong label orders or missing head parameters
fail before scoring. The generic `--use_lora` path supports only the explicit
legacy collection.

The default datasets are **historical source diagnostics**, not independent
Vela release benchmarks. In particular, the old Feedback dataset has no
NO_FEEDBACK gold examples. Results retain all five classes, their support,
unsupported labels, and full-contract macro F1; zero support does not establish
quality for that class. Use a separately held-out custom dataset for five-class
coverage. FactCheck predicts whether external factual knowledge is needed,
not factual truth.

Guard detects instruction attacks. Its `benign`/`jailbreak` class names do not
make the previous mixed toxicity/jailbreak dataset compatible. Served Guard
therefore requires `--custom_dataset`: a JSON array or CSV with complete `text`
and independently reviewed `label` values (`benign`/`jailbreak`, or 0/1 in that
order). Unknown annotations must be resolved or excluded before evaluation;
they are never silently converted to benign. `safe`/`unsafe` aliases remain
exclusive to the legacy collection. No default Guard quality score is emitted
without compatible data.

## Install

```bash
cd src/training/model_eval
pip install -r requirements.txt
```

## Run

Evaluate one merged model:

```bash
python mom_collection_eval.py --model feedback --device cpu --limit 100

python mom_collection_eval.py --model jailbreak --device cpu \
  --custom_dataset reviewed-attacks.json
```

Evaluate several models or their LoRA variants:

```bash
python mom_collection_eval.py \
  --collection legacy-mom \
  --model feedback jailbreak fact-check intent pii \
  --use_lora \
  --device cuda
```

Useful options:

| Option | Purpose |
|---|---|
| `--collection` | `served` (default) or explicit `legacy-mom` |
| `--revision` | explicit revision for an override or legacy artifact |
| `--dtype` | native loading precision; default `float32`, without autocast |
| `--max_length` | full-input token budget (default 32768); over-budget input fails |
| `--model_id` | override the registered checkpoint for a single-model run |
| `--custom_dataset` | use a local JSON or CSV dataset |
| `--language` | filter rows when the dataset exposes a supported language field |
| `--batch_size` | control inference memory use |
| `--limit` | run a small smoke sample |
| `--parallel` | evaluate multiple models concurrently |
| `--output_dir` | choose the result directory |

Use underscores in option names, as shown by
`python mom_collection_eval.py --help`.

Download evaluation native snapshots with `make download-eval-models`. Production
`make download-models` continues to use the Router's downloader, including its
runtime artifact validation. Existing `download-mmbert-*` targets remain legacy
utilities; they do not download Vela. Legacy adapters must declare an available
base and save every newly initialized task-head parameter; otherwise evaluation
rejects them instead of scoring a random head.

## Results

This evaluator uses native full-input argmax inference; it does not reproduce
Router Guard/PII scanning windows or FactCheck threshold decisions. The loaded
precision, pooling, model revision and token budget are recorded in each result.

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
uses the same immutable pins for published Vela models and takes the class order
from the artifact's own mapping, reports calibration and
threshold behaviour alongside accuracy, and writes provenance manifests next to
the result.

```
python src/training/model_eval/quality_baseline.py \
    --task fact-check --device cuda --output-dir baseline/fact-check

# From src/training/model_eval. The served artifacts predate the training-run
# manifests, so this reports one missing run_ref per artifact until a run
# publishes one. Everything else has to pass.
python -m provenance.cli validate baseline/fact-check/manifests

python src/training/model_eval/gap_report.py \
    --baseline baseline/*/*_baseline.json --output baseline/gap-report.md
```

`--artifact-repo` measures a candidate instead of the served artifact, and
`--artifact-dir` with `--artifact-manifest` measures a locally trained artifact
before anything is published. Both are recorded in the result, so a candidate
number is never mistaken for the baseline.

The baseline runner's historical `jailbreak` dataset is restricted to the
explicit original mmBERT merged/adapter artifacts. It rejects current Guard
before accessing that dataset. Use the custom-data collection evaluator above
for reviewed instruction-attack annotations.

A referenced manifest supplies the identity every number is published under, so
it also selects the bytes: the run downloads the repository and revision the
manifest names, and re-hashes the files it lists, whether they came from the Hub
or from `--artifact-dir`. A directory that does not hash to the manifest, or an
`--artifact-repo` the manifest does not describe, fails before scoring starts.

The inventory identifies known task bindings and reports other classifier
artifacts as coverage gaps. A generic `classifiers` signal is not assumed to be
Guard merely because it uses a classification head. Complexity scores embedding
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
