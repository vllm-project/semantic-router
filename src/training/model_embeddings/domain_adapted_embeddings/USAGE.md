# Domain-Adapted Embedding Usage

Run commands from `src/training/model_embeddings/domain_adapted_embeddings/`.

## Prepare Data

The input must provide question-and-answer pairs. JSON is an array; JSONL has
one object per line:

```json
{"question":"How is diabetes diagnosed?","answer":"Diagnosis uses blood glucose and A1C tests..."}
```

Prepare a local file:

```bash
python prepare_data.py \
  --source jsonl \
  --input-file domain_qa.jsonl \
  --output-dir data \
  --test-size 0.2
```

Or use a Hugging Face dataset whose rows contain `question`/`answer` or
`Question`/`Answer`:

```bash
python prepare_data.py \
  --source huggingface \
  --dataset DATASET_ID \
  --split train \
  --output-dir data
```

The command writes:

- `corpus_chunks.pkl`;
- `train_queries.pkl`;
- `test_queries.pkl`;
- small JSON samples for inspection.

Inspect the split and chunks before training. Duplicate or near-duplicate
answers across the train and test sets can inflate retrieval metrics.

## Train

```bash
python train.py \
  --data-dir data \
  --output-dir models/domain-adapted
```

Important options include:

| Option | Purpose |
|---|---|
| `--base-model` | SentenceTransformer checkpoint to adapt |
| `--iterations` | mining and fine-tuning rounds |
| `--num-queries` | cap the training set for a smoke run |
| `--learning-rate` | optimizer learning rate |
| `--epochs` | epochs per iteration |
| `--batch-size` | training batch size |
| `--margin` | triplet-loss margin |
| `--easy-to-hard-ratio` | balance retained easy triplets against mined hard ones |
| `--top-k` | retrieval depth used during mining |
| `--hard-neg-rank` | rank boundary for hard negatives |

Use `python train.py --help` for current defaults. Tune on validation data and
preserve the final test split for one unbiased evaluation.

## Load the Result

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/domain-adapted/best", trust_remote_code=True)
query_embedding = model.encode("How is diabetes diagnosed?")
document_embeddings = model.encode(
    [
        "Blood glucose and A1C tests are used for diagnosis.",
        "Hypertension can be treated with lifestyle changes.",
    ]
)
scores = query_embedding @ document_embeddings.T
```

Confirm that the chosen similarity function and normalization match the
downstream retrieval system.

## Reproducibility Checklist

Record the input dataset revision, split seed, chunking settings, base model
revision, all training and mining options, dependency versions, and the metrics
for both the untouched base model and final checkpoint.
