# LFM2.5-Encoder and SCX Router exploration

Exploratory scripts that try two models on the modality-routing task (`AR`,
`DIFFUSION`, `BOTH`) using the fixed split from `../exported_modality_routing_dataset/`.
They are not part of the planned candidate experiment. The results and how to read
them are in section 12 of `../DECISION_RECORD.md`.

## What is here

- `exploration_common.py`: labels, pinned model revisions, the GLiClass dataset
  builder, and how results are scored and saved.
- `lfm25_classifier.py`: the LFM2.5 wrapper (masked-mean pooling plus a linear head),
  model loading and the prediction loop.
- `train_lfm25_encoder.py`, `eval_lfm25.py`: LoRA fine-tuning of LFM2.5-Encoder-350M,
  and scoring the result.
- `train_scx_router.py`, `eval_scx.py`: full fine-tuning of SCX Router v0.1 with
  gliclass, and scoring the result.
- `try_scx_zeroshot.py`: SCX Router as released, with no training.

## Run

Every script takes `--help`. Outputs go to `runs/`, which is git-ignored.

```bash
python train_lfm25_encoder.py --output-dir runs/lfm25_encoder_finetuned
python eval_lfm25.py --run-dir runs/lfm25_encoder_finetuned

python train_scx_router.py --output-dir runs/scx_router_finetuned
python eval_scx.py runs/scx_router_finetuned runs/scx_finetuned_preds.json

python try_scx_zeroshot.py
```

The prediction files carry the sha256 of every prompt, so they can be passed to the
label audit, for example `judge_labels.py report --preds scx_finetuned=<file>`. The
audit refuses predictions it cannot tie to the rows it scores them on.

## Things to know

- **Pinned revisions.** Both models load custom code or weights from the Hub, so
  `exploration_common.py` pins them to the commits the reported numbers came from.
- **Seeds.** Both trainers take `--seed` (default 42). The reported numbers come
  from runs made before that option existed.
- **LFM2.5 load path.** The model card's `AutoModel` path returns random weights.
  `lfm25_classifier.load_lfm25_body` loads the masked-LM model and takes `.lfm2`.
- **SCX loss.** The single-label loss is broken as shipped, so the task is expressed
  as multi-hot with one active label per row.
- **Large writes.** Trainer checkpoints are off. Multi-gigabyte writes coincided with
  WSL crashes, so each script saves once at the end.

## Tests

`pip install pytest`, then run `pytest` in the parent directory. The tests use fake
encoders and pipelines, so they need no GPU, model download or network.
