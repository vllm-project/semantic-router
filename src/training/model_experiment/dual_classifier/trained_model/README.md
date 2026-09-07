# Local Checkpoint Directory

`train_example.py` writes the dual-classifier checkpoint and tokenizer files
here. Model weights are intentionally excluded from Git.

From the parent directory, regenerate the checkpoint with:

```bash
python train_example.py
```

Expected files include `model.pt`, `config.json`, tokenizer files, and
`training_history.json`. Treat them as local experiment artifacts; use a model
registry with an accompanying model card when a checkpoint needs to be shared.
