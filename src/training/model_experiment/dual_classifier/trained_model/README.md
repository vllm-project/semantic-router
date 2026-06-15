# Trained Model Directory

This directory contains the trained model files for the dual classifier. Due to GitHub's file size limitations, these files are excluded from version control via `.gitignore`.

## Files that should be in this directory:

- `config.json` - Model configuration
- `model.pt` - The trained PyTorch model (main model file, ~270MB)
- `special_tokens_map.json` - Special tokens mapping
- `tokenizer_config.json` - Tokenizer configuration
- `training_history.json` - Training history and metrics
- `vocab.txt` - Vocabulary file

## To generate these files:

Run the training script to create a new model:

```bash
cd dual_classifier
python train_example.py
```

## Alternative storage:

For sharing large model files, consider:

- Git LFS (Large File Storage)
- Cloud storage (S3, Google Drive, etc.)
- Model registries (HuggingFace Hub, MLflow, etc.) 
