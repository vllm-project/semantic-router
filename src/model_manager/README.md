# Model Manager

A Python module for automated ML model download, verification, and caching from HuggingFace.

## Features

- **Automated Download**: Download models from HuggingFace Hub with support for specific revisions and file filtering
- **Integrity Verification**: Verify model integrity using size or SHA256 checksums
- **Smart Caching**: Skip downloads for already-cached models
- **CI-Friendly**: Environment variable-based config selection for minimal CI downloads
- **CLI & Programmatic API**: Use from command line or import as a Python module

## Quick Start

### CLI Usage

```bash
# Download all models (uses config/model_manager/models.yaml by default)
python -m model_manager

# Use a specific config file
python -m model_manager --config config/model_manager/models.yaml

# Download a specific model only
python -m model_manager --model category_classifier_modernbert-base_model

# List all configured models and their cache status
python -m model_manager --list

# Verify existing models without downloading
python -m model_manager --verify-only

# Clean all cached models (from default config)
python -m model_manager --clean

# Clean all models from a specific config
python -m model_manager --config config/model_manager/models.yaml --clean

# Clean a specific model
python -m model_manager --clean-model category_classifier_modernbert-base_model

# Verbose output for debugging
python -m model_manager -v

# Show help message
python -m model_manager --help
```

### CI Mode

For CI environments, use the minimal model set to speed up builds and avoid rate limits:

```bash
# Auto-select minimal config
CI_MINIMAL_MODELS=true python -m model_manager

# Or explicitly specify
python -m model_manager --config config/model_manager/models.minimal.yaml
```

### Programmatic Usage

```python
from model_manager import ensure_models, ModelManager

# Simple: ensure all models are downloaded
ensure_models("config/model_manager/models.yaml")

# Advanced: use ModelManager for more control
manager = ModelManager.from_config("config/model_manager/models.yaml")

# Ensure all models
paths = manager.ensure_all()
print(paths)  # {'model_id': '/path/to/model', ...}

# Ensure a specific model
path = manager.ensure_model("category_classifier_modernbert-base_model")

# Check if a model is cached
from model_manager import is_cached, get_model_path
spec = manager.get_model_spec("category_classifier_modernbert-base_model")
if is_cached(spec, manager.config.cache_dir):
    print(f"Model cached at: {get_model_path(spec, manager.config.cache_dir)}")
```

## Configuration

Models are defined in YAML configuration files. See `config/model_manager/` for examples.

### Configuration Schema

```yaml
# Cache directory for downloaded models
cache_dir: "models"

# Verification level: "none", "size", or "sha256"
verify: "sha256"

# List of models to manage
models:
  - id: my-model # Unique identifier (required)
    repo_id: org/model-name # HuggingFace repo ID (required)
    revision: main # Git revision (optional, default: main)
    local_dir: custom-name # Override local directory name (optional)
    files: # Specific files to download (optional)
      - config.json
      - model.safetensors
```

### Verification Levels

| Level    | Description          | Use Case                  |
| -------- | -------------------- | ------------------------- |
| `none`   | No verification      | Fast, trust HuggingFace   |
| `size`   | Verify file sizes    | Quick verification for CI |
| `sha256` | Full SHA256 checksum | Production environments   |

### Available Configs

| Config                | Description                                  |
| --------------------- | -------------------------------------------- |
| `models.yaml`         | Full model set for local development         |
| `models.minimal.yaml` | Minimal set for CI (faster, no gated models) |
| `models.lora.yaml`    | LoRA adapters only                           |

## Environment Variables

| Variable                    | Description                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------- |
| `CI_MINIMAL_MODELS`         | Set to `true`, `1`, or `yes` to auto-select minimal config                              |
| `HF_TOKEN`                  | HuggingFace token for gated models (e.g., `embeddinggemma-300m`)                        |
| `HF_ENDPOINT`               | A HuggingFace mirror endpoint for accelerated downloads (e.g., `https://hf-mirror.com`) |
| `HF_HUB_ENABLE_HF_TRANSFER` | Set to `1` to enable faster downloads using `hf_transfer` (enabled by default in CI)    |

## API Reference

### Functions

#### `ensure_models(config_path, cache_dir=None)`

Main entry point. Downloads and verifies all models from config.

#### `is_cached(spec, cache_dir)`

Check if a model is already cached.

#### `get_model_path(spec, cache_dir)`

Get the local path for a model.

#### `download_model(spec, cache_dir)`

Download a model from HuggingFace.

#### `verify_model(path, verify_level)`

Verify model integrity.

### Classes

#### `ModelManager`

Central manager for model operations.

- `from_config(config_path)` - Create from config file
- `ensure_all()` - Ensure all models are ready
- `ensure_model(model_id)` - Ensure a specific model
- `get_model_spec(model_id)` - Get model specification

#### `ModelSpec`

Specification for a single model.

- `id` - Unique identifier
- `repo_id` - HuggingFace repository ID
- `revision` - Git revision (default: "main")
- `local_dir` - Override local directory name
- `files` - Specific files to download

#### `ModelsConfig`

Configuration container.

- `models` - List of ModelSpec
- `cache_dir` - Cache directory path
- `verify` - Verification level

### Exceptions

| Exception            | Description               |
| -------------------- | ------------------------- |
| `ModelManagerError`  | Base exception            |
| `MissingModelError`  | Model not found in config |
| `BadChecksumError`   | Verification failed       |
| `DownloadError`      | Download failed           |
| `ConfigurationError` | Invalid config            |

## Development

### Running Tests

```bash
# Run all model_manager tests
pytest src/model_manager/tests/ -v

# Run with coverage
pytest src/model_manager/tests/ --cov=model_manager
```

### Project Structure

```
src/model_manager/
├── __init__.py      # Public API and ModelManager class
├── __main__.py      # CLI entrypoint
├── cli.py           # CLI implementation
├── config.py        # Configuration dataclasses
├── registry.py      # YAML config parsing
├── downloader.py    # HuggingFace download logic
├── verifier.py      # Integrity verification
├── cache.py         # Cache management
├── errors.py        # Custom exceptions
└── tests/           # Unit tests
```
