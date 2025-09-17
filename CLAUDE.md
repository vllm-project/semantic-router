# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM Semantic Router is a intelligent routing system that uses BERT-based semantic classification to select the optimal model for LLM requests. The system consists of a Rust library for ML inference and a Go service implementing the Envoy ExtProc interface for request routing.

## Build Commands

### Essential Build Commands
```bash
# Build everything (Rust library + Go router)
make build

# Build only Rust library (candle-binding)
make rust

# Build only Go router
make build-router

# Download required models from Hugging Face
make download-models

# Clean all build artifacts
make clean
```

### Running the System
```bash
# Run the semantic router (requires models downloaded)
make run-router

# Run Envoy proxy (separate terminal)
make run-envoy

# Use custom config file
CONFIG_FILE=custom.yaml make run-router
```

## Testing Commands

### Core Testing
```bash
# Run all tests (includes go vet, go mod tidy checks, and unit tests)
make test

# Test individual components
make test-binding                    # Test Rust bindings
make test-semantic-router            # Test Go router
make test-category-classifier        # Test category classification
make test-pii-classifier            # Test PII detection
make test-jailbreak-classifier      # Test jailbreak detection
```

### Manual Testing (requires services running)
```bash
# Test different routing scenarios
make test-auto-prompt-reasoning      # Test reasoning mode
make test-auto-prompt-no-reasoning   # Test normal mode
make test-pii                       # Test PII detection
make test-prompt-guard              # Test jailbreak detection
make test-tools                     # Test tool auto-selection
```

### Milvus Cache Testing
```bash
# Start Milvus container
make start-milvus

# Test with Milvus backend
make test-milvus-cache
make test-semantic-router-milvus

# Stop Milvus when done
make stop-milvus
```

### End-to-End Testing
```bash
# Start services first
make run-envoy &
make run-router &

# Run comprehensive e2e tests
python e2e-tests/run_all_tests.py

# Run specific tests
python e2e-tests/00-client-request-test.py
```

## Code Quality

### Pre-commit Hooks
```bash
# Install pre-commit hooks (mandatory for contributions)
pip install pre-commit
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files
```

### Go Module Management
```bash
# Keep Go modules tidy (checked by CI)
cd candle-binding && go mod tidy
cd src/semantic-router && go mod tidy
```

## Architecture

### High-Level Components
- **Candle Binding**: Rust library using the [candle](https://github.com/huggingface/candle) ML framework for BERT-based classification
- **Semantic Router**: Go service implementing Envoy ExtProc interface for intelligent request routing
- **Configuration**: YAML-based configuration for models, endpoints, and routing rules

### Core Classification Models
- **Category Classifier**: Routes requests to appropriate models based on content domain (math, science, law, etc.)
- **PII Classifier**: Detects and blocks personally identifiable information
- **Jailbreak Classifier**: Identifies and blocks prompt injection attempts

### Semantic Caching
- **Memory Backend**: Fast in-memory cache for development
- **Milvus Backend**: Scalable vector database for production deployments

### Directory Structure
```
├── candle-binding/              # Rust ML library with BERT classification
├── src/semantic-router/         # Go router service (Envoy ExtProc)
├── src/training/               # Model training and fine-tuning scripts
├── config/                     # Configuration files (config.yaml, etc.)
├── e2e-tests/                  # End-to-end test suite
├── models/                     # Downloaded classification models
└── website/                    # Documentation website
```

### Key Configuration Files
- `config/config.yaml`: Main configuration for models, endpoints, and routing rules
- `config/tools_db.json`: Tool selection database
- `config/cache/milvus.yaml`: Milvus vector database configuration

## Development Environment Setup

### Prerequisites
- Rust (latest stable)
- Go 1.24.1+
- Hugging Face CLI (`pip install huggingface_hub`)
- Make
- Python 3.8+ (for training and e2e tests)

### Initial Setup
```bash
# Clone and download models
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
make download-models

# Install Python dependencies (optional)
pip install -r requirements.txt
pip install -r e2e-tests/requirements.txt
```

## Documentation

### Documentation Development
```bash
# Start documentation dev server
make docs-dev

# Build documentation for production
make docs-build

# Lint documentation
make docs-lint
```

## Environment Variables

- `LD_LIBRARY_PATH`: Must include `${PWD}/candle-binding/target/release` for Rust library loading
- `CONFIG_FILE`: Path to configuration file (default: `config/config.yaml`)
- `CONTAINER_RUNTIME`: Container runtime for Milvus (`docker` or `podman`)
- `VLLM_ENDPOINT`: vLLM endpoint URL for testing
- `SKIP_MILVUS_TESTS`: Skip Milvus-dependent tests (default: `true`)

## Important Notes

- Always run `make download-models` before first build
- The system requires both Envoy and the router to be running for end-to-end functionality
- Use `make test` before submitting changes to ensure all quality checks pass
- For production deployments, consider using Milvus backend for semantic caching