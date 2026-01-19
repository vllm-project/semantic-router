# Development Guide

This guide covers the prerequisites, setup, and testing procedures for the vLLM Semantic Router.

## Prerequisites

Ensure you have the following installed:

- **Docker** (or Podman)
- **Make** (for build automation)
- **Python** 3.10+ (recommended, for training and testing)

## Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vllm-project/semantic-router.git
   cd semantic-router
   ```

2. **Start the development environment:**

   ```bash
   make vllm-sr-start
   ```

   This single command handles everything:
   - Builds the Docker image with all dependencies
   - Downloads required models from Hugging Face
   - Installs the `vllm-sr` CLI tool
   - Starts all services (semantic router, envoy, dashboard)

3. **Install Python dependencies (Optional):**

   ```bash
   # For training and development
   pip install -r requirements.txt
   
   # For end-to-end testing
   pip install -r e2e/testing/requirements.txt
   ```

## Debugging Tips

- **Rust:** Set `RUST_LOG=debug`.
- **Go:** Set `SR_LOG_LEVEL=debug`.

## Running Tests

### Unit Tests

- **Rust bindings:**

  ```bash
  make test-binding
  ```

- **Go Router:**

  ```bash
  make test-semantic-router
  ```

- **Classifiers:**

  ```bash
  make test-category-classifier
  make test-pii-classifier
  make test-jailbreak-classifier
  ```

### Manual Testing

Use these commands to test specific scenarios:

```bash
# Model auto-selection
make test-auto-prompt-no-reasoning
make test-auto-prompt-reasoning
make test-pii          # PII detection
make test-prompt-guard # Jailbreak detection
make test-tools        # Tools auto-selection
```

### End-to-End Tests

Ensure services are running, then:

```bash
# Run all E2E tests
python e2e/testing/run_all_tests.py

# Run specific test
python e2e/testing/00-client-request-test.py
```
