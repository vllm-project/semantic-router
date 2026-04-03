# Contributing to vLLM Semantic Router

Thank you for your interest in contributing to the vLLM Semantic Router project! This guide will help you get started with development and contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
  - [Code Quality Checks](#code-quality-checks)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)

## Development Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Docker** (or Podman)
- **Make** (for build automation)
- **Python** 3.10+ (Optional: for training and testing)

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vllm-project/semantic-router.git
   cd semantic-router
   ```

2. **Use the canonical local image workflow:**

   ```bash
   make vllm-sr-dev
   vllm-sr serve --image-pull-policy never
   ```

   For AMD ROCm development:

   ```bash
   make vllm-sr-dev VLLM_SR_PLATFORM=amd
   vllm-sr serve --image-pull-policy never --platform amd
   ```

   This workflow:
   - Rebuilds the local image
   - Installs the `vllm-sr` CLI tool
   - Uses the local image only, without pulling a remote fallback

3. **Install Python dependencies (Optional):**

   ```bash
   # For training and development
   pip install -r requirements.txt
   
   # For end-to-end testing
   pip install -r e2e/testing/requirements.txt
   ```

## Running Tests

### Agent Gates

The repository-specific agent harness is indexed in [docs/agent/README.md](docs/agent/README.md). Treat [AGENTS.md](AGENTS.md) as the short entrypoint and `docs/agent/*` plus `tools/agent/*` as the durable source of truth.
If a real architecture or code/spec gap remains after your change, add or update the durable debt entry indexed from [docs/agent/tech-debt/README.md](docs/agent/tech-debt/README.md).

Read these first:

- [docs/agent/testing-strategy.md](docs/agent/testing-strategy.md)
- [docs/agent/module-boundaries.md](docs/agent/module-boundaries.md)

Use the agent-specific gates for changed files:

```bash
make agent-bootstrap
make agent-validate
make agent-scorecard
make agent-report ENV=cpu CHANGED_FILES="path/one,path/two"
make agent-ci-lint CHANGED_FILES="path/one,path/two"
make precommit-branch-gate
make agent-ci-gate CHANGED_FILES="path/one,path/two"
make agent-pr-gate
make test-and-build-local
make agent-feature-gate ENV=cpu CHANGED_FILES="path/one,path/two"
```

Use `make agent-ci-lint` when you want to reproduce the same changed-file lint path that the CI pre-commit workflow runs, including the shared agent bootstrap toolchain and tracked-file codespell check.
Use `make precommit-branch-gate` when you want the same local prelint gate the installed `pre-push` hook runs before a push or PR update.
Use `make agent-pr-gate` when you want the repo-native local baseline for the PR jobs contributors most often miss: `Pre-commit / Run pre-commit hooks` and `Test And Build`.

`ENV=amd` is required when platform-specific behavior changed.

### Unit Tests

1. **Test Rust bindings:**

   ```bash
   make test-binding
   ```

2. **Test Go semantic router:**

   ```bash
   make test-semantic-router
   ```

3. **Test individual classifiers:**

   ```bash
   make test-category-classifier
   make test-pii-classifier
   make test-jailbreak-classifier
   ```

### Manual Testing

Test different routing scenarios:

```bash
# Test model auto-selection
make test-auto-prompt-reasoning
make test-auto-prompt-no-reasoning

# Test PII detection
make test-pii

# Test prompt guard (jailbreak detection)
make test-prompt-guard

# Test tools auto-selection
make test-tools
```

### End-to-End Tests

Ensure both Envoy and the router are running, then:

```bash
# Run all e2e tests
python e2e/testing/run_all_tests.py

# Run specific test
python e2e/testing/00-client-request-test.py

# Run tests matching a pattern
python e2e/testing/run_all_tests.py --pattern "0*-*.py"

# Check if services are running
python e2e/testing/run_all_tests.py --check-only
```

The test suite includes:

- Basic client request tests
- Envoy ExtProc interaction tests
- Router classification tests
- Semantic cache tests
- Category-specific tests
- Metrics validation tests

## Development Workflow

### Making Changes

1. **Create a feature branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the project structure and coding standards.

3. **Build and test:**

   ```bash
   make agent-report ENV=cpu CHANGED_FILES="path/one,path/two"
   make agent-ci-gate CHANGED_FILES="path/one,path/two"
   make agent-feature-gate ENV=cpu CHANGED_FILES="path/one,path/two"
   ```

4. **Run end-to-end tests:**

   ```bash
   make agent-e2e-affected CHANGED_FILES="path/one,path/two"
   # Or run a specific profile directly
   make e2e-test E2E_PROFILE=ai-gateway
   ```

5. **Commit your changes:**

   Commit your changes with a clear message, making sure to **sign off** on your work using the `-s` flag. This is required by the project's **Developer Certificate of Origin (DCO)**. The repository does not require commit messages to use the PR title classification prefixes.

   ```bash
   git add .
   git commit -s -m "clarify PR title guidance"
   ```

### Debugging

- **View logs:** Use `vllm-sr logs` to view service logs
- **Rust library:** Use `RUST_LOG=debug` environment variable for detailed Rust logs
- **Go library:** Use `SR_LOG_LEVEL=debug` environment variable for detailed Go logs

## Code Style and Standards

### Code Quality Checks

Before submitting a PR, please run the pre-commit hooks to ensure code quality and consistency. **These checks are mandatory** and will be automatically run on every commit once installed.

**Step 1: Install pre-commit tool**

```bash
# Using pip (recommended)
pip install pre-commit

# Or using conda
conda install -c conda-forge pre-commit

# Or using homebrew (macOS)
brew install pre-commit
```

**Step 2: Install pre-commit hooks for this repository**

```bash
# Install the repo-native pre-commit + pre-push hooks
make precommit-install

# Run all checks
pre-commit run --all-files
# OR
make precommit-branch-gate
# OR
make precommit-local
```

### Go Code

- Follow standard Go formatting (`gofmt`)
- Use meaningful variable and function names
- Add comments for exported functions and types
- Write unit tests for new functionality
- **Keep Go modules tidy:** Run `make check-go-mod-tidy` to verify all modules are tidy
- **Lint Go code:** Run `make go-lint` to check for issues, or `make go-lint-fix` to auto-fix

### Rust Code

- Follow Rust formatting (`cargo fmt`)
- Use `cargo clippy` for linting
- Handle errors appropriately with `Result` types
- Document public APIs

### Python Code

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for functions and classes

## Submitting Changes

1. **Ensure all tests pass:**

   ```bash
   make test
   python e2e/testing/run_all_tests.py
   ```

   The `make test` command includes:
   - `go vet` for static analysis
   - `check-go-mod-tidy` for Go module dependency verification
   - Unit tests for all components

2. **Create a pull request** with:
   - A module-aligned PR title using the repository prefixes from [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md), such as `[Docs][CI/Build] Align PR template with vLLM` or `[Router][Dashboard] Tighten route visibility in the console`
   - A clear `Purpose` section describing the change and affected module(s)
   - Reference to any related issues
   - A `Test Plan` and `Test Result` section with the actual validation steps and outcomes

3. **Address review feedback** promptly

## Project Structure

```
├── candle-binding/          # Rust library for BERT classification
├── src/semantic-router/     # Go implementation of the router
├── src/training/           # Model training scripts
├── e2e/testing/              # End-to-end test suite
├── config/                 # Configuration files
├── docs/                   # Documentation
├── deploy/                 # Deployment configurations
├── Makefile               # Build automation
└── requirements.txt       # Python dependencies
```

### Key Components

- **Candle Binding:** Rust library providing BERT-based classification
- **Semantic Router:** Go service implementing the Envoy ExtProc interface
- **Training Scripts:** Python scripts for fine-tuning classification models
- **Configuration:** YAML files defining routing rules and model endpoints

## Getting Help

- Check the [documentation](https://vllm-semantic-router.com/)
- Review existing issues and pull requests
- Ask questions in discussions or create a new issue

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (Apache 2.0).
