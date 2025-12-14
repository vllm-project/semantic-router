# Contributing to vLLM Semantic Router

Thank you for your interest in contributing to the vLLM Semantic Router project! This guide will help you get started with development and contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Prerequisites](#prerequisites)
- [Building the Project](#building-the-project)
- [Running Tests](#running-tests)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
  - [Code Quality Checks](#code-quality-checks)
- [Submitting Changes](#submitting-changes)
- [Project Structure](#project-structure)

## Development Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Rust** (latest stable version)
- **Go** 1.24.1 or later
- **Hugging Face CLI** (`pip install huggingface_hub`)
- **Make** (for build automation)
- **Python** 3.8+ (Optional: for training and testing)

### Initial Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vllm-project/semantic-router.git
   cd semantic-router
   ```

2. **Download required models:**

   ```bash
   make download-models
   ```

   This downloads the pre-trained classification models from Hugging Face.

3. **Install Python dependencies (Optional):**

   ```bash
   # For training and development
   pip install -r requirements.txt
   
   # For end-to-end testing
   pip install -r e2e-tests/requirements.txt
   ```

## Building the Project

The project consists of multiple components that need to be built in order:

### Build Everything

```bash
make build
```

### Build Individual Components

1. **Rust library (Candle binding):**

   ```bash
   make rust
   ```

2. **Go router:**

   ```bash
   make build-router
   ```

### Running the System

1. **Start Envoy proxy** (in one terminal):

   ```bash
   make run-envoy
   ```

2. **Start the semantic router** (in another terminal):

   ```bash
   make run-router
   ```

## Running Tests

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
make test-prompt

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
python e2e-tests/run_all_tests.py

# Run specific test
python e2e-tests/00-client-request-test.py

# Run tests matching a pattern
python e2e-tests/run_all_tests.py --pattern "0*-*.py"

# Check if services are running
python e2e-tests/run_all_tests.py --check-only
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
   make clean
   make build
   make test
   ```

4. **Run end-to-end tests:**

   ```bash
   # Start services
   make run-envoy &
   make run-router &
   
   # Run tests
   python e2e-tests/run_all_tests.py
   ```

5. **Commit your changes:**

   Commit your changes with a clear message, making sure to **sign off** on your work using the `-s` flag. This is required by the project's **Developer Certificate of Origin (DCO)**.

   ```bash
   git add .
   git commit -s -m "feat: add your feature description"
   ```

### Debugging

- **Envoy logs:** Check the terminal running `make run-envoy` for detailed request/response logs
- **Router logs:** Check the terminal running `make run-router` for classification and routing decisions
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
# Install pre-commit hooks
pre-commit install

# Run all checks
pre-commit run --all-files
```

### Go Code

- Follow standard Go formatting (`gofmt`)
- Use meaningful variable and function names
- Add comments for exported functions and types
- Write unit tests for new functionality
- **Keep Go modules tidy:** Run `go mod tidy` in the appropriate directory after adding or removing dependencies
  - For `candle-binding`: `cd candle-binding && go mod tidy`
  - For `src/semantic-router`: `cd src/semantic-router && go mod tidy`
  - The CI will automatically check that `go.mod` and `go.sum` files are tidy using `make check-go-mod-tidy`

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
   python e2e-tests/run_all_tests.py
   ```

   The `make test` command includes:
   - `go vet` for static analysis
   - `check-go-mod-tidy` for Go module dependency verification
   - Unit tests for all components

2. **Create a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Test results and validation steps

3. **Address review feedback** promptly

## Project Structure

```
├── candle-binding/          # Rust library for BERT classification
├── src/semantic-router/     # Go implementation of the router
├── src/training/           # Model training scripts
├── e2e-tests/              # End-to-end test suite
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
