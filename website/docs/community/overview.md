# Contributing

:::info Source of Truth
This page is generated from [CONTRIBUTING.md](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md). The repository file is the authoritative source.
:::

## Quick Start

```bash
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router
make download-models  # Download ML models from HuggingFace
make build            # Build Rust + Go components
make test             # Run all tests
```

## Workflow

### Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### Build & Test Locally

```bash
make clean && make build && make test
```

### Run E2E Tests

```bash
make run-envoy &
make run-router &
python e2e/testing/run_all_tests.py
```

### Run Pre-commit Checks

```bash
pre-commit run --all-files
```

If not installed: `pip install pre-commit && pre-commit install`

### Commit with DCO

All commits **must** be signed off (DCO required). Without `-s`, CI will **reject** your PR.

```bash
git commit -s -m "feat: add something"
```

### Submit PR

- Target branch: `main`
- Include: description + related issue links + test results
- `make test` and E2E tests must pass

## Debugging

| Component | How to Debug |
|-----------|--------------|
| Envoy | Check `make run-envoy` terminal for request/response logs |
| Router | Check `make run-router` terminal for routing decisions |
| Rust | `RUST_LOG=debug` (levels: trace/debug/info/warn/error) |
| Go | `SR_LOG_LEVEL=debug` |

## Related Guides

- **[Development Guide](./development)**: Prerequisites, building, running tests
- **[Documentation Guide](./documentation)**: How to write and translate docs
- **[Code Style](./code-style)**: Formatting, linting, pre-commit hooks
