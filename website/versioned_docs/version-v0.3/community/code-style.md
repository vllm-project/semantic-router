# Code Style & Quality

To maintain a high-quality codebase, we adhere to specific style guides and use automated tools to enforce them.

## Mandatory Quality Checks

We use `pre-commit` hooks to ensure consistency. **These checks must pass before submitting a PR.**

### Setup Pre-commit

1. **Install pre-commit:**

   ```bash
   pip install pre-commit
   # OR
   brew install pre-commit
   ```

2. **Install hooks:**

   ```bash
   pre-commit install
   ```

3. **Run checks manually:**

   ```bash
   pre-commit run --all-files
   # OR
   make precommit-local
   ```

## Language-Specific Guidelines

### Go Code

- Format with `gofmt`.
- **Naming:** Use meaningful variable and function names.
- **Comments:** Document exported functions and types.
- **Modules:** Run `make check-go-mod-tidy` to verify all modules are tidy.
- **Lint:** Run `make go-lint` to check for issues, or `make go-lint-fix` to auto-fix.

### Rust Code

- Format with `cargo fmt`.
- Lint with `cargo clippy`.
- Use `Result` types for error handling.
- Document public APIs.

### Python Code

- Follow **PEP 8** style guidelines.
- Use type hints.
- Write docstrings for classes and functions.
