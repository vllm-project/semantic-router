# Code Style and Quality

Formatting and static checks are enforced by repository configuration. Use the
checked-in tools instead of maintaining separate editor-specific rules.

## Run the shared checks

Install the repository-managed hooks and run them across tracked files:

```bash
make precommit-install
make precommit-check
```

To reproduce the containerized pre-commit workflow:

```bash
make precommit-local
```

For the normal changed-file path:

```bash
make impact ENV=cpu CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
```

## Language conventions

### Go

- Format with `gofmt`.
- Prefer cohesive packages and split only where ownership or testability improves.
- Document exported APIs where their purpose is not self-evident.
- Verify module metadata with `make check-go-mod-tidy`.
- Use `make go-lint` for the repository lint configuration.

### Rust

- Format with `cargo fmt`.
- Run `cargo clippy` through the repository's reported validation path.
- Return typed errors rather than panicking in normal failure paths.
- Keep unsafe and FFI boundaries small and documented.

### Python

- Support the Python version declared by the package you are changing.
- Use type hints on public and non-trivial interfaces.
- Keep command orchestration separate from reusable logic.
- Run the component's tests through its Make target when one exists.

### TypeScript and React

- Follow the Dashboard ESLint and TypeScript configuration.
- Keep data fetching and transformation out of presentational components when
  a helper or hook owns that responsibility.
- Add focused component or E2E coverage for user-visible behavior.

## Generated files

Do not hand-edit generated API references, schemas, or catalog blocks. Change
their source and run the owning generator, then include both source and output
in the same pull request.
