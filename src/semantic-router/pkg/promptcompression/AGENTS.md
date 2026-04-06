# Prompt Compression Package Notes

## Scope

- `src/semantic-router/pkg/promptcompression/**`

## Responsibilities

- Keep compressor algorithms, backend-specific support, and regression fixtures on separate seams.
- Treat large regression suites as fixture owners, not as the main implementation surface.
- Keep prompt-rewrite policy separate from transport or provider wiring owned elsewhere in the router stack.

## Change Rules

- `compressor_nlp_test.go` is a ratcheted regression hotspot. Prefer focused fixture helpers or new targeted test files over widening the existing test file.
- Do not mix algorithm changes, backend bootstrapping, and end-to-end fixture setup in one file when adjacent helpers can own them.
- Keep package-local helpers narrow; prompt-compression work should not rejoin classification or extproc transport ownership.
