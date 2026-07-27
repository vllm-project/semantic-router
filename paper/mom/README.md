# Mixture-of-Models Position Paper

This is the standalone source bundle for:

> **Mixture-of-Models: Conditional Computation Across Model Boundaries**

Author: vLLM Semantic Router Project.

The manuscript uses the official ICLR 2026 style in final mode for an arXiv
release; it is not represented as an ICLR submission. The paper defines
Mixture-of-Models as a portable conditional-computation model architecture over
heterogeneous model services and uses vLLM Semantic Router as an initial open
execution substrate for the design.

## Contents

- `main.tex`: manuscript entrypoint.
- `main.pdf`: locally generated delivery artifact (not committed).
- `sections/`: complete paper and reproducibility appendix.
- `figures/`: editable TikZ sources for the schematics and high-resolution
  source scorecards used to construct the empirical figure.
- `references.bib`: programmatically verified bibliography.
- `artifact/`: proposed MoM bundle schema and source/IR/binding/lock examples.
- `iclr2026_conference.sty`, `iclr2026_conference.bst`: unmodified official
  style assets required to compile independently.

## Build

With TeX Live and the standard Helvetica/Courier font packages installed:

```bash
make rebuild
```

The command removes stale outputs, then runs `pdflatex`, `bibtex`, and three
final `pdflatex` passes. It produces `main.pdf` without requiring network
access.

To rebuild the paper and update the website copy:

```bash
make rebuild deploy
```

The published PDF is tracked at `website/static/mom-paper.pdf` and served at
`/mom-paper`; the source tree remains the canonical, editable form.

To verify the closed MoM bundle example and its content identities:

```bash
python -m pip install -r artifact/requirements.txt
make artifact-check
```
