# Repository tools

`tools/` owns repository automation and development support. Subdirectories are
organized by responsibility rather than by the feature that happened to add a
script:

- `make/`, `ci/`, `linter/`, `docker/`, and `release/`: build and CI plumbing;
- `agent/`: the executable agent harness, its internal docs, and skills;
- `codegen/` and `docs/`: generated contracts, API references, and documentation;
- `catalog/`: model catalog generation and validation;
- `calibration/`: recipe verification, routing calibration, and tuning;
- `models/`: model export and classifier operating-point utilities;
- `dev/`: local developer entrypoints, DSL tools, Kind, and examples;
- `test/`: smoke and soak runners plus shared test services;
- `security/`: repository security checks and their policy.

Public product documentation belongs in `website/`; deployable manifests belong
in `deploy/`; router configuration belongs in `config/`. New one-off scripts
should be placed in the narrowest existing tool family or promoted into a Make
target when they become part of the supported workflow.

Model training belongs in `src/training/`; binding-specific reference generators
belong beside their binding, such as `candle-binding/scripts/`. Generated run
reports and local caches are runtime artifacts, not tool source files.
