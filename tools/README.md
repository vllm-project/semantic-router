# Repository tools

`tools/` owns repository automation and development support. Subdirectories are
organized by responsibility rather than by the feature that happened to add a
script:

- `make/`, `ci/`, `linter/`, `docker/`, and `release/`: build and CI plumbing;
- `agent/`: the executable agent harness, its internal docs, and calibration;
- `models/`: model export, reference generation, and training helpers;
- `smoke/`: directly runnable API smoke tests;
- `dev/` and `demos/`: local developer entrypoints and demonstrations;
- `redis/`, `valkey/`, `milvus/`, and `mcp-classifier-server/`: auxiliary
  services and backend validation tools.

Public product documentation belongs in `website/`; deployable manifests belong
in `deploy/`; router configuration belongs in `config/`. New one-off scripts
should be placed in the narrowest existing tool family or promoted into a Make
target when they become part of the supported workflow.
