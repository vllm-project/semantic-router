# MMLU Router Recipe

This recipe is the smallest maintained example of KB-backed domain escalation.
It uses the built-in `mmlu_kb` knowledge base to keep lower-uplift domains on a
local 7B lane and escalate higher-uplift domains to a larger frontier lane.

The maintained assets live here:

- `mmlu-router.yaml`
- `mmlu-router.dsl`

This profile intentionally relies on the canonical global defaults for
`mmlu_kb`. It does not redefine `global.model_catalog.kbs` inside the recipe.
The built-in seed lives under `config/knowledge_bases/mmlu/` and the default KB
definition lives in `global.model_catalog.kbs`.

## Design Goals

- Demonstrate the maintained contract for built-in KB-backed routing without
  extra classifier or plugin noise.
- Keep the YAML profile small enough to read end-to-end.
- Provide a stable example of `routing.signals.kb`, `kb_metric` projections, and
  decision routing driven by built-in knowledge bases.

## Route Order

| Priority | Decision | Target model | Purpose |
|---|---|---|---|
| `200` | `escalate_72b` | `cloud/frontier-72b` | Escalate domains with stronger frontier uplift |
| `100` | `keep_7b` | `local/small-7b` | Keep lower-uplift domains on the local lane |

The current escalation split is:

- Escalate: biology, business, computer science, economics, history, math, other, philosophy, psychology
- Keep local: chemistry, engineering, health, law, physics

## Signal Strategy

- `routing.signals.kb[]` binds one best-match KB signal per MMLU label.
- `escalation_pressure` reads the built-in `mmlu_kb` metric `escalate_vs_keep`.
- `escalation_band` maps that score into `no_escalation` vs `escalation_signal`.
- `escalate_72b` currently uses explicit best-label OR conditions for the
  escalate set, while `keep_7b` catches the non-escalation projection band.

## Validation Commands

Repo-local config and DSL checks:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/src/semantic-router"
go test ./pkg/config ./pkg/dsl ./pkg/apiserver
go run ./cmd/dsl decompile \
  -o ../../deploy/recipes/mmlu/mmlu-router.dsl \
  ../../deploy/recipes/mmlu/mmlu-router.yaml
go run ./cmd/dsl validate ../../deploy/recipes/mmlu/mmlu-router.dsl
```
