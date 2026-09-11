# Configuration and Recipe loop

## Canonical configuration

YAML is the only operator-authored configuration representation. Before
editing, discover only the relevant schema surface:

```bash
vllm-sr config schema --endpoint http://router-management:8080
vllm-sr config schema --endpoint http://router-management:8080 \
  --section routing
vllm-sr config schema --endpoint http://router-management:8080 \
  --surface algorithm:multi_factor
```

Use `--full` only for offline tooling that genuinely needs the entire JSON
Schema.

## Validate, plan, apply

```bash
export VSR_MGMT_TOKEN='...'

vllm-sr config validate --config candidate.yaml \
  --endpoint http://router-management:8080
vllm-sr config plan --config candidate.yaml --mode replace \
  --endpoint http://router-management:8080
vllm-sr config apply --config candidate.yaml --mode replace \
  --endpoint http://router-management:8080
```

`plan` performs the same canonical parse, merge/replace, and hot-reload checks
as mutation but writes nothing. `apply` plans again and uses the returned ETag
as a compare-and-swap precondition. Prefer `replace` for a complete reviewed
document; use `merge` only for an intentionally partial patch.

Inspect or recover state with:

```bash
vllm-sr config get --endpoint http://router-management:8080
vllm-sr config versions --endpoint http://router-management:8080
vllm-sr config rollback <version> --endpoint http://router-management:8080
```

## Recipes

Use the same online lifecycle for one named Recipe:

```bash
vllm-sr recipe validate recipe.yaml --endpoint http://router-management:8080
vllm-sr recipe plan recipe.yaml --endpoint http://router-management:8080
vllm-sr recipe apply recipe.yaml --endpoint http://router-management:8080
```

A Recipe file declares its `name` and `routing` object. Model identities,
evaluation records, index selection, cost data, and minimum coverage remain
typed YAML fields discoverable from the running schema. Do not copy a field
from website examples without validating it against the target Router.
