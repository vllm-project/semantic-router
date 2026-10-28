# Bearer Management API Fixture

A test Recipe for the CLI integration suite. It routes every request to one
test model and turns on bearer authentication for the Router management API.
`test_integration_recipe_bearer.py` imports it through the Dashboard from a
local HTTPS server and activates it with a non-root `vllm-sr serve`.

`recipe.dsl` is the canonical DSL of `config.yaml`. The Dashboard's package
tests check that it stays canonical.
