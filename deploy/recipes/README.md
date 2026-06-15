# Maintained Recipes

`deploy/recipes/` holds maintained routing profiles that should parse and run
against the canonical router defaults without carrying their own copy of the
global KB catalog.

## DSL and YAML Pairs

Some recipes keep a `.dsl` authoring source next to the compiled `.yaml`
runtime artifact. For example, `balance.dsl` uses `DECISION_TREE` syntax to
make conflict-free branches easier to review, while `balance.yaml` stores the
canonical flat `routing.decisions` representation consumed by the router.

`DECISION_TREE` is DSL authoring sugar. Runtime config, Kubernetes translation,
and `DecompileRouting()` operate on canonical flat `ROUTE` / decision entries
and do not preserve the original tree shape. Keep paired DSL/YAML recipes in
sync when the authoring shape matters.

## Built-in Knowledge Bases

Recipes should reference built-in knowledge bases only by name from routing
surfaces such as:

- `routing.signals.kb[].kb`
- `routing.projections.scores[].inputs[].kb`

Do not repeat built-in `global.model_catalog.kbs` entries inside individual
recipe YAML files. The built-in definitions belong in the global defaults so any
recipe can consume them directly.

To add a new built-in knowledge base:

1. Add the bundled seed assets under `config/knowledge_bases/<seed-name>/`.
2. Register the built-in KB in `src/semantic-router/pkg/config/canonical_defaults.go`.
3. Mirror the reference entry in `config/config.yaml`.
4. Keep `source.path` aligned with `knowledge_bases/<seed-name>/`; runtime
   bootstrap will seed that directory into `.vllm-sr/knowledge_bases/<seed-name>/`
   instead of inventing a second logical path layer.

The current built-in recipe-facing KBs are `privacy_kb` and `mmlu_kb`, backed
by `config/knowledge_bases/privacy/` and `config/knowledge_bases/mmlu/`.
