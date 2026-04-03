# Maintained Recipes

`deploy/recipes/` holds maintained routing profiles that should parse and run
against the canonical router defaults without carrying their own copy of the
global KB catalog.

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
