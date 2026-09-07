# Runtime Examples

This directory holds backend-specific configuration examples used by router
configs, tutorials, and tests. They are user-facing support assets, but they are
not reusable routing-schema fragments and they are not deployment manifests.

- `memory/`: agentic memory backend configuration references (Milvus, Valkey)
- `response-cache/`: external response-cache backend example files
- `response-api/`: external Response API Redis example files
- `tools/`: local tools database examples
- `vector-store/`: vector store backend configuration references

Start with `config/config.yaml` or `config/recipes/` for a complete router
configuration, then reference or copy only the runtime backend example needed
by that configuration.
