# Cache-Embedding Domains

[`prompts.yaml`](prompts.yaml) is the only domain prompt registry used by
`generate_training_data.py`.

Configured domains are:

- `computer_science` (alias `programming`)
- `health` (alias `medical`)
- `law` (alias `legal`)
- `psychology`

List the current keys directly from the registry:

```bash
python -c 'import yaml; print(*yaml.safe_load(open("src/training/model_embeddings/cache_embeddings/domains/prompts.yaml"))["domains"], sep="\n")'
```

## Add a Domain

1. Collect representative queries as JSONL, one `query` field per line.
2. Add one top-level entry under `domains:` in `prompts.yaml`. Include a role,
   topic name, paraphrase guidance and examples, negative guidance and examples,
   and the generation model to use.
3. Generate a small sample and review whether positives are answer-equivalent
   and negatives require different answers.
4. Generate the full training set, then keep a separate held-out test set that
   was not produced from the same templates.
5. Train and evaluate using the [parent guide](../README.md).

Do not add a domain based only on expected improvement. Publish measured results
with the resulting model card and retain the exact dataset and prompt version
used for that artifact.
