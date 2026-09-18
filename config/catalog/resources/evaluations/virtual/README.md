# Virtual-model evaluations

Store benchmark results for recipe-backed virtual Model Cards in this
directory. Use the same evaluation schema as `../single/`, but set `model` to a
canonical `vllm-sr/*` virtual-model ID and record the exact recipe run in
`subject` metadata. No built-in virtual recipe has a reproducible published run
yet, so the directory intentionally contains no YAML measurements rather than
fabricated placeholders.
