# Domain Configurations

This directory holds domain-specific prompts for training cache embedding
adapters. The live config is [`prompts.yaml`](prompts.yaml) — there are no
per-domain YAML files and no AWS helper script in this tree.

## Available Domains

Canonical domains currently defined in `prompts.yaml`:

- **computer_science** (alias: `programming`)
- **health** (alias: `medical`)
- **law**
- **psychology**

`generate_training_data.py --domain` accepts either the canonical name or the
alias (`medical` → `health`, `programming` → `computer_science`).

## Domain Selection Guide

The multi-domain LoRA approach has proven effective across diverse domains. Validated results show consistent improvements:

### Proven Domains

Domains with validated improvement results:

- **Medical/Healthcare** (+14.6%) - Clinical terms, diseases, treatments
- **Law** (+16.9%) - Case law, legal concepts
- **Programming** (+11.3%) - Code, technical documentation
- **Psychology** (+34.9%) - Mental health, theories

**Multi-domain average: +19.4% improvement**

### Recommended Domains

Additional domains likely to benefit from training:

- **Biology** - Taxonomy, molecular structures
- **Chemistry** - Reactions, compounds
- **Engineering** - Technical specs, standards
- **Economics** - Mathematical models, theories
- **Finance** - Financial terminology, regulations
- **Scientific** - Research and academia

## Adding a New Domain

### 1. Prepare Your Data

Create unlabeled queries file:

```jsonl
{"query": "Your domain-specific question 1"}
{"query": "Your domain-specific question 2"}
{"query": "Your domain-specific question 3"}
```

Save to: `data/cache_embeddings/<domain>/unlabeled_queries.jsonl`

### 2. Add prompts and train

Add a new top-level block under `domains:` in [`prompts.yaml`](prompts.yaml)
(do **not** create a separate per-domain YAML file). Then follow the python
commands in the [Main README](../README.md) (`generate_training_data.py` and
`lora_trainer.py`).

## Planned Domains

Future domain adapters to train:

- [ ] legal - Legal and law queries
- [ ] financial - Banking and finance
- [ ] scientific - Research and academia
- [ ] programming - Code and technical docs
- [ ] history - Historical queries
- [ ] philosophy - Philosophical concepts
- [ ] psychology - Mental health and psychology
- [ ] engineering - Engineering and technical
- [ ] business - Business and management
- [ ] education - Educational content
- [ ] mathematics - Math and statistics
- [ ] literature - Books and literary analysis
- [ ] art - Art history and criticism

Total: 13 domains planned

## Troubleshooting

### "No prompts defined for domain"

`generate_training_data.py` prints this, followed by `Available domains: ...`,
when `--domain` is neither a key under `domains:` in `prompts.yaml` nor a
supported alias (`medical` → `health`, `programming` → `computer_science`).
List the configured domains with:

```bash
# From the repository root
python3 -c "import yaml; print(sorted(yaml.safe_load(open('src/training/model_embeddings/cache_embeddings/domains/prompts.yaml'))['domains']))"
```

## See Also

- [Main README](../README.md) - Complete technical documentation
