# Domain Configurations

This directory contains configuration files for training domain-specific cache embedding adapters.

## Quick Start

### Train a Domain Model

```bash
# From src/training/cache_embeddings/
./train-domain.sh medical
```

That's it! The script will:

1. ✅ Provision AWS GPU instance (g5.12xlarge with 4x A10G)
2. ✅ Upload your data and code
3. ✅ Run vLLM data generation (~2 hours)
4. ✅ Train LoRA adapter (~5 minutes)
5. ✅ Download trained model
6. ✅ Cleanup AWS instance

### Push to HuggingFace

```bash
./train-domain.sh medical --push-hf
```

## Available Domains

Current configurations:

- **medical** - Medical and healthcare queries (44K queries)

To add more domains, copy `TEMPLATE.yaml` and fill in the details.

## Domain Selection Guide

The multi-domain LoRA approach has proven effective across diverse domains. Validated results show consistent improvements:

### Proven Domains

Domains with validated improvement results:

- ✅ **Medical/Healthcare** (+14.6%) - Clinical terms, diseases, treatments
- ✅ **Law** (+16.9%) - Case law, legal concepts
- ✅ **Programming** (+11.3%) - Code, technical documentation
- ✅ **Psychology** (+34.9%) - Mental health, theories

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

### 2. Create Domain Config

```bash
cd domains/
cp TEMPLATE.yaml <domain>.yaml
```

Edit the file:

```yaml
domain: "<domain-name>"
description: "<Brief description>"
data_file: "data/cache_embeddings/<domain>/unlabeled_queries.jsonl"
queries_count: <number>
output_dir: "models/<domain>-cache-lora"
hf_repo: "your-org/semantic-router-<domain>-cache"  # optional
```

### 3. Train

```bash
cd ..
./train-domain.sh <domain>
```

## Advanced Usage

### Keep AWS Instance Running

Useful for debugging or multiple training runs:

```bash
./train-domain.sh medical --skip-cleanup
# Do your work...
./train-domain.sh medical --skip-aws --skip-upload  # Reuse instance
# When done:
cd aws/ && ./deploy-vllm.sh cleanup
```

### Dry Run

See what would happen without actually running:

```bash
./train-domain.sh medical --dry-run
```

## Domain Config Reference

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `domain` | ✅ | Domain identifier | `medical` |
| `description` | ✅ | Brief description | `"Medical queries"` |
| `data_file` | ✅ | Path to queries (from repo root) | `data/cache_embeddings/medical/...` |
| `queries_count` | ✅ | Approximate query count | `44603` |
| `output_dir` | ✅ | Where to save model | `models/medical-cache-lora` |
| `hf_repo` | ❌ | HuggingFace repo for upload | `org/model-name` |
| `vllm_model` | ❌ | LLM for generation | `Qwen/Qwen2.5-1.5B-Instruct` |
| `base_model` | ❌ | Base embedding model | `sentence-transformers/all-MiniLM-L12-v2` |

## Cost Estimation

| Queries | GPU Time | Cost (g5.12xlarge @ $5/hr) |
|---------|----------|----------------------------|
| 10K | ~30 min | ~$2.50 |
| 50K | ~2.5 hrs | ~$12.50 |
| 100K | ~5 hrs | ~$25 |

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

### "Domain config not found"

```bash
ls -la domains/  # Check available configs
```

### "AWS credentials not configured"

```bash
aws configure  # Set up AWS credentials
```

### "Instance IP not found"

Check `aws/vllm-instance-*.txt` for instance details

### Training failed

SSH to instance and check logs:

```bash
# Get SSH command from aws/vllm-instance-*.txt
ssh -i ~/.ssh/your-key.pem ubuntu@<instance-ip>
# Check what went wrong
```

## See Also

- [Main README](../docs/README.md) - Complete technical documentation
- [AWS Deployment Guide](../docs/QUICK_START_AWS.md) - AWS setup details
- [Validation Guide](../docs/blog.md) - How to test trained models
