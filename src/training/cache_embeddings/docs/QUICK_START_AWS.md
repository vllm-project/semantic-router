# Quick Start: AWS vLLM Training

TL;DR for running cache embedding training on AWS with multiple GPUs.

## What This Does

Generates training triplets for LoRA fine-tuning following [arXiv:2504.02268v1](https://arxiv.org/pdf/2504.02268v1):
- **Anchor**: LLM-generated paraphrase (semantically identical)
- **Positive**: Original query
- **Negative**: LLM-generated related but distinct query (different intent)

Each triplet enables proper contrastive learning with MNR loss.

## Prerequisites

```bash
pip install ansible boto3 botocore
aws configure  # Enter credentials for us-east-2
```

## Deploy (One Command)

```bash
cd src/training/cache_embeddings/aws
./deploy-vllm.sh deploy
```

Wait ~5 minutes. You'll get:
- Instance details in `vllm-instance-i-xxxxx.txt`
- SSH command
- SCP commands for upload/download

## Upload Data

```bash
# From the repo root, run the SCP command from instance details file
scp -i ~/.ssh/router-team-us-east2.pem -r data/cache_embeddings ubuntu@<PUBLIC_IP>:~/semantic-router/
```

## Run Training (4 GPUs)

```bash
# SSH to instance
ssh -i ~/.ssh/router-team-us-east2.pem ubuntu@<PUBLIC_IP>

# Run training
cd ~/semantic-router
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/augmented_full.jsonl \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 64 \
  --gpu-memory 0.9 \
  --tensor-parallel 4 \
  --checkpoint-interval 50

# Expected: ~1.5-2 hours for 44K queries → ~130K triplets
# Output format: {"anchor": "...", "positive": "...", "negative": "...", "is_duplicate": 1}
```

## Download Results

```bash
# From local machine
scp -i ~/.ssh/router-team-us-east2.pem ubuntu@<PUBLIC_IP>:~/semantic-router/data/cache_embeddings/medical/augmented_full.jsonl data/cache_embeddings/medical/
```

## Cleanup (IMPORTANT!)

```bash
cd src/training/cache_embeddings/aws
./deploy-vllm.sh cleanup
```

## Cost

| Instance | GPUs | Time (44K) | Cost |
|----------|------|------------|------|
| g5.12xlarge | 4x A10G | ~1.5-2 hours | ~$7-10 |
| g5.xlarge | 1x A10G | ~6-8 hours | ~$6-8 |

**Always cleanup to avoid charges!**

## Troubleshooting

**Instance launch fails?**
```bash
aws sts get-caller-identity  # Check credentials
```

**NVIDIA not found?**
```bash
ssh to instance and run: nvidia-smi
```

**Training interrupted?**
```bash
# Resume from checkpoint
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/augmented_full.jsonl \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --resume
```

## Files

- [aws/README.md](aws/README.md) - Full documentation
- [aws/DEPLOYMENT_SUMMARY.md](aws/DEPLOYMENT_SUMMARY.md) - Design details
- [PRODUCTION_REFACTOR.md](PRODUCTION_REFACTOR.md) - Code improvements
