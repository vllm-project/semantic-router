# AWS vLLM Training Deployment

Simple Ansible scripts to deploy AWS GPU instances for vLLM cache embedding training.

## Prerequisites

```bash
# Install Ansible and AWS dependencies
pip install ansible boto3 botocore

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region (us-east-2)
```

## Quick Start

### 1. Deploy Instance

```bash
cd src/training/cache_embeddings/aws
./deploy-vllm.sh deploy
```

This will:
- Launch g5.12xlarge instance (4x A10G GPUs) in us-east-2
- Install NVIDIA drivers, Python, vLLM, and dependencies
- Clone semantic-router repository
- Generate SSH commands and instance details file

### 2. Upload Training Data

```bash
# Use the SCP command from the instance details file
scp -i ~/.ssh/router-team-us-east2.pem -r ../../../../data/cache_embeddings ubuntu@<PUBLIC_IP>:~/semantic-router/
```

### 3. SSH and Run Training

```bash
# SSH to instance (command in instance details file)
ssh -i ~/.ssh/router-team-us-east2.pem ubuntu@<PUBLIC_IP>

# Run training with 4 GPUs
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
```

### 4. Download Results

```bash
# Download augmented data
scp -i ~/.ssh/router-team-us-east2.pem ubuntu@<PUBLIC_IP>:~/semantic-router/data/cache_embeddings/medical/augmented_full.jsonl ../../../../data/cache_embeddings/medical/
```

### 5. Cleanup

```bash
# Terminate instance and clean up files
./deploy-vllm.sh cleanup
```

## Configuration

Edit [launch-vllm-instance.yaml](launch-vllm-instance.yaml) to customize:

```yaml
vars:
  aws_region: us-east-2
  instance_type: g5.12xlarge  # Change for different GPU configs
  root_volume_size: 200  # GB
```

### Instance Types

| Instance Type | GPUs | GPU Memory | vCPUs | RAM | Approx Cost/hr |
|--------------|------|------------|-------|-----|----------------|
| g5.xlarge | 1x A10G | 24GB | 4 | 16GB | $1.00 |
| g5.2xlarge | 1x A10G | 24GB | 8 | 32GB | $1.20 |
| g5.12xlarge | 4x A10G | 96GB | 48 | 192GB | $5.00 |
| g5.48xlarge | 8x A10G | 192GB | 192 | 768GB | $16.00 |

### Expected Performance (44K queries)

| GPUs | Batch Size | Tensor Parallel | Expected Time |
|------|------------|-----------------|---------------|
| 1 | 32 | 1 | ~2 hours |
| 4 | 64 | 4 | ~30 minutes |
| 8 | 128 | 8 | ~15 minutes |

## Files Created

After deployment:
- `vllm-instance-i-xxxxx.txt` - Instance details and commands
- `vllm-inventory-i-xxxxx.ini` - Ansible inventory file

These files are used for cleanup and are automatically removed when you run `./deploy-vllm.sh cleanup`.

## Troubleshooting

### Instance launch fails

```bash
# Check AWS credentials
aws sts get-caller-identity

# Check if you have access to us-east-2
aws ec2 describe-instances --region us-east-2
```

### NVIDIA driver not found

```bash
# SSH to instance and check
nvidia-smi

# If not found, the AMI may need updating
# Check latest Ubuntu 22.04 Deep Learning AMI in us-east-2
```

### vLLM installation issues

```bash
# SSH to instance and reinstall
pip3 uninstall vllm -y
pip3 install vllm --no-cache-dir
```

## Cost Optimization

1. **Use Spot Instances** - Edit YAML to add spot instance request (70% savings)
2. **Stop vs Terminate** - Stop instance when not in use, restart later
3. **Right-size** - Start with g5.xlarge (1 GPU) for testing
4. **Monitor** - Set up AWS Budgets alerts

## Security

- Instances use existing security group `sg-0704e6bcaa5e655b1`
- SSH key: `router-team-us-east2`
- All EBS volumes encrypted by default
- Instances tagged with creation metadata

## Notes

- **Do not commit** the generated inventory and instance detail files (they contain IPs and instance IDs)
- These scripts are already in `.gitignore` via the `aws/` directory pattern
- Always run `cleanup` when done to avoid unnecessary AWS charges
