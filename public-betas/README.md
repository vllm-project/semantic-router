# Public Beta Directory

This directory contains configurations for public beta deployments of the Semantic Router with vLLM on DigitalOcean AMD GPU instances.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CREATE PR with your config in public-betas/my-beta/         │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Validation runs automatically
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. PR APPROVED by maintainer                                    │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Instance created automatically
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. INSTANCE RUNNING                                             │
│    - Access via SSH or API endpoints                            │
│    - Push commits to update configuration                       │
└───────────────────────┬─────────────────────────────────────────┘
                        │ PR closed or merged
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. INSTANCE DELETED automatically                               │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Create Your Configuration

Create a new directory under `public-betas/` with your beta name:

```bash
mkdir public-betas/my-project
cp public-betas/_template/config.yaml public-betas/my-project/
```

Edit `config.yaml` with your settings:

```yaml
name: "my-project"
description: "Testing semantic router with Qwen3"
owners:
  - "your-github-username"

instance:
  region: "nyc1"
  size: "gd-40vcpu-160gb-400gb-1x-amd-mi300x"

vllm:
  model: "Qwen/Qwen2.5-7B-Instruct"
```

### 2. Submit a Pull Request

```bash
git checkout -b public-beta/my-project
git add public-betas/my-project/
git commit -m "Add public beta: my-project"
git push origin public-beta/my-project
```

### 3. Wait for Approval

- A maintainer will review your PR
- Validation runs automatically to check your config
- Once approved, the GPU instance is created **automatically**

### 4. Access Your Instance

After approval, a comment will be posted with:

- **IP Address**: For SSH access
- **API Endpoints**: For sending requests
- **Connection examples**: Ready-to-use curl commands

Example:
```bash
# SSH access
ssh root@<instance-ip>

# Send a request
curl -X POST http://<instance-ip>:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### 5. Update Configuration

Push new commits to your PR branch to update the instance configuration:

```bash
# Edit config
vim public-betas/my-project/config.yaml

# Push changes
git add -A && git commit -m "Update config" && git push
```

The instance will automatically sync with your changes.

### 6. Cleanup

Simply **close or merge** your PR - the instance is deleted automatically!

## Directory Structure

```
public-betas/
├── README.md                    # This file
├── _template/                   # Template for new betas
│   └── config.yaml
├── my-beta-1/                   # Example beta
│   └── config.yaml
└── my-beta-2/                   # Another beta
    └── config.yaml
```

## AMD GPU Instance Sizes

DigitalOcean provides AMD MI300X GPU instances with ROCm drivers pre-installed:

| Size | GPU | Memory | vCPUs | RAM | Use Case |
|------|-----|--------|-------|-----|----------|
| `gd-40vcpu-160gb-400gb-1x-amd-mi300x` | 1x MI300X | 192GB HBM3 | 40 | 160GB | Large models (70B+) |
| `s-1vcpu-1gb` | - | - | 1 | 1GB | Testing only (no GPU) |

### Model Size Guide for MI300X (192GB)

| Model | Parameters | Memory Required | Fits? |
|-------|------------|-----------------|-------|
| Qwen2.5-7B | 7B | ~14GB | ✅ Yes |
| Qwen2.5-32B | 32B | ~64GB | ✅ Yes |
| Qwen2.5-72B | 72B | ~144GB | ✅ Yes |
| Llama-3.1-70B | 70B | ~140GB | ✅ Yes |
| Llama-3.1-405B | 405B | ~800GB | ❌ No |
| Mixtral-8x7B | 47B | ~94GB | ✅ Yes |

## Configuration Reference

### Required Fields

| Field | Description |
|-------|-------------|
| `name` | Beta identifier (must match directory name, kebab-case) |
| `owners` | GitHub usernames who have SSH access |
| `instance.region` | DigitalOcean region (`nyc1`, `sfo3`, etc.) |
| `instance.size` | Droplet size |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `vllm.model` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID |
| `vllm.tensor_parallel_size` | `1` | Number of GPUs for TP |
| `vllm.gpu_memory_utilization` | `0.9` | GPU memory fraction |
| `vllm.max_model_len` | `32768` | Max sequence length |
| `semantic_router.config_source` | `default` | Config source type |

### Supported Regions

GPU instances are available in:
- `nyc1`, `nyc3` - New York
- `sfo2`, `sfo3` - San Francisco  
- `ams3` - Amsterdam
- `tor1` - Toronto

## Firewall Configuration

UFW firewall is automatically configured with these rules:

| Port | Protocol | Description |
|------|----------|-------------|
| 22 | TCP | SSH access |
| 8080 | TCP | Semantic Router API |
| 8801 | TCP | Envoy Proxy (main entry) |
| 9190 | TCP | Prometheus Metrics |

All other incoming traffic is blocked by default.

## Security Notes

- **SSH Access**: Restricted to keys configured in your DigitalOcean account
- **API Access**: Open on configured ports (consider using SSH tunnels for sensitive workloads)
- **Firewall**: UFW enabled by default, denies all non-configured incoming traffic
- **Isolation**: Each PR gets its own isolated droplet
- **Cleanup**: Instances are automatically deleted when PR is closed/merged

## Lifecycle Events

| Event | Action |
|-------|--------|
| PR Opened | Configuration validated |
| PR Approved | Instance created, services deployed |
| PR Synchronized (commits pushed) | Configuration synced to instance |
| PR Closed | Instance deleted |
| PR Merged | Instance deleted |

## Troubleshooting

### Instance not created after approval

1. Check workflow run in Actions tab
2. Verify your config passes validation
3. Ensure secrets are configured (see below)

### Can't SSH to instance

1. Verify your SSH key is in your DigitalOcean account
2. Check the IP address in the PR comment
3. Wait a few minutes after creation for cloud-init to complete

### Model won't load

1. Check if model fits in GPU memory (192GB for MI300X)
2. Verify HuggingFace token is set for gated models
3. Check vLLM container logs: `ssh root@<ip> docker logs vllm`

### Services not healthy

1. SSH into the instance
2. Check logs: `docker compose logs -f`
3. Verify GPU is accessible: `rocm-smi`

## Required Repository Secrets

| Secret | Description |
|--------|-------------|
| `DO_PERSONAL_ACCESS_TOKEN` | DigitalOcean API token with droplet write access |
| `DO_SSH_PRIVATE_KEY` | SSH private key for deployment (public key must be in DO account) |
| `HF_TOKEN` | HuggingFace token for downloading gated models |

## Support

- Open an issue with label `public-beta`
- Include your beta name and PR number
- Attach relevant logs from the workflow run
