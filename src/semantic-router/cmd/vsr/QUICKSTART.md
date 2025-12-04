# VSR Quick Start Guide

Get the vLLM Semantic Router up and running in minutes.

## Prerequisites

- **Go 1.21+** (for building)
- **Docker** (for Docker deployments)
- **kubectl** (for Kubernetes deployments)
- **Helm** (for Helm deployments)

## 1. Build VSR

```bash
cd semantic-router/src/semantic-router
make build-cli
export PATH=$PATH:$(pwd)/bin
```

## 2. Initialize Configuration

```bash
vsr init
```

This creates `config/config.yaml`. Edit it to configure your model and endpoints.

## 3. Download Models

```bash
make download-models
```

## 4. Validate Configuration

```bash
vsr config validate
```

Fix any errors reported before proceeding.

## 5. Deploy

Choose your deployment environment:

### Local (Development)

```bash
vsr deploy local
```

### Docker Compose (Recommended)

```bash
vsr deploy docker
```

### Kubernetes

```bash
vsr deploy kubernetes --namespace default
```

### Helm

```bash
vsr deploy helm --namespace default
```

## 6. Check Status

```bash
vsr status
```

## 7. Test the Router

```bash
vsr test-prompt "What is the weather today?"
```

## 8. View Logs

```bash
vsr logs --follow
```

## Common Commands

| Command | Purpose |
|---------|---------|
| `vsr status` | Check deployment status |
| `vsr logs` | View logs |
| `vsr health` | Quick health check |
| `vsr dashboard` | Open dashboard in browser |
| `vsr model list` | List available models |
| `vsr undeploy [env]` | Stop deployment |
| `vsr upgrade [env]` | Upgrade to latest version |
| `vsr debug` | Run diagnostics |

## Troubleshooting

### Configuration Issues

```bash
vsr config validate --verbose
```

### Deployment Issues

```bash
vsr debug
```

### Port Conflicts

Check which ports are in use:

```bash
vsr debug
```

### Can't Connect to Dashboard

```bash
# For Docker/Local
vsr dashboard

# For Kubernetes/Helm
vsr dashboard --namespace [your-namespace]
```

## Next Steps

- Read the [full documentation](README.md) for advanced features
- Learn about [model management](README.md#model-commands)
- Explore [deployment options](README.md#-deployment)
- Set up [monitoring and metrics](README.md#monitoring-commands)

## Getting Help

```bash
# General help
vsr --help

# Command-specific help
vsr deploy --help
vsr model --help
```

## Quick Reference

```bash
# Full workflow
vsr init                        # Initialize config
make download-models            # Download models
vsr config validate            # Validate
vsr deploy docker              # Deploy
vsr status                     # Check status
vsr test-prompt "hello"        # Test
vsr logs --follow              # Monitor
vsr undeploy docker            # Clean up
```

For complete documentation, see [README.md](README.md).
