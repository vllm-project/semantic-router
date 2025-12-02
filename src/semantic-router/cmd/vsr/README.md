# VSR - vLLM Semantic Router CLI

[![Go Version](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go)](https://golang.org/doc/install)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

VSR is a comprehensive command-line tool for managing the vLLM Semantic Router. It reduces setup time from hours to minutes and provides a unified interface for deployment, monitoring, and troubleshooting across multiple environments.

## 🚀 Quick Start

```bash
# Initialize configuration
vsr init

# Validate configuration
vsr config validate

# Deploy locally
vsr deploy local

# Check status
vsr status

# Test a prompt
vsr test-prompt "What is the weather today?"

# View logs
vsr logs --follow
```

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Deployment](#-deployment)
- [Commands](#-commands)
- [Workflows](#-common-workflows)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Features

### 🎯 Core Features

- **Multi-Environment Deployment**: Support for Local, Docker Compose, Kubernetes, and Helm
- **Lifecycle Management**: Deploy, undeploy, upgrade, start, stop, restart
- **Model Management**: Download, list, validate, remove, and inspect models
- **Health Monitoring**: Status checks, health monitoring, and diagnostics
- **Debug Tools**: Interactive debugging, health checks, and diagnostic reports

### 🔧 Advanced Features

- **Enhanced Logging**: Multi-environment log fetching with filtering and following
- **Dashboard Integration**: Auto-detect and open dashboard in browser
- **Metrics Display**: View request counts, latency, and model usage
- **Configuration Validation**: Pre-deployment config validation
- **Port Forwarding**: Automatic port-forwarding for Kubernetes/Helm deployments

### 🎨 User Experience

- **Beautiful CLI Output**: Box drawing, colors, and status symbols
- **Smart Auto-Detection**: Automatically detects deployment types
- **Helpful Error Messages**: Actionable suggestions for every error
- **Comprehensive Help**: Detailed help text with examples for every command
- **Progress Indicators**: Visual feedback for long-running operations

## 📦 Installation

### Prerequisites

- **Go 1.21+** (for building from source)
- **kubectl** (optional, for Kubernetes deployments)
- **docker** (optional, for Docker deployments)
- **helm** (optional, for Helm deployments)
- **make** (optional, for building and downloading models)

### From Source

```bash
# Clone the repository
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router/src/semantic-router

# Build the CLI
make build-cli

# Or use go directly
go build -o bin/vsr ./cmd/vsr

# Add to PATH
export PATH=$PATH:$(pwd)/bin

# Verify installation
vsr --version
```

### Using Pre-built Binary

```bash
# Download the latest release
wget https://github.com/vllm-project/semantic-router/releases/latest/download/vsr-linux-amd64

# Make executable
chmod +x vsr-linux-amd64
mv vsr-linux-amd64 /usr/local/bin/vsr

# Verify installation
vsr --version
```

## ⚙️ Configuration

### Initialize Configuration

```bash
# Create a new configuration file
vsr init

# Create with template
vsr init --template basic

# Specify output location
vsr init --output config/my-config.yaml
```

### Validate Configuration

```bash
# Validate configuration file
vsr config validate

# Validate specific file
vsr config validate --config path/to/config.yaml

# Validate and show details
vsr config validate --verbose
```

### Configuration File Structure

```yaml
# config/config.yaml
bert_model:
  model_id: "your-model-id"
  threshold: 0.8

vllm_endpoints:
  - name: "primary"
    address: "127.0.0.1"
    port: 8000

model_config:
  your-model-id:
    pricing:
      prompt: 0.01
      completion: 0.02

default_model: "your-model-id"
```

## 🚢 Deployment

### Local Deployment

```bash
# Deploy locally (runs as background process)
vsr deploy local

# Deploy with custom config
vsr deploy local --config custom-config.yaml

# Check status
vsr status

# Stop
vsr undeploy local
```

### Docker Compose Deployment

```bash
# Deploy with Docker Compose
vsr deploy docker

# Deploy with observability disabled
vsr deploy docker --with-observability=false

# Stop and remove volumes
vsr undeploy docker --volumes
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
vsr deploy kubernetes

# Deploy to specific namespace
vsr deploy kubernetes --namespace production

# Check status
vsr status --namespace production

# Undeploy and wait for cleanup
vsr undeploy kubernetes --namespace production --wait
```

### Helm Deployment

```bash
# Deploy using Helm
vsr deploy helm

# Deploy with custom release name
vsr deploy helm --release-name my-router --namespace production

# Deploy with custom values
vsr deploy helm --set replicas=3 --set resources.memory=4Gi

# Upgrade release
vsr upgrade helm --namespace production

# Undeploy
vsr undeploy helm --namespace production --wait
```

## 📖 Commands

### Deployment Commands

| Command | Description |
|---------|-------------|
| `vsr deploy [env]` | Deploy router to specified environment |
| `vsr undeploy [env]` | Remove router deployment |
| `vsr upgrade [env]` | Upgrade router to latest version |
| `vsr status` | Check router and components status |
| `vsr start` | Start router service (deprecated) |
| `vsr stop` | Stop router service (deprecated) |
| `vsr restart` | Restart router service (deprecated) |

### Configuration Commands

| Command | Description |
|---------|-------------|
| `vsr init` | Initialize new configuration file |
| `vsr config validate` | Validate configuration |
| `vsr config view` | View current configuration |
| `vsr config set [key] [value]` | Set configuration value |

### Model Commands

| Command | Description |
|---------|-------------|
| `vsr model list` | List all models |
| `vsr model info [id]` | Show model details |
| `vsr model validate [id]` | Validate model integrity |
| `vsr model remove [id]` | Remove downloaded model |
| `vsr model download` | Download models |

### Monitoring Commands

| Command | Description |
|---------|-------------|
| `vsr logs` | Fetch router logs |
| `vsr status` | Check deployment status |
| `vsr health` | Quick health check |
| `vsr metrics` | Display router metrics |
| `vsr dashboard` | Open dashboard in browser |

### Debug Commands

| Command | Description |
|---------|-------------|
| `vsr debug` | Run interactive debugging session |
| `vsr health` | Perform health check |
| `vsr diagnose` | Generate diagnostic report |

### Other Commands

| Command | Description |
|---------|-------------|
| `vsr test-prompt [text]` | Send test prompt to router |
| `vsr install` | Install semantic router |
| `vsr get [resource]` | Get resource information |

## 🔄 Common Workflows

### First-Time Setup

```bash
# 1. Initialize configuration
vsr init

# 2. Download models
make download-models

# 3. Validate configuration
vsr config validate

# 4. Deploy locally for testing
vsr deploy local

# 5. Test with a prompt
vsr test-prompt "Hello, router!"

# 6. Check status and logs
vsr status
vsr logs --tail 50
```

### Development Workflow

```bash
# Start local deployment
vsr deploy local

# Make code changes
# ...

# Upgrade deployment
vsr upgrade local --force

# View logs in real-time
vsr logs --follow

# Test changes
vsr test-prompt "Test prompt"

# Stop when done
vsr undeploy local
```

### Production Deployment

```bash
# 1. Validate configuration
vsr config validate

# 2. Run diagnostics
vsr debug

# 3. Deploy to Kubernetes
vsr deploy kubernetes --namespace production

# 4. Verify deployment
vsr status --namespace production
vsr health

# 5. Monitor
vsr logs --namespace production --follow
vsr metrics --watch

# 6. Access dashboard
vsr dashboard --namespace production
```

### Troubleshooting Workflow

```bash
# 1. Check health
vsr health

# 2. Run full diagnostics
vsr debug

# 3. Check deployment status
vsr status

# 4. View recent logs
vsr logs --tail 100 --grep error

# 5. Generate diagnostic report
vsr diagnose --output diagnostics.txt

# 6. Validate models
vsr model validate --all

# 7. Check specific components
vsr logs --component router --since 10m
```

### Upgrade Workflow

```bash
# 1. Check current status
vsr status

# 2. Backup configuration
cp config/config.yaml config/config.yaml.bak

# 3. Pull latest changes
git pull origin main

# 4. Rebuild
make build-cli

# 5. Upgrade deployment
vsr upgrade kubernetes --namespace production --wait

# 6. Verify upgrade
vsr health
vsr logs --tail 50

# 7. Rollback if needed
git checkout <previous-commit>
vsr upgrade kubernetes --namespace production
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Configuration Validation Fails

```bash
# Check what's wrong
vsr config validate --verbose

# Common issues:
# - Missing required fields
# - Invalid YAML syntax
# - Model references not found
# - Invalid endpoint addresses

# Solution: Fix the issues and validate again
vsr config validate
```

#### 2. Models Not Found

```bash
# Check model status
vsr model list

# Download models
make download-models

# Or manually download specific model
# (future feature)
vsr model download [model-id]

# Validate models
vsr model validate --all
```

#### 3. Deployment Fails

```bash
# Run diagnostics
vsr debug

# Check prerequisites
# - kubectl installed? (for K8s)
# - docker running? (for Docker)
# - helm installed? (for Helm)

# Check resources
# - Disk space available?
# - Ports available?
# - Network connectivity?

# View detailed logs
vsr logs --tail 100
```

#### 4. Port Already in Use

```bash
# Check which ports are in use
vsr debug

# Find process using port
netstat -tulpn | grep 8080

# Kill process or use different port
# (configure in config.yaml)
```

#### 5. Kubernetes Deployment Issues

```bash
# Check cluster connection
kubectl cluster-info

# Check namespace
kubectl get namespaces

# Check pods
kubectl get pods -n [namespace]

# View pod logs
kubectl logs -n [namespace] [pod-name]

# Or use vsr
vsr logs --namespace [namespace] --follow
```

### Debug Mode

```bash
# Run comprehensive diagnostics
vsr debug

# This checks:
# ✓ Prerequisites (Go, kubectl, docker, helm, make)
# ✓ Configuration (file exists, valid YAML, passes validation)
# ✓ Models (directory exists, models downloaded)
# ✓ Resources (disk space, port availability)
# ✓ Connectivity (endpoint reachability)

# Provides recommendations based on failures
```

### Health Check

```bash
# Quick health check
vsr health

# Status indicators:
# 🟢 GOOD - All systems operational
# 🟡 DEGRADED - Environment ready, router not running
# 🔴 POOR - Critical issues detected
```

### Getting Help

```bash
# General help
vsr --help

# Command-specific help
vsr deploy --help
vsr model list --help

# View examples
vsr upgrade --help  # Shows examples in help text
```

## 📊 Advanced Features

### Log Filtering

```bash
# Filter by component
vsr logs --component router

# Filter by time
vsr logs --since 10m
vsr logs --since 1h

# Filter by pattern
vsr logs --grep error
vsr logs --grep "HTTP 500"

# Combine filters
vsr logs --component router --since 10m --grep error --follow
```

### Multi-Format Output

```bash
# JSON output
vsr model list --output json

# YAML output
vsr model list --output yaml

# Table output (default)
vsr model list --output table
```

### Environment Variables

```bash
# Set default config path
export VSR_CONFIG=config/production.yaml

# Set default namespace
export VSR_NAMESPACE=production

# Enable verbose output
export VSR_VERBOSE=true

# Use in commands
vsr deploy kubernetes  # Uses VSR_CONFIG and VSR_NAMESPACE
```

### Shell Completion

```bash
# Generate bash completion
vsr completion bash > /etc/bash_completion.d/vsr

# Generate zsh completion
vsr completion zsh > "${fpath[1]}/_vsr"

# Generate fish completion
vsr completion fish > ~/.config/fish/completions/vsr.fish

# Source completion
source <(vsr completion bash)
```

## 🏗️ Architecture

### Command Structure

```
vsr
├── config          # Configuration management
├── deploy          # Deployment operations
├── undeploy        # Removal operations
├── upgrade         # Upgrade operations
├── status          # Status checking
├── logs            # Log fetching
├── model           # Model management
│   ├── list
│   ├── info
│   ├── validate
│   ├── remove
│   └── download
├── debug           # Debugging tools
├── health          # Health checking
├── diagnose        # Diagnostics
├── dashboard       # Dashboard access
├── metrics         # Metrics display
├── test-prompt     # Testing
├── install         # Installation
├── init            # Initialization
└── get             # Resource querying
```

### Deployment Detection

VSR automatically detects active deployments:

1. **Local**: Checks for PID file at `/tmp/vsr-local-deployment.pid`
2. **Docker**: Queries Docker for containers matching `semantic-router`
3. **Kubernetes**: Queries kubectl for pods with label `app=semantic-router`
4. **Helm**: Lists Helm releases containing `semantic-router`

### Model Discovery

VSR discovers models using intelligent architecture selection:

1. **Priority**: BERT → RoBERTa → ModernBERT
2. **Types**: LoRA models (preferred) or Legacy models
3. **Categories**: Intent, PII, Security classifiers

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/vllm-project/semantic-router.git
cd semantic-router/src/semantic-router

# Install dependencies
go mod download

# Build
make build-cli

# Run tests
go test ./cmd/vsr/commands/... -v
go test ./pkg/cli/... -v

# Run linting
golangci-lint run
```

### Adding a New Command

1. Create command file in `cmd/vsr/commands/`
2. Implement `New[Command]Cmd() *cobra.Command`
3. Add command to `main.go`
4. Add help text and examples
5. Write tests
6. Update documentation

### Code Style

- Follow Go best practices
- Use Cobra patterns for commands
- Include comprehensive help text
- Add examples to help text
- Write table-driven tests
- Use existing CLI utilities (`pkg/cli`)

## 📝 License

Apache License 2.0 - See [LICENSE](../../LICENSE) for details.

## 🔗 Links

- [Main Repository](https://github.com/vllm-project/semantic-router)
- [Documentation](https://docs.vllm-project.com)
- [Issue Tracker](https://github.com/vllm-project/semantic-router/issues)
- [Discussions](https://github.com/vllm-project/semantic-router/discussions)

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vllm-project/semantic-router/discussions)
- **Email**: support@vllm-project.com

## 🙏 Acknowledgments

Built with:

- [Cobra](https://github.com/spf13/cobra) - CLI framework
- [vLLM](https://github.com/vllm-project/vllm) - Inference engine
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) - Model architecture

---

**Made with ❤️ by the vLLM Semantic Router team**
