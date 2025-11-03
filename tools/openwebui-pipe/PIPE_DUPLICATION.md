# vLLM Semantic Router Pipe - Deployment Duplication

## Overview

The `vllm_semantic_router_pipe.py` file exists in three locations for different deployment scenarios:

1. **docker-compose**: `deploy/docker-compose/addons/vllm_semantic_router_pipe.py`
2. **kubernetes**: `deploy/kubernetes/observability/pipelines/vllm_semantic_router_pipe.py`  
3. **openwebui**: `tools/openwebui-pipe/vllm_semantic_router_pipe.py`

These files are **intentionally duplicated** because OpenWebUI pipes are designed to be standalone, single-file deployments that users can copy and use independently.

## Differences

The three versions differ only in default configuration values:

### docker-compose version:
- `vsr_base_url`: `"http://envoy-proxy:8801"`
- `pipeline_id`: `"auto"`
- `pipeline_name`: `"vsr/"`
- `model_name`: `"auto"` (sent to VSR backend)

### kubernetes version:
- `vsr_base_url`: `"http://localhost:8000"`
- `pipeline_id`: `"vllm_semantic_router"`
- `pipeline_name`: `"vllm-semantic-router/"`
- `model_name`: `"auto"` (sent to VSR backend)

### openwebui version:
- `vsr_base_url`: `"http://localhost:8000"`
- `pipeline_id`: `"auto"`
- `pipeline_name`: `"vllm-semantic-router/"`
- `model_name`: `"MoM"` (sent to VSR backend, backward compatible with "auto")

## Current State

**Total duplication**: 648 lines × 3 files = **1,944 lines of duplicated code**

## Consolidation Strategy

### Option 1: Environment Variables (Recommended for future)

Create a single canonical version that reads configuration from environment variables:

```python
class Pipeline:
    def __init__(self):
        self.vsr_base_url = os.getenv("VSR_BASE_URL", "http://localhost:8000")
        self.pipeline_id = os.getenv("PIPELINE_ID", "auto")
        self.pipeline_name = os.getenv("PIPELINE_NAME", "vllm-semantic-router/")
        self.model_name = os.getenv("VSR_MODEL_NAME", "auto")
        # ... rest of initialization
```

**Pros**: Single source of truth, easy to maintain
**Cons**: Users need to set environment variables for each deployment

### Option 2: Keep as Deployment-Specific Copies (Current approach)

Keep the three separate files but establish a sync process:

1. Designate one file as the "canonical" version (e.g., `tools/openwebui-pipe/vllm_semantic_router_pipe.py`)
2. When updating logic, update the canonical version first
3. Use a script to sync changes to deployment-specific copies, preserving only configuration differences
4. Add pre-commit hooks to verify synchronization

**Pros**: Maintains standalone deployment convenience
**Cons**: Requires discipline to keep synchronized

### Option 3: Shared Base with Deployment Wrappers

Create a shared base module and thin deployment-specific wrappers:

```
tools/openwebui-pipe/
├── vllm_semantic_router_pipe_base.py  (common ~640 lines)
├── vllm_semantic_router_pipe_docker.py  (wrapper ~8 lines)
├── vllm_semantic_router_pipe_k8s.py  (wrapper ~8 lines)
└── vllm_semantic_router_pipe.py  (wrapper ~8 lines)
```

**Pros**: Reduces duplication significantly, easy to maintain common code
**Cons**: Not truly standalone (requires copying base module), adds complexity

## Recommendation

**Short term**: Keep current structure but:
- Add this documentation
- Establish one file as canonical
- Create sync script for updates

**Long term**: Migrate to Option 1 (environment variables) when OpenWebUI pipe deployment process is standardized

## Sync Script

A sync script template is provided in `tools/openwebui-pipe/sync_pipe_versions.sh`:

```bash
#!/bin/bash
# Sync pipe versions while preserving configuration differences
# Usage: ./sync_pipe_versions.sh
```

## Maintenance Guidelines

When updating the pipe logic:

1. Update the canonical version: `tools/openwebui-pipe/vllm_semantic_router_pipe.py`
2. Run the sync script to propagate changes
3. Verify configuration differences are preserved
4. Test in all three deployment scenarios
5. Commit all three files together with a note about synchronization
