---
title: Common Errors
sidebar_label: Common Errors
---

# Common Errors and Fixes

This guide provides a quick reference for common log messages and errors you may encounter when running vLLM Semantic Router. Each section maps error patterns to their root causes and configuration fixes.

:::tip
Use the [Quick Diagnostic Commands](#quick-diagnostic-commands) at the end of this page to quickly identify issues.
:::

## Configuration Loading Errors

### Failed to create ExtProc server

**Log Pattern:**

```
Failed to create ExtProc server: <error>
```

**Causes & Fixes:**

| Cause                   | Fix                                                 |
| ----------------------- | --------------------------------------------------- |
| Invalid config path     | Verify `--config` flag points to existing YAML file |
| YAML syntax error       | Validate YAML with `yq` or online validator         |
| Missing required fields | Check all required fields are present               |

```bash
# Verify config path
./router --config /app/config/config.yaml
```

---

### Failed to read config file

**Log Pattern:**

```
failed to read config file: <error>
```

**Fixes:**

- Verify file exists: `ls -la config/config.yaml`
- Check permissions: `chmod 644 config/config.yaml`
- Ensure path is absolute or correct relative path

> See code: [cmd/main.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/cmd/main.go).

---

## Cache & Storage Errors

### Milvus config path is required

**Log Pattern:**

```
milvus config path is required
```

**Fix:** Set `backend_config_path` when using Milvus backend:

```yaml
semantic_cache:
  enabled: true
  backend_type: "milvus"
  backend_config_path: "config/milvus.yaml" # ← Add this
```

---

### Index does not exist and auto-creation is disabled

**Log Pattern:**

```
index <name> does not exist and auto-creation is disabled
```

**Fix:** Enable auto-creation in Redis/Milvus config:

```yaml
# In config/redis.yaml
index:
  auto_create: true # ← Enable this
```

---

### Redis store not yet implemented

**Log Pattern:**

```
redis store not yet implemented
```

**Note:** Redis response store is not yet available. Use `memory` or `milvus` instead:

```yaml
semantic_cache:
  backend_type: "memory" # or "milvus"
```

> See code: [pkg/cache](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/cache) AND [pkg/responsestore](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/responsestore).

---

## PII & Security Errors

### PII policy violation

**Log Pattern:**

```
PII policy violation for decision <name>: denied PII types [<types>]
```

**Fixes:**

1. **Allow the PII type** if it should be permitted:

```yaml
plugins:
  - type: "pii"
    configuration:
      pii_types_allowed:
        - "LOCATION" # Add denied type here
```

2. **Raise threshold** if false positives:

```yaml
classifier:
  pii_model:
    threshold: 0.95 # Increase from default 0.9
```

---

### Jailbreak detected

**Log Pattern:**

```
Jailbreak detected: type=<type>, confidence=<score>
```

**Fixes:**

1. **Raise threshold** to reduce false positives:

```yaml
prompt_guard:
  threshold: 0.8 # Increase from default 0.7
```

2. **Disable for specific decision**:

```yaml
decisions:
  - name: "internal_decision"
    jailbreak_enabled: false
```

> See code: [pii/policy.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/utils/pii/policy.go) AND [req_filter_jailbreak.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/extproc/req_filter_jailbreak.go).

---

## MCP Client Errors

### Either command or URL must be specified

**Log Pattern:**

```
either command or URL must be specified
```

**Fix:** Specify transport configuration:

```yaml
# For stdio transport
mcp_clients:
  my_client:
    transport_type: "stdio"
    command: "/path/to/mcp-server"

# For HTTP transport
mcp_clients:
  my_client:
    transport_type: "streamable-http"
    url: "http://localhost:8080"
```

---

### Command is required for stdio transport

**Log Pattern:**

```
command is required for stdio transport
```

**Fix:** Add command for stdio transport:

```yaml
mcp_clients:
  my_client:
    transport_type: "stdio"
    command: "python"
    args: ["-m", "my_mcp_server"]
```

> See code: [pkg/mcp/factory.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/mcp/factory.go).

---

## Endpoint Errors

### Invalid address format

**Log Pattern:**

```
invalid endpoint address: <address>
```

**Fixes:**

| Wrong                  | Correct                              |
| ---------------------- | ------------------------------------ |
| `http://10.0.0.1:8000` | `10.0.0.1` (address) + `8000` (port) |
| `vllm.example.com`     | Use IP address instead               |
| `10.0.0.1:8000`        | Separate address and port fields     |

```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: "10.0.0.1" # IP only, no protocol/port
    port: 8000 # Port separate
```

> See: [config.yaml#vllm_endpoints](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L43-L51) AND [pkg/extproc](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/extproc).

---

## Model Loading Errors

### Model not found

**Log Pattern:**

```
failed to load model: <path>
```

**Fixes:**

- Verify model path exists
- Check model is downloaded: `ls -la models/`
- Ensure path is accessible inside container

```yaml
bert_model:
  model_id: /app/models/all-MiniLM-L12-v2 # Use absolute path in container
```

---

## Performance Issues

### Low cache hit ratio

**Symptoms:** Cache rarely returns hits, high backend latency

**Fix:** Lower similarity threshold:

```yaml
semantic_cache:
  similarity_threshold: 0.75 # Lower from default 0.8

# Or per-decision
plugins:
  - type: "semantic-cache"
    configuration:
      similarity_threshold: 0.70
```

---

### Classification confidence too low

**Symptoms:** Many queries fall through to "other" category

**Fix:** Lower category threshold:

```yaml
classifier:
  category_model:
    threshold: 0.5 # Lower from default 0.6
```

---

## Quick Diagnostic Commands

```bash
# Check config syntax
yq eval '.' config/config.yaml

# Test endpoint connectivity
curl -s http://<address>:<port>/health

# Check model files
ls -la models/

# View recent logs
docker logs semantic-router --tail 100

# Check metrics
curl -s http://localhost:9190/metrics | grep semantic_router
```
