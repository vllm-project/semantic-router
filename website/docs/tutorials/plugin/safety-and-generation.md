# Safety and Generation Plugins

## Overview

This page covers the route-local plugins that enforce safety or generation-time controls after a decision already matched.

It aligns to:

- `config/plugin/content-safety/`
- `config/plugin/jailbreak/`
- `config/plugin/pii/`
- `config/plugin/hallucination/`
- `config/plugin/response-jailbreak/`

## Key Advantages

- Applies safety controls only where they are needed.
- Complements signal-driven safety without forcing every route to share one policy.
- Supports route-local response screening for retrieved or generated content.
- Keeps generation-time enforcement visible in the route config.

## What Problem Does It Solve?

Sometimes a request is safe enough to route, but the selected route still needs additional output checks or safety enforcement. If every route inherits the same behavior, policy becomes blunt and expensive.

These plugins solve that by making post-match safety and generation controls route-local.

## When to Use

Use this family when:

- one route needs compact route-local safety controls
- output should be screened after the decision matches
- retrieved or tool-based routes need hallucination checks
- response content needs a final jailbreak screen before returning

## Configuration

Each plugin lives inside `routing.decisions[].plugins`.

### Content Safety

```yaml
routing:
  decisions:
    - name: moderated_route
      plugins:
        - type: content-safety
          configuration:
            enabled: true
            mode: hybrid
```

### Jailbreak and PII

```yaml
routing:
  decisions:
    - name: guarded_route
      plugins:
        - type: jailbreak
          configuration:
            enabled: true
            action: block
        - type: pii
          configuration:
            enabled: true
            pii_types_allowed: []
```

### Hallucination

```yaml
routing:
  decisions:
    - name: grounded_route
      plugins:
        - type: hallucination
          configuration:
            enabled: true
            hallucination_action: header
```

### Response Jailbreak

```yaml
routing:
  decisions:
    - name: screened_response_route
      plugins:
        - type: response_jailbreak
          configuration:
            enabled: true
            action: block
```
