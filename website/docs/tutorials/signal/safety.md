# Safety Signals

## Overview

Safety signals detect policy-sensitive requests before the router commits to a route.

This page covers the safety-oriented signal families that map to:

- `config/signal/jailbreak/`
- `config/signal/pii/`
- `config/signal/authz/`

## Key Advantages

- Moves security checks into the same decision graph as routing logic.
- Lets one policy feed both route selection and later plugins.
- Keeps authorization, privacy, and jailbreak detection explicit in config.
- Makes safety behavior easier to review than implicit downstream filtering.

## What Problem Does It Solve?

If safety logic only exists downstream, the router can still send unsafe traffic to the wrong model or path. If it only exists as ad hoc filters, policy becomes hard to audit and hard to reuse.

Safety signals solve that by making security and access checks first-class routing inputs.

## When to Use

Use safety signals when:

- unsafe traffic must be blocked or downgraded before model selection
- privacy-sensitive traffic needs a different route or plugin stack
- route eligibility depends on user tier, tenant role, or policy membership
- you want a visible policy boundary inside `routing.decisions`

## Configuration

Configure the relevant signal family under `routing.signals`.

### Jailbreak

Source fragment family: `config/signal/jailbreak/`

```yaml
routing:
  signals:
    jailbreak:
      - name: prompt_injection
        threshold: 0.65
```

Use this when a decision should block or downgrade prompt-injection traffic.

### PII

Source fragment family: `config/signal/pii/`

```yaml
routing:
  signals:
    pii:
      - name: strict_pii
        threshold: 0.9
        pii_types_allowed: []
```

Use this when personal data detection must influence route selection or a later plugin.

### Authz

Source fragment family: `config/signal/authz/`

```yaml
routing:
  signals:
    role_bindings:
      - name: admin_only
        role: admin
        subjects:
          - kind: Group
            name: admins
```

Use this when routing depends on user role, tenant tier, or policy membership.
