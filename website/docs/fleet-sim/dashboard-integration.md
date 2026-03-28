---
title: Dashboard Integration
---

# Dashboard Integration

Fleet Sim is integrated into the dashboard through the backend proxy layer. The
dashboard does not talk to a deprecated standalone simulator frontend.

## Default local behavior

When you run:

```bash
cd src/vllm-sr
vllm-sr serve --image-pull-policy never
```

the CLI starts `vllm-sr-sim` as a sibling container on the same runtime network and
sets `TARGET_FLEET_SIM_URL` inside the router stack to the sidecar service URL.

## Proxy path

The dashboard backend proxies simulator requests at:

```text
/api/fleet-sim/*
```

If the simulator is not configured, that proxy surface returns a structured
`Service not available` response instead of silently failing.

## External service mode

Set `TARGET_FLEET_SIM_URL` when the simulator lives in another container, host, or
environment:

```bash
export TARGET_FLEET_SIM_URL=http://your-simulator:8000
```

With that variable present, `vllm-sr serve` uses the external simulator and skips the
default local sidecar startup.

## Dashboard surfaces

The top-bar `Fleet Sim` menu exposes:

- `Overview` for high-level simulator state and recent assets
- `Workloads` for built-in workload libraries and trace inputs
- `Fleets` for saved fleet definitions and planning outputs
- `Runs` for optimize, simulate, and what-if jobs
