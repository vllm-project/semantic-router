# TD010: Public Website Capability Contract Has Drifted from the Current Repo Surfaces

## Status

Open

## Scope

public website, guides, and capability messaging

## Summary

The public website currently mixes multiple incompatible product contracts: the dashboard-first `vllm-sr serve` bootstrap flow, the current CLI user-config schema (`providers`, `signals`, `decisions`), and legacy/raw router guidance centered on `config/config.yaml` and `make run-router`.

This is no longer limited to internal architecture debt. The latest public site exposes the drift directly through installation, configuration, overview, tutorial, troubleshooting, and community pages. A representative scan found at least 14 website files that still refer to `config/config.yaml`, 9 that tell users to run `make run-router`, and 5 that advertise decision plugin types such as `jailbreak` or `pii` that are not valid in the current CLI plugin schema.

## Evidence

- [website/docs/installation/installation.md](../../../website/docs/installation/installation.md)
- [website/docs/installation/configuration.md](../../../website/docs/installation/configuration.md)
- [website/docs/overview/semantic-router-overview.md](../../../website/docs/overview/semantic-router-overview.md)
- [website/docs/community/overview.md](../../../website/docs/community/overview.md)
- [src/vllm-sr/cli/bootstrap.py](../../../src/vllm-sr/cli/bootstrap.py)
- [src/vllm-sr/cli/commands/runtime.py](../../../src/vllm-sr/cli/commands/runtime.py)
- [src/vllm-sr/cli/models.py](../../../src/vllm-sr/cli/models.py)
- [src/vllm-sr/tests/test_config_template.py](../../../src/vllm-sr/tests/test_config_template.py)
- [docs/agent/tech-debt/TD001-config-surface-fragmentation.md](TD001-config-surface-fragmentation.md)
- [docs/agent/tech-debt/TD004-python-cli-kubernetes-workflow-separation.md](TD004-python-cli-kubernetes-workflow-separation.md)

## Why It Matters

- The website is the latest public product surface. When it presents incompatible install and config stories, users cannot tell which path is canonical, which is advanced, and which is legacy.
- The current CLI explicitly supports dashboard-first bootstrap plus a lean YAML-first sample, but public docs still describe large areas of the product using raw router config and internal developer Make targets.
- Public examples currently advertise unsupported decision plugin types such as `jailbreak` and `pii`, which means some public examples cannot pass the current CLI schema as written.
- This makes the existing config and environment split debt from TD001 and TD004 user-visible instead of staying an internal implementation concern.

## Desired End State

- One explicit public capability contract that distinguishes local CLI bootstrap, YAML-first CLI authoring, raw router config, dashboard editing, and Kubernetes/operator deployment.
- Public pages declare which contract they target instead of silently mixing CLI, router-internal, and contributor-only flows.
- Representative public examples are generated from, or mechanically validated against, the schema they claim to document.
- Capability messaging on the homepage, overview, guides, and tutorials is tied to implemented and tested surfaces rather than aspirational or legacy behavior.

## Exit Criteria

- Public install, configuration, tutorial, troubleshooting, and community guides no longer mix `vllm-sr serve`, `providers`-style CLI config, and `config/config.yaml` or `make run-router` workflows without explicit scoping.
- Public examples that target the CLI parse against the current `UserConfig` contract, and raw router/operator examples are clearly labeled and validated against their own contract.
- Public docs do not advertise unsupported decision plugin types or other invalid config fields for the targeted workflow.
- The website or CI path includes mechanical checks that keep representative examples aligned with the canonical public capability contract.
