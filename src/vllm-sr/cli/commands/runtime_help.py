"""Help text for runtime-oriented Click commands."""

SERVE_HELP = """
Start vLLM Semantic Router.

Serve reads ./config.yaml, accepts another canonical v0.3 config with --config,
or creates a secure local Management workspace when the default path is absent.
Semantic Router starts Router, Envoy, the Dashboard, PostgreSQL, and Valkey;
it does not download or launch physical LLM engines. Connect provider endpoints,
create Recipes, and publish Mixture-of-Model entrypoints in the Dashboard.
Prometheus, Grafana, and Jaeger are available with --with-observability.

Ports are configured in the selected config under the listeners section.

DEPLOYMENT TARGETS:

\b
docker  - Local Docker deployment (default)
k8s     - Kubernetes deployment via Helm

Examples:

\b
  # Start from the current workspace
  vllm-sr serve
  # Start from an explicit canonical v0.3 config
  vllm-sr serve --config /path/to/config.yaml
  # Deploy the current workspace config to Kubernetes
  vllm-sr serve --target k8s --namespace my-ns
  # Infrastructure and runtime overrides
  vllm-sr serve --image-pull-policy always
  vllm-sr serve --readonly
  vllm-sr serve --minimal
  vllm-sr serve --with-observability
  vllm-sr serve --log-level debug
  # AMD ROCm image, device passthrough, and router internal GPU defaults
  vllm-sr serve --platform amd
  VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES=7 vllm-sr serve --platform amd
"""
