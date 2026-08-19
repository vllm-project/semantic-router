"""Help text for runtime-oriented Click commands."""

SERVE_HELP = """
Start vLLM Semantic Router.

With no MODEL, serve uses --config or config.yaml and preserves the existing
Dashboard-first setup flow. MODEL operands select installed catalog virtual
entrypoints such as vllm-sr/mom-v1-blend for the local Docker target.

Virtual models are routing policies. Semantic Router starts Router, Envoy, the
Dashboard, and supporting services; it does not download or launch the physical
LLM engines referenced by provider backends. Connect user-owned single or
multiple model endpoints through one canonical config or the Dashboard.

Ports are configured in the selected config under the listeners section.

DEPLOYMENT TARGETS:

\b
docker  - Local Docker deployment (default)
k8s     - Kubernetes deployment via Helm

MODEL SELECTION ALGORITHMS:

\b
static     - Use first configured model (default, no learning)
router_dc  - Query-model matching via embedding similarity
automix    - Cost-quality optimization using POMDP
hybrid     - Combine multiple methods with configurable weights
workflows  - Router Flow static/dynamic micro-agent orchestration
latency_aware - TPOT/TTFT percentile-aware selection
knn        - KNN selector using shared ML model-selection settings
kmeans     - KMeans selector using shared ML model-selection settings
svm        - SVM selector using shared ML model-selection settings
mlp        - MLP selector using shared ML model-selection settings
multi_factor - Quality, latency, cost, and load scoring

Cross-request learning lives under global.router.learning.adaptation and
global.router.learning.protection instead of --algorithm. Catalog MODEL
operands retain their verified recipe algorithms; fork and serve an edited
config when you need an algorithm override.

Examples:

\b
  # Dashboard-first setup or an existing ./config.yaml
  vllm-sr serve
  # One installed virtual model
  vllm-sr serve vllm-sr/mom-v1-blend
  # Multiple virtual entrypoints sharing one provider/backend pool
  vllm-sr serve vllm-sr/mom-v1-lite vllm-sr/mom-v1-flash
  # User-owned single or multi-model topology
  vllm-sr serve --config my-models.yaml
  # Deploy a user-owned config to Kubernetes
  vllm-sr serve --target k8s --config my-models.yaml --namespace my-ns
  # Runtime policy and image overrides
  vllm-sr serve --algorithm latency_aware
  vllm-sr serve --image-pull-policy always
  vllm-sr serve --readonly
  vllm-sr serve --minimal
  vllm-sr serve --log-level debug
  # AMD ROCm image, device passthrough, and router internal GPU defaults
  vllm-sr serve --platform amd
  VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES=7 vllm-sr serve --platform amd
"""
