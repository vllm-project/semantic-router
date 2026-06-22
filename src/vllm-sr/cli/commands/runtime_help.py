"""Help text for runtime-oriented Click commands."""

SERVE_HELP = """
Start vLLM Semantic Router.

Ports are configured in config.yaml under 'listeners' section.

DEPLOYMENT TARGETS:

\b
docker  - Local Docker deployment (default)
k8s     - Kubernetes deployment via Helm

MODEL SELECTION ALGORITHMS:

\b
static     - Use first configured model (default, no learning)
elo        - Rating-based selection using user feedback
router_dc  - Query-model matching via embedding similarity
automix    - Cost-quality optimization using POMDP
hybrid     - Combine multiple methods with configurable weights
gmtrouter  - Graph neural network for personalized routing (RL-driven)
latency_aware - TPOT/TTFT percentile-aware selection
knn        - KNN selector using shared ML model-selection settings
kmeans     - KMeans selector using shared ML model-selection settings
svm        - SVM selector using shared ML model-selection settings
mlp        - MLP selector using shared ML model-selection settings
multi_factor - Quality, latency, cost, and load scoring
rl_driven  - Online learning selector with optional Thompson / Router-R1 modes

Examples:
    # Basic usage (uses config.yaml, Docker target)
    vllm-sr serve

    # Deploy to Kubernetes
    vllm-sr serve --target k8s --namespace my-ns --profile dev

    # Custom config file
    vllm-sr serve --config my-config.yaml

    # Use Elo rating selection (learns from feedback)
    vllm-sr serve --algorithm elo

    # Use cost-optimized selection
    vllm-sr serve --algorithm automix

    # Use latency-aware selection
    vllm-sr serve --algorithm latency_aware

    # Custom image
    vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

    # Pull policy
    vllm-sr serve --image-pull-policy always

    # Read-only dashboard (for public beta)
    vllm-sr serve --readonly

    # Minimal mode (no dashboard, no observability)
    vllm-sr serve --minimal

    # Start router with debug logs
    vllm-sr serve --log-level debug

    # AMD ROCm image, device passthrough, and router internal GPU defaults
    vllm-sr serve --platform amd

    # Preserve CPU settings for router internal signal models on AMD
    VLLM_SR_AMD_PRESERVE_CPU=1 vllm-sr serve --platform amd
"""
