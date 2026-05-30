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
rl_driven  - Online learning selector with optional persistence
thompson   - Thompson Sampling with exploration/exploitation (RL-driven)
gmtrouter  - Graph neural network for personalized routing (RL-driven)
router_r1  - LLM-as-router with think/route actions (RL-driven)

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

    # Platform branding (for AMD deployments)
    vllm-sr serve --platform amd
"""
