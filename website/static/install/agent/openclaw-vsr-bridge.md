# Agent-safe OpenClaw to VSR bridge

Install vLLM Semantic Router from <https://vllm-semantic-router.com/install.sh>
using the supported installer in agent-safe mode.

Run:

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --mode cli --runtime skip --no-launch
```

If the `openclaw-vsr-bridge` skill is available, use it.

Otherwise:

1. Locate the OpenClaw config file.
2. Run:

```bash
vllm-sr config import --from openclaw --source <openclaw.json> --target config.yaml
```

Then:

1. Report the rewritten OpenClaw base URL.
2. Suggest:

```bash
vllm-sr validate config.yaml
```
