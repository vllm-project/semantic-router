# Agent-safe vLLM Semantic Router install

Install vLLM Semantic Router from <https://vllm-sr.ai/install.sh>
using the supported installer in agent-safe mode.

Run:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --mode cli --runtime skip --no-launch
```

Then verify the install with:

```bash
vllm-sr --version
```
