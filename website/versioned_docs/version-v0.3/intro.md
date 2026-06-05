---
sidebar_position: 1
description: Research-driven introduction to vLLM Semantic Router, an Envoy-based control plane for signal-aware LLM routing, policy enforcement, and token efficiency.
---

# vLLM Semantic Router

vLLM Semantic Router is a research-driven project focused on frontier problems in
**LLMRouting** and **Token economy**. We build system-level intelligence for
Mixture-of-Models (MoM): deciding how to capture the right signals, select the
right model path, enforce the right policy, and spend the right token budget for
each request.

The project sits between clients and model backends as an Envoy External
Processor (`ext_proc`), turning routing from ad hoc application logic into an
observable, configurable control plane for multi-model systems.

## Research Focus

We use the project to answer a small set of hard systems questions:

1. **How do we capture missing signals** from requests, responses, users, and runtime context?
2. **How do we compose those signals** into robust routing and policy decisions?
3. **How do multiple models collaborate** as a system instead of serving as isolated endpoints?
4. **How do we optimize latency, spend, and tool usage** as part of a practical token economy?
5. **How do we add safety, feedback, and observability** without fragmenting the serving stack?

## Core System

### Signal and Projection Routing

Captures **16 maintained signal families** and coordinates them with reusable
projections before route selection:

| Layer           | Components                                                                                                                                                               | Role                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| **Signals**     | `authz`, `context`, `keyword`, `language`, `structure`, `complexity`, `domain`, `embedding`, `kb`, `modality`, `fact-check`, `jailbreak`, `pii`, `preference`, `reask`, `user-feedback` | Extract reusable request, safety, follow-up, and preference facts |
| **Projections** | `partitions`, `scores`, `mappings`                                                                                                                                       | Coordinate competing matches and emit named routing bands |
| **Decisions**   | AND/OR policy rules over signals and projections                                                                                                                         | Select the active route and model candidates              |

**How it works**: Signals are extracted from requests, projections coordinate
matched evidence, decision rules evaluate the resulting facts, and the chosen
route drives plugins plus model dispatch.

### Plugin Chain Architecture

Extensible plugin system for request/response processing:

| Plugin Type         | Description                                   | Use Case                                      |
| ------------------- | --------------------------------------------- | --------------------------------------------- |
| **semantic-cache**  | Semantic similarity-based caching             | Reduce latency and costs for similar queries  |
| **jailbreak**       | Adversarial prompt detection                  | Block prompt injection and jailbreak attempts |
| **pii**             | Personally identifiable information detection | Protect sensitive data and ensure compliance  |
| **system_prompt**   | Dynamic system prompt injection               | Add context-aware instructions per route      |
| **header_mutation** | HTTP header manipulation                      | Control routing and backend behavior          |
| **hallucination**   | Token-level hallucination detection           | Real-time fact verification during generation |

**How it works**: Plugins form a processing chain, each plugin can inspect/modify requests and responses, with configurable enable/disable per decision.

## Key Benefits

### A Control Plane for LLMRouting

- **Policy instead of hard-coded branches**: Move routing logic out of application code into reusable signals, decisions, and configuration.
- **Capability-aware selection**: Route by task shape, risk, and quality requirements instead of defaulting every request to one model.

### A Practical Token Economy Layer

- **Spend budget where it matters**: Reserve premium models, long context, and tool calls for the requests that need them.
- **Reduce waste without collapsing quality**: Use semantic caching, context-aware routing, and explicit policy to control latency and token spend.

### Governance in the Request Path

- **Built-in safety and compliance**: Apply jailbreak, PII, hallucination, prompt, and header controls at the same layer that makes routing decisions.
- **Observable decisions**: Keep routing and policy outcomes auditable so teams can tune behavior with data instead of guesswork.

### A Research Surface That Can Ship

- **Fast experimentation**: Add new signals, algorithms, and plugins without rewriting the serving path.
- **Production alignment**: Connect experimentation, observability, and deployment in one maintained system.

## Use Cases

- **Multi-model inference gateways**: Route to specialized models based on capability, context, and policy.
- **Cost-aware copilots**: Balance quality, latency, and spend for internal assistants and developer tooling.
- **Safety-sensitive assistants**: Enforce PII, jailbreak, and hallucination controls in the live request path.
- **Research platforms**: Evaluate routing policies, collect feedback signals, and iterate on model collaboration strategies.

## Start Here

- [**Overview**](overview/goals) for project goals, semantic routing concepts, and collective intelligence.
- [**Installation**](installation) for setup, deployment options, and configuration.
- [**Fleet Simulator**](fleet-sim/overview) for planning GPU fleets, evaluating routing strategies, and reading the guide PDF.
- [**Capacities**](tutorials/signal/overview) for signals, projections, decisions, plugins, algorithms, and global controls.
- [**Proposals**](proposals/unified-config-contract-v0-3) for design work that has not yet been folded into the stable docs set.

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/vllm-project/semantic-router/blob/main/LICENSE) file for details.
