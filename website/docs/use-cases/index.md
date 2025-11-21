---
sidebar_position: 1
---

# Use Cases

This section covers proposed use cases and scenarios for the vLLM Semantic Router. These features are currently **Work in Progress** and not yet implemented in the current version.

:::warning Work in Progress
The use cases described in this section are **proposed features** that are not yet implemented in the current version of the semantic router. They represent future capabilities that would enhance the router's functionality.
:::

## Proposed Enterprise Use Cases

### [Multi-Provider Routing](multi-provider-routing.md) 🚧

**Status: Work in Progress**

Route queries to the best AI provider for each specific task - coding questions to Claude, business writing to GPT-4, research to Gemini, and sensitive data to internal models.

**Key Scenarios:**
- **Tech Startups**: Different teams get specialized AI assistance (engineering → Claude, marketing → GPT-4, legal → internal)
- **Consulting Firms**: Route strategic projects to premium models, routine tasks to cost-effective ones
- **Financial Services**: Balance external AI capabilities with regulatory compliance requirements

**Proposed Benefits:**
- Get the best AI for each specific task instead of one-size-fits-all
- Cost optimization by using appropriate models for each query type
- Automatic compliance routing for sensitive data
- Leverage each provider's unique strengths

### [Keyword-Based Routing](keyword-based-routing.md) 🚧

**Status: Work in Progress**

Implement transparent, deterministic routing rules for enterprise security and compliance. Route queries based on keywords and business policies with complete visibility into routing decisions.

**Key Scenarios:**
- **Financial Services**: Ensure confidential trading strategies never reach external AI providers
- **Healthcare**: Route patient data to HIPAA-compliant models while using external AI for general medical knowledge
- **Enterprise Search**: Route queries requiring web search to providers with search capabilities

**Proposed Benefits:**
- Complete data sovereignty and regulatory compliance
- Transparent, auditable routing decisions
- Sub-millisecond routing for deterministic cases
- Enterprise policy enforcement with full visibility

## Documentation Structure

Each proposed use case includes:
- **User Stories**: Real-world scenarios and challenges these features would solve
- **Business Value**: Clear benefits for users and organizations
- **Use Case Examples**: Specific scenarios with business needs and solutions
- **Next Steps**: Guidance for planning and implementation

## Contributing to Development

These use cases represent proposed features that could be implemented in future versions. We welcome contributions to help implement these capabilities! Please see our [Contributing Guide](https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md) for details on how to contribute to the development of these features.
