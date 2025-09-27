# Multi-Cloud and Hybrid Cloud Routing

The Semantic Router now supports **inter-cluster and multi-cloud routing**, enabling sophisticated routing across multiple clusters, cloud providers, and deployment environments. This feature extends the existing semantic classification and routing capabilities to work seamlessly across distributed infrastructure.

## Overview

Multi-cloud routing allows you to:

- **Route across multiple clusters**: Distribute load across vLLM clusters in different regions or environments
- **Integrate cloud providers**: Route to OpenAI, Anthropic Claude, Grok, and other API providers
- **Optimize for performance**: Route based on latency, throughput, and availability
- **Control costs**: Route to the most cost-effective clusters and providers
- **Ensure compliance**: Route based on data residency and regulatory requirements
- **Provide fault tolerance**: Automatic failover and circuit breaker patterns

## Configuration

### Enabling Inter-Cluster Routing

Add the `inter_cluster_routing` section to your configuration:

```yaml
inter_cluster_routing:
  enabled: true
  
  cluster_discovery:
    method: "static"  # Options: "static", "kubernetes", "consul", "etcd"
    refresh_interval: "30s"
    health_check_interval: "10s"
```

### Cluster Configuration

Define your clusters in the `static_clusters` section:

```yaml
inter_cluster_routing:
  cluster_discovery:
    static_clusters:
      - name: "on-prem-gpu-cluster"
        location: "us-west-2"
        type: "vllm"
        endpoint: "https://on-prem.company.com:8000"
        authentication:
          type: "bearer"
          token: "bearer-token-secret"
        models:
          - "llama-2-70b"
          - "codellama-34b"
        capabilities:
          max_context_length: 4096
          max_tokens_per_second: 100
        performance:
          avg_latency_ms: 150
          throughput_rps: 50
          availability: 99.5
        compliance:
          - "hipaa"
          - "sox"
        cost_per_token: 0.001
```

For more detailed examples and configuration options, see the [complete multi-cloud configuration example](https://github.com/vllm-project/semantic-router/blob/main/config/multi-cloud-config-example.yaml).

## Routing Strategies

Configure sophisticated routing logic with prioritized strategies. Strategies are evaluated in priority order (higher number = higher priority).

### Example Strategy Configuration

```yaml
routing_strategies:
  # Highest Priority: Compliance-based routing
  - name: "gdpr-compliance-routing"
    priority: 300
    conditions:
      - type: "compliance_requirement"
        required_compliance: ["gdpr"]
    actions:
      - type: "route_to_cluster"
        target: "eu-west-cluster"
  
  # Medium Priority: Latency-optimized routing
  - name: "latency-optimized-routing"
    priority: 200
    conditions:
      - type: "latency_requirement"
        max_latency_ms: 200
    actions:
      - type: "route_to_cluster"
        target: "edge-cluster"
      - type: "failover"
        failover_targets: ["backup-cluster"]
```

## Supported Providers

The system supports routing to multiple types of providers:

- **vLLM Clusters**: Self-hosted vLLM deployments
- **OpenAI API**: GPT-4, GPT-3.5-turbo, and other OpenAI models
- **Anthropic Claude**: Claude-3 and related models
- **Grok API**: xAI's Grok models
- **Custom Providers**: Extensible plugin architecture

## Use Cases

### 1. On-Premises + Cloud Hybrid

Route sensitive data to on-premises clusters while using cloud providers for general queries:

```yaml
routing_strategies:
  - name: "sensitive-data-routing"
    priority: 300
    conditions:
      - type: "compliance_requirement"
        required_compliance: ["hipaa"]
    actions:
      - type: "route_to_cluster"
        target: "on-prem-secure-cluster"
```

### 2. Multi-Region GDPR Compliance

Ensure EU data stays in EU clusters:

```yaml
routing_strategies:
  - name: "eu-data-residency"
    priority: 300
    conditions:
      - type: "data_residency"
        required_region: "eu-west-1"
    actions:
      - type: "route_to_cluster"
        target: "eu-west-cluster"
```

### 3. Cost Optimization

Route to the most cost-effective clusters:

```yaml
routing_strategies:
  - name: "cost-optimization"
    priority: 150
    conditions:
      - type: "cost_sensitivity"
        max_cost_per_1k_tokens: 0.001
    actions:
      - type: "route_to_cluster"
        target: "cost-effective-cluster"
```

## Key Features

### ✅ **Condition-Based Routing**
- Latency requirements
- Cost sensitivity
- Compliance needs
- Data residency
- Model-specific routing

### ✅ **Multiple Action Types**
- Direct cluster routing
- Provider routing
- Load balancing
- Failover strategies

### ✅ **Fault Tolerance**
- Circuit breaker patterns
- Retry policies with backoff
- Automatic failover
- Health monitoring

### ✅ **Authentication Support**
- Bearer tokens
- API keys
- OAuth (future)
- Per-cluster/provider auth

### ✅ **Monitoring & Observability**
- Detailed routing logs
- Performance metrics
- Cost tracking
- Health status

## Migration from Single-Cluster

Existing configurations remain fully compatible! To add multi-cloud routing:

1. **Enable the feature**: Add `inter_cluster_routing.enabled: true`
2. **Define clusters**: Move existing endpoints to `static_clusters`
3. **Add strategies**: Configure routing logic based on your needs
4. **Test gradually**: Start with simple strategies and expand

## Getting Started

1. **Review the [complete configuration example](https://github.com/vllm-project/semantic-router/blob/main/config/multi-cloud-config-example.yaml)**
2. **Read the [detailed configuration guide](../getting-started/configuration.md)**
3. **Check the [API reference](../api/router/) for programmatic access**
4. **See [troubleshooting tips](../categories/technical-details.md) for common issues**

This powerful feature enables enterprise-grade routing across complex, distributed LLM infrastructure while maintaining the simplicity and intelligence of semantic classification.