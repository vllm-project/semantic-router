# vSR Semantic Router BBR Plugin POC

This POC demonstrates integrating vLLM Semantic Router components as a **single combined BBR plugin** 
that works with the [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) 
BBR Pluggable Framework (PR #1981).

## 🎯 What This POC Demonstrates

A **single `SemanticRouterPlugin`** that combines:

1. **Category/Intent Classifier** - Classifies user intent (coding, math, creative, etc.)
2. **PII Detector** - Detects personally identifiable information (email, phone, SSN, etc.)
3. **Jailbreak Detector** - Detects prompt injection and jailbreak attempts

All three components run in a single `Execute()` call and return headers for downstream routing.

## 📁 Project Structure

```
poc-bbr-plugin/
├── go.mod                          # Module with GIE dependency
├── pkg/
│   ├── classifier/
│   │   ├── interfaces.go           # Classifier interfaces
│   │   └── mock.go                 # Mock implementation for testing
│   └── plugin/
│       └── semantic_router_plugin.go  # Combined BBRPlugin implementation
├── tests/
│   └── semantic_router_plugin_test.go # Comprehensive tests
└── cmd/
    └── demo/
        └── main.go                 # Demo server
```

## 🔌 BBR Integration

The plugin implements the `BBRPlugin` interface from PR #1981:

```go
type BBRPlugin interface {
    plugins.Plugin  // TypedName()
    RequiresFullParsing() bool
    Execute(requestBodyBytes []byte) (headers map[string]string, mutatedBodyBytes []byte, err error)
}
```

### Headers Output

| Header | Description | Example |
|--------|-------------|---------|
| `X-Gateway-Intent-Category` | Classified category | `coding` |
| `X-Gateway-Intent-Confidence` | Classification confidence | `0.9200` |
| `X-Gateway-PII-Detected` | Whether PII was found | `true` |
| `X-Gateway-PII-Types` | Types of PII detected | `EMAIL,PHONE` |
| `X-Gateway-PII-Blocked` | Whether blocked due to PII | `true` |
| `X-Gateway-Security-Threat` | Jailbreak threat type | `prompt_injection` |
| `X-Gateway-Security-Blocked` | Whether blocked due to jailbreak | `true` |
| `X-Gateway-Semantic-Router-Latency-Ms` | Total processing time | `5` |

## 🚀 Running the POC

### Prerequisites

1. Clone the GIE repo with PR #1981:
   ```bash
   git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git
   cd gateway-api-inference-extension
   git fetch origin pull/1981/head:bbr-pluggable-framework
   git checkout bbr-pluggable-framework
   ```

2. The `go.mod` in this POC uses a `replace` directive to point to the local GIE repo.

### Run Tests

```bash
cd poc-bbr-plugin
go mod tidy
go test -v ./tests/...
```

### Run Demo

```bash
go run ./cmd/demo/
```

### Run HTTP Server

```bash
go run ./cmd/demo/ --serve
# Then test with:
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Write Python code"}]}'
```

## 📊 Test Results

```
=== RUN   TestPluginImplementsBBRPlugin
    ✓ SemanticRouterPlugin implements BBRPlugin interface
=== RUN   TestBBRPluginChainIntegration
    ✓ Plugin chain created with 2 plugins: [MetaDataExtractor Guardrail]
    ✓ Model header: gpt-4
    ✓ Intent category: coding
=== PASS
```

## 🔄 Integration with BBR

```go
// Create registry
registry := framework.NewPluginRegistry()

// Register factory
registry.RegisterFactory("Guardrail", plugin.NewSemanticRouterPlugin)

// Create and register instance
srPlugin, _ := registry.CreatePlugin("Guardrail")
registry.RegisterPlugin(srPlugin)

// Add to chain
chain := framework.NewPluginsChain()
chain.AddPlugin("MetaDataExtractor", registry)  // Default BBR plugin
chain.AddPlugin("Guardrail", registry)          // vSR plugin

// Execute chain
headers, body, err := chain.Run(requestBody, registry)
```

## 🔮 Next Steps

1. **Replace mock classifiers with real vSR implementations** (Candle/LoRA)
2. **Add CGO bindings** for Rust-based ML inference
3. **Contribute to GIE** as reference Guardrail implementation
4. **Add configuration via K8s ConfigMaps** (Phase 6 of BBR roadmap)

## 📚 References

- [PR #1964: BBR Pluggable Framework Proposal](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1964)
- [PR #1981: BBR Pluggable Framework Implementation](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/1981)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)

