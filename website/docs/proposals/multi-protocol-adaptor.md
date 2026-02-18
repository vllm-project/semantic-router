# Design Doc: Multi-Protocol Adapter Architecture

**Author:** vLLM Semantic Router Team  
**Status:** To be Implemented  
**Created:** February 2026  
**Last Updated:** February 2026

## Overview

This document describes the design and implementation of the multi-protocol adapter architecture for vLLM Semantic Router, which abstracts the API layer to support multiple front-end protocols beyond Envoy ExtProc.

## Background

The Semantic Router was tightly coupled to Envoy's External Processor (ExtProc) protocol via gRPC. While this provides powerful integration with Envoy, it created barriers for users who:

- Want to use the router without deploying Envoy
- Prefer direct HTTP/REST API integration
- Use Nginx or other reverse proxies
- Need simpler deployment architectures for development or testing

### Motivation

- **Flexibility:** Users need direct HTTP API access without requiring Envoy infrastructure
- **Testing:** Developers need lightweight testing without full Envoy deployment
- **Extensibility:** Support for nginx, native gRPC, and custom protocols
- **Reusability:** Single routing engine shared across all protocols
- **Deployment Options:** Enable serverless, edge, and simplified deployment scenarios

## Goals

### Primary Goals

1. **Protocol Abstraction:** Separate routing logic from protocol-specific code
2. **Multi-Protocol Support:** Enable simultaneous operation of multiple protocols
3. **Backward Compatibility:** Preserve existing ExtProc functionality
4. **Shared State:** Single source of truth for cache, replay, and routing decisions
5. **Easy Extension:** Simple pattern for adding new protocol adapters

### Non-Goals

1. Replace or deprecate Envoy ExtProc support
2. Change routing decision algorithms or classification logic
3. Modify configuration format beyond adapter section
4. Support protocol-specific features that break abstraction

## Design Principles

### 1. Single Routing Pipeline

**CRITICAL:** All routing logic MUST flow through `RouterEngine.Route()`. No exceptions.

- ✅ Adapters translate protocol → `RouteRequest` → call `RouterEngine.Route()`
- ✅ `RouterEngine.Route()` returns `RouteResponse` → adapters translate → protocol
- ❌ Adapters MUST NOT duplicate classification, security, cache, replay logic
- ❌ Adapters MUST NOT directly call classifiers, cache, or replay recorders

### 2. Thin Adapter Layer

Adapters are **protocol translation only**:

- Parse protocol-specific request format
- Convert to `RouteRequest`
- Call `RouterEngine.Route()`
- Convert `RouteResponse` to protocol format
- Return to client

### 3. RouterEngine Owns All Routing

`RouterEngine.Route()` is the ONLY place where:

- Classification happens
- PII/jailbreak detection runs
- Cache is checked/updated
- Tools are selected
- Replay is recorded
- Backend selection occurs
- Proxying happens (or proxy info is returned)

## Design

### Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    Application Layer                       │
│                                                            │
│  ┌───────────────────────────────────────────────────┐     │
│  │                Adapter Manager                    │     │
│  │  - Reads adapter config                           │     │
│  │  - Creates protocol adapters                      │     │
│  │  - Manages lifecycle                              │     │
│  └──────┬────────┬────────┬───────────┬──────────────┘     │
│         │        │        │           │                    │
│  ┌──────▼──┐ ┌───▼─── ┐ ┌─▼──────┐ ┌──▼─────┐              │
│  │ ExtProc │ │ HTTP   │ │ gRPC   │ │ Nginx  │              │
│  │ Adapter │ │Adapter │ │Adapter │ │Adapter │              │
│  │ ┌─────┐ │ │ ┌─────┐│ │ ┌─────┐│ │ ┌─────┐│              │
│  │ │Parse│ │ │ │Parse││ │ │Parse││ │ │Parse││              │
│  │ │ExtP │ │ │ │HTTP ││ │ │gRPC ││ │ │NJS  ││              │
│  │ └──┬──┘ │ │ └─┬───┘│ │ └─┬───┘│ │ └──┬──┘│              │
│  │    │Conv│ │   │Con │ │   │Con │ │    │Con│              │
│  │    ▼    │ │   ▼    │ │   ▼    │ │    ▼   │              │
│  │ ┌─────┐ │ │ ┌────┐ │ │ ┌────┐ │ │ ┌─────┐│              │
│  │ │Req  │ │ │ │Req │ │ │ │Req │ │ │ │Req  ││              │
│  │ └──┬──┘ │ │ └─┬──┘ │ │ └─┬──┘ │ │ └──┬──┘│              │
│  └────┼────┘ └───┼────┘ └───┼────┘ └────┼───┘              │
│       │          │          │          │                   │
│       └──────────┴──────────┴──────────┘                   │
│                    Single Entry Point                      │
│                             │                              │
│                             ▼                              │
│        ┌──────────────────────────────────────────┐        │
│        │           RouterEngine.Route()           │        │
│        │  1. Classify request                     │        │
│        │  2. Check PII / jailbreak                │        │
│        │  3. Check cache                          │        │
│        │  4. Select tools                         │        │
│        │  5. Select model/backend                 │        │
│        │  6. Record replay                        │        │
│        │  7. Proxy to backend (via Backend Layer) │        │
│        │  8. Update cache                         │        │
│        └──────────────┬───────────────────────────┘        │
│                       │                                    │
│                       ▼                                    │
│                  RouteResponse                             │
│                       │                                    │
│        ┌──────────────┼──────────────┬───────────┐         │
│        │              │              │           │         │
│  ┌─────▼─────┐ ┌──────▼────┐ ┌───────▼───┐ ┌─────▼─────┐   │
│  │ ExtProc   │ │ HTTP      │ │ gRPC      │ │ Nginx     │   │
│  │ Adapter   │ │ Adapter   │ │ Adapter   │ │ Adapter   │   │
│  │ ┌───────┐ │ │ ┌───────┐ │ │ ┌───────┐ │ │ ┌───────┐ │   │
│  │ │Convert│ │ │ │Convert│ │ │ │Convert│ │ │ │Convert│ │   │
│  │ │to gRPC│ │ │ │to HTTP│ │ │ │gRPC   │ │ │ │to NJS │ │   │
│  │ └───────┘ │ │ └───────┘ │ │ └───────┘ │ │ └───────┘ │   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘   │
│        │             │             │             │         │
└────────┼─────────────┼─────────────┼─────────────┼─────────┘
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                            │
                            ▼
         ┌─────────────────────────────────────────┐
         │      Backend Abstraction Layer          │
         └──────┬──────────────────┬───────────────┘
                │                  │
       ┌────────▼────────┐  ┌──────▼──────────┐
       │ Envoy Proxy     │  │ Direct Proxy    │
       │ (ExtProc mode)  │  │ (HTTP/gRPC)     │
       │ - Dynamic fwd   │  │ - HTTP client   │
       │ - Headers only  │  │ - Full response │
       └────────┬────────┘  └──────┬──────────┘
                │                  │
                └──────────┬───────┘
                           ▼
             ┌────────────────────────────┐
             │      Inference Backends    │
             │  ┌────────┐  ┌────────┐    │
             │  │ vLLM   │  │Ollama  │    │
             │  │Server  │  │Server  │    │
             │  └────────┘  └────────┘    │
             └────────────────────────────┘
```

**Key Insight:** Adapters are **thin translation layers**. All intelligence lives in RouterEngine.

### Component Design

#### 1. RouterEngine (Core)

**Location:** `pkg/router/engine/`

**Responsibilities:**

- Protocol-agnostic routing logic
- Request classification and decision evaluation
- Semantic cache operations
- Tool selection and embedding
- Router replay recording
- PII and jailbreak detection
- Model selection

**Key Methods:**

```go
type RouterEngine struct {
    Config               *config.RouterConfig
    Classifier           *classification.Classifier
    PIIChecker           *pii.PolicyChecker
    Cache                cache.CacheBackend
    ToolsDatabase        *tools.ToolsDatabase
    ModelSelector        *selection.Registry
    ReplayRecorders      map[string]*routerreplay.Recorder
}

func (e *RouterEngine) Route(ctx context.Context, req *RouteRequest) (*RouteResponse, error)
func (e *RouterEngine) ClassifyRequest(ctx context.Context, messages []Message) (*ClassificationResult, error)
func (e *RouterEngine) CheckCache(ctx context.Context, model, query, decisionName string) (string, bool, error)
func (e *RouterEngine) UpdateCache(ctx context.Context, model, query, response, decisionName string) error
func (e *RouterEngine) SelectTools(ctx context.Context, query string, topK int) ([]openai.ChatCompletionToolParam, error)
func (e *RouterEngine) RecordReplay(ctx context.Context, decisionName string, record *routerreplay.RoutingRecord) error
```

**Design Decisions:**

- Single instance shared across all adapters
- Stateful (maintains cache, replay recorders)
- No protocol-specific logic
- Returns protocol-agnostic data structures

#### 2. Adapter Interface

**Location:** `pkg/adapter/manager.go`

```go
type Adapter interface {
    Start() error                      // Start the adapter (blocks)
    Stop() error                       // Graceful shutdown
    GetEngine() *engine.RouterEngine  // Access to shared engine
}
```

**Design Decisions:**

- Minimal interface for maximum flexibility
- Each adapter owns its lifecycle
- No protocol-specific methods in interface
- Adapters run in separate goroutines

#### 3. Adapter Manager

**Location:** `pkg/adapter/manager.go`

**Responsibilities:**

- Parse adapter configuration
- Instantiate adapters based on config
- Start adapters in separate goroutines
- Coordinate graceful shutdown

**Key Methods:**

```go
func (m *Manager) CreateAdapters(cfg *config.RouterConfig, eng *engine.RouterEngine, configPath string) error
func (m *Manager) StartAll() error
func (m *Manager) StopAll() error
func (m *Manager) Wait()
```

#### 4. ExtProc Adapter

**Location:** `pkg/adapter/extproc/`

**Responsibilities:**

- Wrap existing Envoy ExtProc implementation
- Maintain backward compatibility
- Handle gRPC/Envoy protocol specifics
- Support TLS configuration

**Key Features:**

- Uses existing `extproc.OpenAIRouter` internally
- Translates Envoy requests to RouterEngine calls
- Preserves all existing ExtProc functionality
- Configurable TLS support

#### 5. HTTP Adapter

**Location:** `pkg/adapter/http/`

**Responsibilities:**

- Provide OpenAI-compatible REST API
- Direct access without Envoy
- Handle HTTP-specific concerns (CORS, headers, etc.)

**Endpoints:**

- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions (future)
- `GET /v1/models` - List available models
- `POST /v1/classify` - Classification endpoint
- `POST /v1/route` - Routing decision endpoint
- `GET /v1/router_replay` - List replay records
- `GET /v1/router_replay/{id}` - Get replay record
- `GET /health` - Health check
- `GET /ready` - Readiness check

#### 6. gRPC Adapter

**Location:** `pkg/adapter/grpc/`

**Responsibilities:**

- Provide native gRPC API for routing
- More efficient than ExtProc for direct gRPC clients
- Custom service definition optimized for routing
- Support for streaming and unary RPCs

**Key Features:**

- Custom `.proto` service definition
- Optimized for low-latency routing decisions
- Supports both synchronous and asynchronous routing
- Built-in load balancing and connection pooling
- Compatible with gRPC ecosystem (grpc-gateway, etc.)

**Service Definition:**

```protobuf
service SemanticRouter {
  rpc Route(RouteRequest) returns (RouteResponse);
  rpc Classify(ClassifyRequest) returns (ClassifyResponse);
  rpc StreamRoute(stream RouteRequest) returns (stream RouteResponse);
}
```

#### 7. Nginx Adapter

**Location:** `pkg/adapter/nginx/`

**Responsibilities:**

- Integration with Nginx via NJS (JavaScript) module
- Lua script support for OpenResty
- Header-based routing similar to ExtProc
- Direct Nginx upstream configuration

**Key Features:**

- NJS module for request/response processing
- Communicates with RouterEngine via HTTP or gRPC
- Sets upstream selection based on routing decision
- Minimal overhead compared to ExtProc
- Native Nginx performance characteristics

**Integration Methods:**

1. **NJS Module:** JavaScript-based request processing
2. **Lua/OpenResty:** For OpenResty deployments
3. **HTTP Subrequest:** Calls HTTP adapter internally
4. **Shared Memory:** Direct IPC with RouterEngine process

### Configuration Design

#### Adapter Configuration

```yaml
adapters:
  - type: "envoy" # ExtProc adapter
    enabled: true
    port: 50051
    tls:
      enabled: true
      cert_file: "/path/to/cert.pem"
      key_file: "/path/to/key.pem"

  - type: "http" # HTTP REST API
    enabled: true
    port: 9000

  - type: "grpc" # Native gRPC API
    enabled: true
    port: 50052
    tls:
      enabled: true
      cert_file: "/path/to/cert.pem"
      key_file: "/path/to/key.pem"

  - type: "nginx" # Nginx integration
    enabled: true
    port: 9001
    mode: "njs" # Options: njs, lua, http
    config:
      upstream_variable: "backend_upstream"
      header_prefix: "x-vsr-"
```

**Design Decisions:**

- Array allows multiple adapters of same type (future: multiple HTTP on different ports)
- Per-adapter TLS configuration
- Simple enable/disable without removing config
- Port configuration at adapter level

### Data Flow

#### Request Flow (HTTP Adapter Example)

```
1. Client Request
   ↓
2. HTTP Adapter receives POST /v1/chat/completions
   ↓
3. Parse OpenAI request format
   ↓
4. Call RouterEngine.Route(RouteRequest)
   ↓
5. RouterEngine performs:
   - Classification (which decision matches?)
   - Cache check (semantic similarity)
   - Tool selection (if enabled)
   - Replay recording (if configured)
   ↓
6. RouterEngine returns RouteResponse
   ↓
7. HTTP Adapter proxies to selected backend
   ↓
8. Return response to client
```

#### Shared State Flow

```
HTTP Request A → HTTP Adapter
                     ↓
                RouterEngine → Cache (hit/miss)
                     ↑
ExtProc Request B → ExtProc Adapter
```

Both adapters share:

- Same cache entries
- Same replay recorders
- Same classification decisions
- Same model selection state

## Implementation Details

### Initialization Sequence

```go
// main.go
1. Load configuration
2. Initialize embedding models
3. Create RouterEngine (NewRouterEngine)
   - Initialize classifier
   - Create semantic cache
   - Setup tools database
   - Initialize replay recorders per decision
   - Setup model selector
4. Create Adapter Manager
5. Manager creates adapters (CreateAdapters)
   - Each adapter gets reference to RouterEngine
   - Per-adapter configuration (port, TLS)
6. Manager starts all adapters (StartAll)
   - Each adapter in separate goroutine
7. Main blocks on Manager.Wait()
```

### Error Handling

- **Adapter Creation Failure:** Fatal error, application exits
- **Adapter Start Failure:** Fatal error, application exits
- **Runtime Errors:** Logged, adapter continues if possible
- **RouterEngine Errors:** Returned to adapter for protocol-specific handling

### Concurrency Model

- **RouterEngine:** Thread-safe, multiple adapters can call concurrently
- **Cache:** Backend handles concurrency (Redis, Milvus, etc.)
- **Replay Recorders:** Thread-safe map with per-decision locks
- **Adapters:** Independent goroutines, no shared adapter state

## Trade-offs and Alternatives

### Design Decisions

#### 1. Single Shared RouterEngine vs. Per-Adapter Engines

**Chosen:** Single shared RouterEngine

**Rationale:**

- Consistent routing decisions across protocols
- Shared cache improves hit rate
- Single source of truth for replay records
- Reduced memory footprint

**Trade-off:** Potential contention point (mitigated by thread-safe design)

#### 2. Adapter Interface Design

**Alternatives Considered:**

A. **Rich Interface:**

```go
type Adapter interface {
    Start() error
    Stop() error
    HandleRequest(req *Request) (*Response, error)
    GetMetrics() *Metrics
    Configure(cfg *Config) error
}
```

B. **Minimal Interface (Chosen):**

```go
type Adapter interface {
    Start() error
    Stop() error
    GetEngine() *engine.RouterEngine
}
```

**Rationale:** Minimal interface allows maximum protocol flexibility. Different protocols have vastly different request/response models.

#### 3. Configuration Approach

**Alternatives:**

A. Separate files per adapter
B. Environment variables
C. Single config with adapters section (Chosen)

**Rationale:** Single config file keeps all configuration in one place, easier to manage and version control.

#### 4. Backward Compatibility

**Approach:** Wrap existing ExtProc implementation rather than rewrite

**Rationale:**

- No breaking changes
- Gradual migration path
- Proven, tested code remains in use
- Reduced risk

### Known Limitations

1. **No Protocol-Specific Optimization:** Abstraction prevents protocol-specific optimizations
2. **Adapter Isolation:** Adapters can't directly communicate (by design)
3. **Shared State Challenges:** Race conditions if RouterEngine not thread-safe
4. **Configuration Complexity:** More options for users to configure

## Testing Strategy

### Unit Tests

- RouterEngine methods with mock adapters
- Individual adapter logic
- Configuration parsing

### Integration Tests

- Multiple adapters running simultaneously
- Shared state consistency (cache hits across adapters)
- Replay recording from both protocols

### E2E Tests

- ExtProc via Envoy on port 8801
- HTTP direct on port 9000
- Verify identical routing decisions
- Verify replay records visible from both

## Future Work

### Short Term

1. **Graceful Shutdown**
   - Drain in-flight requests
   - Close connections cleanly
   - Flush replay records

2. **Adapter Metrics**
   - Per-adapter request counters
   - Latency histograms
   - Error rates

3. **Enhanced Nginx Integration**
   - OpenResty Lua module
   - Nginx Plus dynamic upstream API
   - Shared memory IPC for zero-copy

### Long Term

1. **gRPC Streaming Enhancements**
   - Bi-directional streaming support
   - Server-side streaming for batch requests
   - Client-side streaming for large inputs

2. **WebSocket Adapter**
   - Real-time streaming
   - Bi-directional communication

3. **Plugin System**
   - Dynamic adapter loading
   - Third-party adapters

4. **Per-Adapter Configuration**
   - Rate limiting
   - Authentication
   - Custom middleware

## Migration Guide

### From Old ExtProc-Only to Adapter Architecture

**Before:**

```go
server := extproc.NewServer(configPath, port, secure, certPath)
server.Start()
```

**After:**

```go
engine := engine.NewRouterEngine(configPath)
manager := adapter.NewManager()
manager.CreateAdapters(cfg, engine, configPath)
manager.StartAll()
manager.Wait()
```

**Configuration:**

```yaml
# Add this to config.yaml
adapters:
  - type: "envoy"
    enabled: true
    port: 50051
```

### Adding New Adapter

1. Create package `pkg/adapter/myprotocol/`
2. Implement `Adapter` interface:

```go
type MyAdapter struct {
    engine *engine.RouterEngine
    port   int
}

func NewAdapter(eng *engine.RouterEngine, port int) (*MyAdapter, error) {
    return &MyAdapter{engine: eng, port: port}, nil
}

func (a *MyAdapter) Start() error {
    // Protocol-specific server setup
    // Call a.engine.Route() for routing logic
}

func (a *MyAdapter) Stop() error {
    // Graceful shutdown
}

func (a *MyAdapter) GetEngine() *engine.RouterEngine {
    return a.engine
}
```

3. Register in manager:

```go
// pkg/adapter/manager.go
case "myprotocol":
    adapter, err = myprotocol.NewAdapter(eng, adapterCfg.Port)
```

4. Add configuration support:

```yaml
adapters:
  - type: "myprotocol"
    enabled: true
    port: 9001
```

## References

- [OpenAI API Specification](https://platform.openai.com/docs/api-reference)
- [Envoy ExtProc Documentation](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter)
- [vLLM Semantic Router Documentation](https://vllm-semantic-router.com)

## Appendix

### Performance Considerations

- **RouterEngine:** Single instance reduces memory, but could be bottleneck
- **Cache:** Backend choice critical (Redis/Milvus for production)
- **Replay Recording:** Async writes recommended for high throughput
- **Adapter Overhead:** Minimal, mostly network/protocol serialization

### Security Considerations

- **TLS Support:** Per-adapter TLS configuration
- **Authentication:** Handled at adapter level (future work: external authz abstraction)
- **Authorization:** Future work to abstract external authz providers (OPA, custom)
- **PII Detection:** Shared across all adapters
- **Jailbreak Detection:** Shared across all adapters

### Monitoring and Observability

- **Metrics:** Per-adapter and RouterEngine metrics
- **Tracing:** Distributed tracing spans adapters
- **Logging:** Structured logs with adapter context
- **Health Checks:** Per-adapter health endpoints
