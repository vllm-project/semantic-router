# E2E Test Framework

A comprehensive end-to-end testing framework for Semantic Router with support for multiple deployment profiles.

## Architecture

The framework follows a **separation of concerns** design:

- **Profiles**: Define deployment environments and configurations
- **Test Cases**: Reusable test logic that can be shared across profiles
- **Framework**: Core infrastructure for test execution and reporting

### Supported Profiles

- **ai-gateway**: Tests Semantic Router with Envoy AI Gateway integration
- **aibrix**: Tests Semantic Router with vLLM AIBrix integration
- **dynamic-config**: Tests Semantic Router with Kubernetes CRD-based configuration (IntelligentRoute/IntelligentPool)
- **istio**: Tests Semantic Router with Istio service mesh integration
- **production-stack**: Tests vLLM Production Stack configurations
- **llm-d**: Tests Semantic Router with LLM-D distributed inference
- **response-api**: Tests Responses API endpoints (POST/GET/DELETE /v1/responses)
- **response-api-redis**: Tests Responses API endpoints with Redis storage backend
- **response-api-redis-cluster**: Tests Responses API endpoints with Redis Cluster backend
- **dynamo**: Tests with Nvidia Dynamo (future)

## Directory Structure

```
e2e/
├── cmd/
│   └── e2e/              # Main test runner
├── pkg/
│   ├── framework/        # Core test framework
│   ├── cluster/          # Kind cluster management
│   ├── docker/           # Docker image operations
│   ├── helm/             # Helm deployment utilities
│   ├── helpers/          # Kubernetes helper functions
│   └── testcases/        # Test case registry
├── testcases/            # Reusable test cases (shared across profiles)
│   ├── testdata/         # Test data files
│   ├── common.go         # Common helper functions
│   ├── chat_completions_request.go
│   ├── domain_classify.go
│   ├── cache.go
│   ├── pii_detection.go
│   ├── jailbreak_detection.go
│   ├── decision_priority.go           # Signal-decision: Priority selection
│   ├── plugin_chain_execution.go      # Signal-decision: Plugin chains
│   ├── rule_condition_logic.go        # Signal-decision: AND/OR operators
│   ├── decision_fallback.go           # Signal-decision: Fallback behavior
│   ├── keyword_routing.go             # Signal-decision: Keyword matching
│   ├── plugin_config_variations.go    # Signal-decision: Plugin configs
│   └── embedding_signal_routing.go    # Signal-decision: Embedding signals
├── profiles/
│   ├── ai-gateway/       # AI Gateway test profile
│   │   └── profile.go    # Profile definition and environment setup
│   ├── aibrix/           # AIBrix test profile
│   │   └── profile.go
│   └── dynamic-config/   # Dynamic CRD-based configuration profile
│       ├── profile.go
│       └── crds/         # IntelligentRoute and IntelligentPool CRDs
│           ├── intelligentroute.yaml
│           └── intelligentpool.yaml
└── README.md
```

## Available Test Cases

The framework includes the following test cases (all in `e2e/testcases/`):

### Basic Functionality Tests

| Test Case | Description | Metrics |
|-----------|-------------|---------|
| `chat-completions-request` | Basic chat completions API test | Response validation |
| `chat-completions-stress-request` | Sequential stress test with 1000 requests | Success rate, avg duration |
| `chat-completions-progressive-stress` | Progressive QPS stress test (10/20/50/100 QPS) | Per-stage success rate, latency stats |

### Classification and Feature Tests

| Test Case | Description | Metrics |
|-----------|-------------|---------|
| `domain-classify` | Domain classification accuracy | 65 cases, accuracy rate |
| `semantic-cache` | Semantic cache hit rate | 5 groups, cache hit rate |
| `pii-detection` | PII detection and blocking | 10 PII types, detection rate, block rate |
| `jailbreak-detection` | Jailbreak attack detection | 10 attack types, detection rate, block rate |

### Response API Tests

| Test Case | Description | Metrics |
|-----------|-------------|---------|
| `response-api-create` | POST /v1/responses - Create a new response | Response ID validation, status check |
| `response-api-get` | GET /v1/responses/{id} - Retrieve a response | Response retrieval, ID matching |
| `response-api-delete` | DELETE /v1/responses/{id} - Delete a response | Deletion confirmation, 404 verification |
| `response-api-input-items` | GET /v1/responses/{id}/input_items - List input items | Input items list, pagination |
| `response-api-conversation-chaining` | Conversation chaining with previous_response_id (3-turn chain) | History preservation, instruction inheritance |
| `response-api-error-missing-input` | Error handling - Invalid request format (missing input field) | 400 error, error message validation |
| `response-api-error-nonexistent-previous-response-id` | Error handling - Non-existent previous_response_id | Graceful degradation or 404 error |
| `response-api-error-nonexistent-response-id-get` | Error handling - Non-existent response ID for GET | 404 error response |
| `response-api-error-nonexistent-response-id-delete` | Error handling - Non-existent response ID for DELETE | 404 error response |
| `response-api-error-backend-passthrough` | Error handling - Backend error passthrough | Error format validation, passthrough behavior |

### Signal-Decision Engine Tests

| Test Case | Description | Metrics |
|-----------|-------------|---------|
| `decision-priority-selection` | Decision priority selection with multiple matches | 4 cases, priority validation (indirect) |
| `plugin-chain-execution` | Plugin execution order (PII → Cache → System Prompt) | 4 cases, chain validation, blocking behavior |
| `rule-condition-logic` | AND/OR operators and keyword matching | 6 cases, operator validation |
| `decision-fallback-behavior` | Fallback to default decision when no match | 5 cases, fallback validation |
| `keyword-routing` | Keyword-based routing decisions | 6 cases, keyword matching (case-insensitive) |
| `plugin-config-variations` | Plugin configuration variations (PII allowlist, cache thresholds) | 6 cases, config validation |
| `embedding-signal-routing` | EmbeddingSignal CRD routing with semantic similarity | 31 cases, PII/security/technical/domain routing accuracy |

**Signal-Decision Engine Features Tested:**

- ✅ Decision priority selection (priority 15 > 10) - validated by checking which decision wins when multiple match
- ✅ Plugin chain execution order and blocking
- ✅ Rule condition logic (AND/OR operators)
- ✅ Keyword-based routing (case-insensitive)
- ✅ Decision fallback behavior
- ✅ Per-decision plugin configurations
- ✅ PII allowlist handling
- ✅ Per-decision cache thresholds (0.75, 0.92, 0.95)
- ✅ Embedding signal routing (semantic similarity-based routing via IntelligentRoute CRD)

All test cases:

- Use model name `"MoM"` to trigger decision engine
- Automatically clean up port forwarding
- Generate detailed reports with statistics
- Support verbose logging

## Quick Start

### Install dependencies (optional)

```bash
make e2e-deps
```

### Run all tests with default profile (ai-gateway)

```bash
make e2e-test
```

### Run specific profile

```bash
make e2e-test E2E_PROFILE=ai-gateway
make e2e-test E2E_PROFILE=production-stack
```

### Run specific test cases

```bash
# Run only specific test cases using make
make e2e-test-specific E2E_TESTS="chat-completions-progressive-stress"

# Run multiple specific test cases using make
make e2e-test-specific E2E_TESTS="chat-completions-request,chat-completions-progressive-stress"

# Or run directly with the binary
./bin/e2e -profile ai-gateway -tests chat-completions-progressive-stress -verbose

# Run multiple specific test cases with the binary
./bin/e2e -profile ai-gateway -tests "chat-completions-request,chat-completions-progressive-stress"
```

### Run with custom options

```bash
# Keep cluster after test
make e2e-test E2E_KEEP_CLUSTER=true

# Use existing cluster
make e2e-test E2E_USE_EXISTING_CLUSTER=true

# Disable verbose output
make e2e-test E2E_VERBOSE=false

# Run tests in parallel
make e2e-test E2E_PARALLEL=true

# Combine multiple options
make e2e-test E2E_PROFILE=ai-gateway E2E_KEEP_CLUSTER=true E2E_VERBOSE=true
```

### Debug mode

```bash
# Run tests with debug mode (keeps cluster and enables verbose logging)
make e2e-test-debug
```

### Advanced Workflows

#### Setup environment once, run tests multiple times

This is useful for development and debugging:

```bash
# Step 1: Setup environment only (no tests)
make e2e-setup

# Step 2: Run all tests (skip setup)
make e2e-test-only

# Step 3: Run specific tests (skip setup)
make e2e-test-only E2E_TESTS="chat-completions-request"

# Step 4: Make code changes and re-run tests
make e2e-test-only E2E_TESTS="domain-classify"

# Step 5: Clean up when done
make e2e-cleanup
```

#### Using binary directly

```bash
# Setup only
./bin/e2e -profile ai-gateway -setup-only -keep-cluster -verbose

# Run tests only (assumes environment is already deployed)
./bin/e2e -profile ai-gateway -skip-setup -use-existing-cluster -verbose

# Run specific tests only
./bin/e2e -profile ai-gateway -skip-setup -use-existing-cluster -tests "chat-completions-request"
```

### Test Reports

After running tests, reports are generated:

- `test-report.json`: Structured test results
- `test-report.md`: Human-readable Markdown report
- `semantic-router-logs.txt`: Complete semantic-router pod logs

Each test case also prints detailed statistics to the console.

## Environment Variables

The following environment variables can be used to customize test execution:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `E2E_PROFILE` | Test profile to run | `ai-gateway` | `make e2e-test E2E_PROFILE=ai-gateway` |
| `E2E_CLUSTER_NAME` | Kind cluster name | `semantic-router-e2e` | `make e2e-test E2E_CLUSTER_NAME=my-cluster` |
| `E2E_IMAGE_TAG` | Docker image tag | `e2e-test` | `make e2e-test E2E_IMAGE_TAG=v1.0.0` |
| `E2E_KEEP_CLUSTER` | Keep cluster after tests | `false` | `make e2e-test E2E_KEEP_CLUSTER=true` |
| `E2E_USE_EXISTING_CLUSTER` | Use existing cluster | `false` | `make e2e-test E2E_USE_EXISTING_CLUSTER=true` |
| `E2E_VERBOSE` | Enable verbose logging | `true` | `make e2e-test E2E_VERBOSE=false` |
| `E2E_PARALLEL` | Run tests in parallel | `false` | `make e2e-test E2E_PARALLEL=true` |
| `E2E_TESTS` | Specific test cases to run | (all tests) | `make e2e-test-specific E2E_TESTS="test1,test2"` |
| `E2E_SETUP_ONLY` | Only setup profile without running tests (automatically keeps cluster) | `false` | `make e2e-test E2E_SETUP_ONLY=true` |
| `E2E_SKIP_SETUP` | Skip setup and only run tests | `false` | `make e2e-test E2E_SKIP_SETUP=true` |

**Note**:

- When `E2E_SETUP_ONLY=true` is set, the cluster is automatically kept (no need to set `E2E_KEEP_CLUSTER=true`)
- When using the binary directly (`./bin/e2e`), use command-line flags instead:

- `-profile` instead of `E2E_PROFILE`
- `-cluster` instead of `E2E_CLUSTER_NAME`
- `-image-tag` instead of `E2E_IMAGE_TAG`
- `-keep-cluster` instead of `E2E_KEEP_CLUSTER`
- `-use-existing-cluster` instead of `E2E_USE_EXISTING_CLUSTER`
- `-verbose` instead of `E2E_VERBOSE`
- `-parallel` instead of `E2E_PARALLEL`
- `-tests` instead of `E2E_TESTS`
- `-setup-only` instead of `E2E_SETUP_ONLY`
- `-skip-setup` instead of `E2E_SKIP_SETUP`

Example:

```bash
./bin/e2e -profile ai-gateway -keep-cluster -verbose -tests "chat-completions-request"
```

## Adding New Test Profiles

1. Create a new directory under `profiles/`
2. Implement the `Profile` interface
3. Register test cases using the test case registry
4. Add profile-specific deployment configurations

See `profiles/ai-gateway/` for a complete example.

## Key Concepts

### Profile vs Test Case Separation

**Profiles** are responsible for:

- Deploying the test environment (Helm charts, Kubernetes resources)
- Verifying environment health
- Providing service configuration (namespace, labels, port mappings)

**Test Cases** are responsible for:

- Executing test logic
- Validating functionality
- Reporting results

This separation allows test cases to be **reused across different profiles** by simply providing different service configurations.

**Benefits:**

- ✅ Test cases are independent of deployment details
- ✅ Easy to add new profiles without duplicating test logic
- ✅ Profiles can share common test cases
- ✅ Test cases can be maintained in one place
- ✅ Clear separation of concerns

### Service Configuration

Profiles provide service configuration to test cases via `ServiceConfig`:

```go
type ServiceConfig struct {
    LabelSelector string  // e.g., "gateway.envoyproxy.io/owning-gateway-namespace=default,..."
    Namespace     string  // Service namespace
    Name          string  // Service name (optional, if empty uses LabelSelector)
    PortMapping   string  // e.g., "8080:80" (localPort:servicePort)
}
```

Test cases use this configuration to connect to the deployed service without knowing the specific deployment details.

## Test Case Registration

Test cases are registered using a simple function-based approach:

```go
func init() {
    pkgtestcases.Register("my-test", pkgtestcases.TestCase{
        Description: "Description of what this test does",
        Tags:        []string{"functional", "llm"},
        Fn:          testMyFeature,
    })
}

func testMyFeature(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
    // Setup connection to service
    localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
    if err != nil {
        return err
    }
    defer stopPortForward() // Always clean up port forwarding

    // Test implementation using localPort
    // ...
    return nil
}
```

## Framework Features

- **Automatic cluster lifecycle management**: Creates and cleans up Kind clusters
- **Docker image building and loading**: Builds images and loads them into Kind
- **Helm deployment automation**: Deploys required Helm charts
- **Automatic port forwarding cleanup**: Each test case cleans up its port forwarding
- **Detailed logging**: Provides comprehensive test output
- **Test reporting**: Generates JSON and Markdown reports
- **Resource cleanup**: Ensures proper cleanup even on failures

## Test Data

Test data is stored in `e2e/testcases/testdata/` as JSON files. Each test case loads its own test data.

**Available Test Data:**

- `domain_classify_cases.json`: 65 test cases across 13 categories
- `cache_cases.json`: 5 groups of similar questions for semantic cache testing
- `pii_detection_cases.json`: 10 PII types (email, phone, SSN, etc.)
- `jailbreak_detection_cases.json`: 10 attack types (prompt injection, DAN, etc.)
- `embedding_signal_cases.json`: 31 test cases for EmbeddingSignal routing (PII, security, technical, domain classification)

**Signal-Decision Engine Tests** use embedded test cases (defined inline in test files) to validate:

- Decision priority mechanisms (4 test cases)
- Plugin chain execution and blocking (4 test cases)
- Rule condition logic with AND/OR operators (6 test cases)
- Decision fallback behavior (5 test cases)
- Keyword-based routing (6 test cases)
- Plugin configuration variations (6 test cases)

### Embedding Signal Routing

The `embedding-signal-routing` test validates the `IntelligentRoute` CRD with `EmbeddingSignal` configurations. This test:

**Features Tested:**

- Semantic similarity-based routing using embedding models (Qwen3/Gemma)
- PII detection via embedding signals (semantic patterns like "share my credit card")
- Security threat detection (SQL injection, unauthorized access attempts)
- Technical domain routing (Kubernetes, container orchestration)
- Domain classification (healthcare, finance, general knowledge)
- Threshold behavior (0.75 similarity threshold)
- Aggregation methods (max similarity across multiple candidates)
- Paraphrase handling (different wording, same intent)
- Multi-signal evaluation (multiple signals in one request)

**Test Categories:**

- PII Detection (7 cases): Semantic PII pattern matching
- Security Threats (4 cases): Malicious intent detection
- Technical Topics (4 cases): Kubernetes-specific routing
- Domain Classification (4 cases): Healthcare, finance domains
- Threshold Tests (3 cases): Similarity boundary testing
- Aggregation Tests (2 cases): Multi-candidate matching
- Paraphrase Tests (2 cases): Intent recognition
- Multi-signal (1 case): Combined signal evaluation
- Edge Cases (4 cases): Empty content, short/long queries

**Profile Support:**

- ✅ `dynamic-config` profile (uses CRDs)
- ❌ `ai-gateway` profile (uses static YAML config)
- ❌ `aibrix` profile (uses static YAML config)

**Requirements:**

- Embedding models must be initialized (Qwen3 or Gemma)
- `EMBEDDING_MODEL_OVERRIDE=qwen3` environment variable for consistent test results
- IntelligentRoute CRD with EmbeddingSignal definitions
- Model requests must use `"model": "auto"` to trigger decision evaluation

**Note:** This test differs from `pii-detection` (which uses regex/NER plugins) and `domain-classify` (which uses academic domain routing). Embedding signals use semantic similarity to detect **intent** rather than exact patterns.

**Test Data Format Example:**

```json
{
  "cases": [
    {
      "question": "What is 2+2?",
      "expected_category": "math",
      "expected_reasoning": "Basic arithmetic question"
    }
  ]
}
```

## Prerequisites

Before running E2E tests, ensure you have the following tools installed:

- [Go](https://golang.org/doc/install) 1.24 or later
- [Docker](https://docs.docker.com/get-docker/)
- [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/docs/intro/install/)

## Development

### Adding a New Test Case

Test cases are created in the `e2e/testcases/` directory and can be reused across multiple profiles.

**Steps:**

1. Create a new file in `e2e/testcases/` (e.g., `my_feature.go`)
2. Implement the test function with proper cleanup
3. Register it in the `init()` function
4. Add test data to `e2e/testcases/testdata/` if needed
5. Add the test case name to any profile's `GetTestCases()` method

**Example:**

```go
package testcases

import (
    "context"
    "fmt"

    "k8s.io/client-go/kubernetes"
    pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
    pkgtestcases.Register("my-feature", pkgtestcases.TestCase{
        Description: "Test my new feature",
        Tags:        []string{"functional", "llm"},
        Fn:          testMyFeature,
    })
}

func testMyFeature(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
    if opts.Verbose {
        fmt.Println("[Test] Testing my feature")
    }

    // Setup service connection and get local port
    localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
    if err != nil {
        return err
    }
    defer stopPortForward() // IMPORTANT: Always clean up port forwarding

    // Test implementation using localPort
    url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
    // ... send requests and validate responses

    return nil
}
```

**Important Notes:**

- Always use `defer stopPortForward()` to clean up port forwarding
- Use `opts.ServiceConfig` to get service connection details
- Use `opts.Verbose` for debug logging
- Load test data from `e2e/testcases/testdata/`
- Use model name `"MoM"` in all requests

### Adding a New Profile

Profiles define deployment environments and can reuse existing test cases.

**Steps:**

1. Create a new directory under `profiles/` (e.g., `profiles/istio/`)
2. Create `profile.go` implementing the `Profile` interface
3. Implement required methods:
   - `Setup()`: Deploy environment
   - `Teardown()`: Clean up resources
   - `GetTestCases()`: Return list of test case names to run
   - `GetServiceConfig()`: Provide service configuration
4. Import the `testcases` package to register test cases
5. Update `cmd/e2e/main.go` to include the new profile

**Example:**

```go
package myprofile

import (
    "context"

    "github.com/vllm-project/semantic-router/e2e/pkg/framework"

    // Import testcases to register them
    _ "github.com/vllm-project/semantic-router/e2e/testcases"
)

type Profile struct {
    verbose bool
}

func NewProfile(verbose bool) *Profile {
    return &Profile{verbose: verbose}
}

func (p *Profile) Name() string {
    return "my-profile"
}

func (p *Profile) Description() string {
    return "My custom deployment profile"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
    // Deploy your environment
    return nil
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
    // Clean up resources
    return nil
}

func (p *Profile) GetTestCases() []string {
    return []string{
        "chat-completions-request",
        "domain-classify",
        // ... other test cases
    }
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
    return framework.ServiceConfig{
        LabelSelector: "app=my-service",
        Namespace:     "default",
        PortMapping:   "8080:80",
    }
}
```

See `profiles/ai-gateway/` for a complete example.

## Profile Details

### Istio Profile

The Istio profile tests Semantic Router deployment and functionality in an Istio service mesh environment. It validates both Istio-specific features (sidecars, mTLS, tracing) and general Semantic Router functionality through Istio Gateway + VirtualService routing.

**What it Tests:**

- **Istio-Specific Features:**
  - Istio sidecar injection and health
  - Traffic routing through Istio ingress gateway
  - Mutual TLS (mTLS) between services
  - Distributed tracing and observability

- **Semantic Router Features (through Istio):**
  - Chat completions API and stress testing
  - Domain classification and routing
  - Semantic cache, PII detection, jailbreak detection
  - Signal-Decision engine (priority, plugins, keywords, fallback)

**Prerequisites:**

- Docker and Kind (managed by E2E framework)
- Helm (for installing Istio components)

**Components Deployed:**

1. **Istio Control Plane** (`istio-system` namespace):
   - `istiod` - Istio control plane
   - `istio-ingressgateway` - Ingress gateway for external traffic

2. **Semantic Router** (`semantic-router` namespace):
   - Deployed via Helm with Istio sidecar injection enabled
   - Namespace labeled with `istio-injection=enabled`

3. **Istio Resources**:
   - `Gateway` - Configures ingress gateway on port 80
   - `VirtualService` - Routes traffic to Semantic Router service
   - `DestinationRule` - Enables mTLS with `ISTIO_MUTUAL` mode

**Test Cases:**

**Istio-Specific Tests (4):**

| Test Case | Description | What it Validates |
|-----------|-------------|-------------------|
| `istio-sidecar-health-check` | Verify Envoy sidecar injection | - Istio-proxy container exists<br>- Sidecar is healthy and ready<br>- Namespace has `istio-injection=enabled` label |
| `istio-traffic-routing` | Test routing through Istio gateway | - Gateway and VirtualService exist<br>- Requests route correctly to Semantic Router<br>- Istio/Envoy headers present in responses |
| `istio-mtls-verification` | Verify mutual TLS configuration | - DestinationRule has `ISTIO_MUTUAL` mode<br>- mTLS certificates present in istio-proxy<br>- PeerAuthentication policy (if configured) |
| `istio-tracing-observability` | Check distributed tracing and metrics | - Trace headers propagated<br>- Envoy metrics exposed<br>- Telemetry configuration<br>- Access logs enabled |

**Common Functionality Tests (through Istio Gateway):**

These tests validate that Semantic Router features work correctly when routed through Istio Gateway and VirtualService:

- `chat-completions-request` - Basic API functionality
- `chat-completions-stress-request` - Sequential stress (1000 requests)
- `domain-classify` - Classification accuracy (65 cases)
- `semantic-cache` - Cache hit rate (5 groups)
- `pii-detection` - PII detection and blocking (10 types)
- `jailbreak-detection` - Attack detection (10 types)
- `decision-priority-selection` - Priority-based routing (4 cases)
- `plugin-chain-execution` - Plugin ordering (4 cases)
- `rule-condition-logic` - AND/OR operators (6 cases)
- `decision-fallback-behavior` - Fallback handling (5 cases)
- `keyword-routing` - Keyword matching (6 cases)
- `plugin-config-variations` - Config variations (6 cases)
- `chat-completions-progressive-stress` - Progressive QPS stress test

**Total: 17 test cases** (4 Istio-specific + 13 common functionality)

**Usage:**

```bash
# Run all Istio tests
make e2e-test E2E_PROFILE=istio

# Run specific Istio tests
make e2e-test-specific E2E_PROFILE=istio E2E_TESTS="istio-sidecar-health-check,istio-mtls-verification"

# Run with verbose output
./bin/e2e -profile istio -verbose

# Keep cluster for debugging
make e2e-test E2E_PROFILE=istio E2E_KEEP_CLUSTER=true
```

**Architecture:**

```
┌─────────────────────────────────────────┐
│         Istio Ingress Gateway            │
│      (istio-system namespace)            │
│   Port 80 → semantic-router service      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    Semantic Router Pod                   │
│  (semantic-router namespace)             │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │   Main      │  │  Istio-Proxy     │  │
│  │ Container   │◄─┤  (Envoy Sidecar) │  │
│  │             │  │                  │  │
│  │  :8801      │  │  mTLS, Tracing   │  │
│  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│         Istiod (Control Plane)           │
│  - Config distribution                   │
│  - Certificate management (mTLS)         │
│  - Sidecar injection                     │
└─────────────────────────────────────────┘
```

**Key Features Tested:**

**Istio Integration:**

- ✅ **Automatic Sidecar Injection**: Istio automatically injects Envoy proxy sidecars into pods
- ✅ **Traffic Management**: Requests route through Istio Gateway → VirtualService → Semantic Router
- ✅ **Security (mTLS)**: Automatic mutual TLS encryption and authentication between services
- ✅ **Observability**: Distributed tracing, metrics collection, and access logs
- ✅ **Service Mesh Integration**: Semantic Router operates correctly within Istio mesh

**Test Coverage:**

Istio-Specific Tests (4):

- ✅ **istio-sidecar-health-check**: Validates sidecar injection and health
- ✅ **istio-traffic-routing**: Tests routing through Gateway and VirtualService
- ✅ **istio-mtls-verification**: Confirms mTLS configuration and certificates
- ✅ **istio-tracing-observability**: Validates distributed tracing and metrics

Common Functionality Tests (13):

- ✅ **Chat Completions**: API functionality and stress testing
- ✅ **Classification**: Domain-based routing with 65 test cases
- ✅ **Security Features**: PII detection, jailbreak detection, semantic cache
- ✅ **Signal-Decision Engine**: Priority routing, plugin chains, keyword matching, fallback behavior
- ✅ **Load Handling**: Progressive stress testing (10-100 QPS)

**Total: 17 comprehensive test cases validating both Istio integration and Semantic Router functionality through the service mesh**

**Setup Steps (Automated by Profile):**

1. Install Istio control plane using Helm (base, istiod, ingress gateway)
2. Create namespace with `istio-injection=enabled` label
3. Deploy Semantic Router via Helm (sidecar auto-injected)
4. Create Istio Gateway and VirtualService for traffic routing
5. Create DestinationRule for mTLS configuration
6. Verify all components are ready

**Troubleshooting:**

If tests fail, check:

```bash
# Check Istio installation
kubectl get pods -n istio-system

# Check sidecar injection
kubectl get pods -n semantic-router -o jsonpath='{.items[*].spec.containers[*].name}'

# Check Istio resources
kubectl get gateway,virtualservice,destinationrule -n semantic-router

# Check mTLS configuration
kubectl get destinationrule semantic-router -n semantic-router -o yaml

# View Istio proxy logs
kubectl logs -n semantic-router <pod-name> -c istio-proxy
```

**Related Resources:**

- [Istio Documentation](https://istio.io/latest/docs/)
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [Istio Security (mTLS)](https://istio.io/latest/docs/concepts/security/)
- [Istio Observability](https://istio.io/latest/docs/concepts/observability/)
