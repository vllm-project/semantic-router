# E2E Test Framework

A comprehensive end-to-end testing framework for Semantic Router with support for multiple deployment profiles.

## Architecture

The framework follows a **separation of concerns** design:

- **Profiles**: Define deployment environments and configurations
- **Test Cases**: Reusable test logic that can be shared across profiles
- **Framework**: Core infrastructure for test execution and reporting

### Supported Profiles

- **ai-gateway**: Tests Semantic Router with Envoy AI Gateway integration
- **istio**: Tests Semantic Router with Istio Gateway (future)
- **production-stack**: Tests vLLM Production Stack configurations (future)
- **llm-d**: Tests with LLM-D (future)
- **dynamo**: Tests with Nvidia Dynamo (future)
- **aibrix**: Tests with vLLM AIBrix (future)

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
│   └── jailbreak_detection.go
├── profiles/
│   └── ai-gateway/       # AI Gateway test profile
│       └── profile.go    # Profile definition and environment setup
└── README.md
```

## Available Test Cases

The framework includes the following test cases (all in `e2e/testcases/`):

| Test Case | Description | Metrics |
|-----------|-------------|---------|
| `chat-completions-request` | Basic chat completions API test | Response validation |
| `chat-completions-stress-request` | Sequential stress test with 1000 requests | Success rate, avg duration |
| `chat-completions-progressive-stress` | Progressive QPS stress test (10/20/50/100 QPS) | Per-stage success rate, latency stats |
| `domain-classify` | Domain classification accuracy | 65 cases, accuracy rate |
| `semantic-cache` | Semantic cache hit rate | 5 groups, cache hit rate |
| `pii-detection` | PII detection and blocking | 10 PII types, detection rate, block rate |
| `jailbreak-detection` | Jailbreak attack detection | 10 attack types, detection rate, block rate |

All test cases:

- Use model name `"MoM"`
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
