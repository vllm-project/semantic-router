# E2E Test Framework

A comprehensive end-to-end testing framework for Semantic Router with support for multiple deployment profiles.

## Architecture

The framework is designed to be extensible and supports multiple test profiles:

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
│   └── testcases/        # Test case definitions
├── profiles/
│   ├── ai-gateway/       # AI Gateway test profile
│   ├── istio/            # Istio test profile (future)
│   └── ...
└── README.md
```

## Quick Start

### Run all tests with default profile (ai-gateway)

```bash
make e2e-test
```

### Run specific profile

```bash
make e2e-test PROFILE=ai-gateway
```

### Run with custom options

```bash
# Keep cluster after test
make e2e-test KEEP_CLUSTER=true

# Use existing cluster
make e2e-test USE_EXISTING_CLUSTER=true

# Verbose output
make e2e-test VERBOSE=true
```

## Adding New Test Profiles

1. Create a new directory under `profiles/`
2. Implement the `Profile` interface
3. Register test cases using the test case registry
4. Add profile-specific deployment configurations

See `profiles/ai-gateway/` for a complete example.

## Test Case Registration

Test cases are registered using a simple function-based approach:

```go
func init() {
    testcases.Register("my-test", testcases.TestCase{
        Name:        "My Test",
        Description: "Description of what this test does",
        Fn: func(ctx context.Context, client *kubernetes.Clientset) error {
            // Test implementation
            return nil
        },
    })
}
```

## Framework Features

- **Automatic cluster lifecycle management**: Creates and cleans up Kind clusters
- **Docker image building and loading**: Builds images and loads them into Kind
- **Helm deployment automation**: Deploys required Helm charts
- **Parallel test execution**: Runs independent tests in parallel
- **Detailed logging**: Provides comprehensive test output
- **Resource cleanup**: Ensures proper cleanup even on failures

## Prerequisites

Before running E2E tests, ensure you have the following tools installed:

- [Go](https://golang.org/doc/install) 1.24 or later
- [Docker](https://docs.docker.com/get-docker/)
- [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Helm](https://helm.sh/docs/intro/install/)

## Getting Started

### 1. Install dependencies

```bash
make e2e-deps
```

### 2. Build the E2E test binary

```bash
make build-e2e
```

### 3. Run tests

```bash
# Run all tests with default profile (ai-gateway)
make e2e-test

# Run with verbose output
make e2e-test E2E_VERBOSE=true

# Run and keep cluster for debugging
make e2e-test-debug

# Run specific test cases
make e2e-test-specific E2E_TESTS="basic-health-check,chat-completions-request"
```

## CI Integration

The E2E tests are automatically run in GitHub Actions on:

- Pull requests to `main` branch
- Pushes to `main` branch

See `.github/workflows/integration-test-ai-gateway.yml` for the CI configuration.

## Troubleshooting

### Cluster creation fails

```bash
# Clean up any existing cluster
make e2e-cleanup

# Try again
make e2e-test
```

### Tests fail with timeout

Increase the timeout in the test case or check if the cluster has enough resources:

```bash
# Check cluster status
kubectl get nodes
kubectl get pods --all-namespaces
```

### Port forward fails

Make sure no other process is using port 8080:

```bash
# Check what's using port 8080
lsof -i :8080

# Kill the process if needed
kill -9 <PID>
```

## Development

### Adding a new test case

1. Create a new test function in `profiles/<profile>/testcases.go`
2. Register it in the `init()` function
3. Add the test case name to the profile's `GetTestCases()` method

Example:

```go
func init() {
    testcases.Register("my-new-test", testcases.TestCase{
        Description: "My new test description",
        Tags:        []string{"ai-gateway", "functional"},
        Fn:          testMyNewFeature,
    })
}

func testMyNewFeature(ctx context.Context, client *kubernetes.Clientset, opts testcases.TestCaseOptions) error {
    // Test implementation
    return nil
}
```

### Adding a new profile

1. Create a new directory under `profiles/`
2. Implement the `Profile` interface
3. Register test cases
4. Update `cmd/e2e/main.go` to include the new profile

See `profiles/ai-gateway/` for a complete example.
