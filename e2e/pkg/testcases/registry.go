package testcases

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// TestCase represents a single test case
type TestCase struct {
	// Name is the unique identifier for the test case
	Name string

	// Description describes what the test does
	Description string

	// Tags are optional tags for filtering tests
	Tags []string

	// Fn is the test function to execute
	Fn func(ctx context.Context, client *kubernetes.Clientset, opts TestCaseOptions) error
}

// TestCaseOptions contains options passed to test cases
type TestCaseOptions struct {
	// Verbose enables verbose logging
	Verbose bool

	// Namespace is the Kubernetes namespace to use
	Namespace string

	// ServiceURL is the URL of the service to test
	ServiceURL string

	// Timeout is the test timeout duration
	Timeout string

	// RestConfig is the Kubernetes REST config
	RestConfig *rest.Config

	// ServiceConfig contains service-specific configuration
	ServiceConfig ServiceConfig

	// SetDetails allows test cases to set structured details for reporting
	// Example: opts.SetDetails(map[string]interface{}{"accuracy": 95.5, "total_tests": 100})
	SetDetails func(details map[string]interface{})
}

// ServiceConfig contains configuration for accessing a service
type ServiceConfig struct {
	// LabelSelector is the label selector to find the service
	// Example: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router"
	LabelSelector string

	// Namespace is the namespace where the service is located
	Namespace string

	// Name is the service name (optional, if not using label selector)
	Name string

	// PortMapping is the port mapping for port-forwarding
	// Format: "localPort:servicePort" (e.g., "8080:80")
	PortMapping string
}

var (
	registry = make(map[string]TestCase)
	mu       sync.RWMutex
)

// Register registers a test case
func Register(name string, tc TestCase) {
	mu.Lock()
	defer mu.Unlock()

	if _, exists := registry[name]; exists {
		panic(fmt.Sprintf("test case %q already registered", name))
	}

	tc.Name = name
	registry[name] = tc
}

// Get retrieves a test case by name
func Get(name string) (TestCase, bool) {
	mu.RLock()
	defer mu.RUnlock()

	tc, ok := registry[name]
	return tc, ok
}

// List returns all registered test cases
func List() []TestCase {
	mu.RLock()
	defer mu.RUnlock()

	cases := make([]TestCase, 0, len(registry))
	for _, tc := range registry {
		cases = append(cases, tc)
	}
	return cases
}

// ListByNames returns test cases matching the given names
func ListByNames(names ...string) ([]TestCase, error) {
	mu.RLock()
	defer mu.RUnlock()

	cases := make([]TestCase, 0, len(names))
	for _, name := range names {
		tc, ok := registry[name]
		if !ok {
			return nil, fmt.Errorf("test case %q not found", name)
		}
		cases = append(cases, tc)
	}
	return cases, nil
}
