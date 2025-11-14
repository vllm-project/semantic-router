package testcases

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/client-go/kubernetes"
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

// ListByTags returns test cases matching any of the given tags
func ListByTags(tags ...string) []TestCase {
	mu.RLock()
	defer mu.RUnlock()

	tagSet := make(map[string]bool)
	for _, tag := range tags {
		tagSet[tag] = true
	}

	cases := make([]TestCase, 0)
	for _, tc := range registry {
		for _, tag := range tc.Tags {
			if tagSet[tag] {
				cases = append(cases, tc)
				break
			}
		}
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

