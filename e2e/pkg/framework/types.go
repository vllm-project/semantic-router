package framework

import (
	"context"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// Profile defines the interface that all test profiles must implement
type Profile interface {
	// Name returns the profile name
	Name() string

	// Setup prepares the environment for testing (e.g., deploy Helm charts)
	Setup(ctx context.Context, opts *SetupOptions) error

	// Teardown cleans up resources created during setup
	Teardown(ctx context.Context, opts *TeardownOptions) error

	// GetTestCases returns the list of test cases to run for this profile
	GetTestCases() []string

	// GetServiceConfig returns the service configuration for accessing the deployed service
	GetServiceConfig() ServiceConfig
}

// ServiceConfig is the canonical service-connection contract shared by profiles and testcases.
type ServiceConfig = pkgtestcases.ServiceConfig

// SetupOptions contains options for profile setup
type SetupOptions struct {
	// KubeClient is the Kubernetes client
	KubeClient *kubernetes.Clientset

	// KubeConfig is the path to kubeconfig file
	KubeConfig string

	// ClusterName is the name of the Kind cluster
	ClusterName string

	// ImageTag is the Docker image tag to use
	ImageTag string

	// Verbose enables verbose logging
	Verbose bool

	// ValuesFiles contains paths to Helm values files
	// keyed by Helm release name for profile-specific overlays.
	ValuesFiles map[string]string
}

// TeardownOptions contains options for profile teardown
type TeardownOptions struct {
	// KubeClient is the Kubernetes client
	KubeClient *kubernetes.Clientset

	// KubeConfig is the path to kubeconfig file
	KubeConfig string

	// ClusterName is the name of the Kind cluster
	ClusterName string

	// Verbose enables verbose logging
	Verbose bool
}

// TestOptions contains options for running tests
type TestOptions struct {
	// Profile is the test profile to run
	Profile string

	// ClusterName is the name of the Kind cluster
	ClusterName string

	// ImageTag is the Docker image tag to use
	ImageTag string

	// KeepCluster preserves the kind cluster and profile resources after the run.
	KeepCluster bool

	// UseExistingCluster uses an existing cluster instead of creating a new one
	UseExistingCluster bool

	// Verbose enables verbose logging
	Verbose bool

	// Parallel runs tests in parallel
	Parallel bool

	// TestCases is a list of specific test cases to run (empty means all)
	TestCases []string

	// SetupOnly only sets up the profile without running tests
	SetupOnly bool

	// SkipSetup skips profile setup and only runs tests (assumes environment is already deployed)
	SkipSetup bool

	// UseWorkspaceModels mounts the workspace models/ directory into semantic-router.
	UseWorkspaceModels bool
}

// TestResult represents the result of a test case
type TestResult struct {
	// Name is the test case name
	Name string

	// Passed indicates if the test passed
	Passed bool

	// Error contains the error if the test failed
	Error error

	// Duration is how long the test took
	Duration string

	// Logs contains test output logs
	Logs string

	// Details contains structured test-specific details (e.g., accuracy, hit rate)
	Details map[string]interface{}
}
