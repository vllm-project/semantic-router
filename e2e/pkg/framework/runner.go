package framework

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/cluster"
	"github.com/vllm-project/semantic-router/e2e/pkg/docker"
	"github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// Runner orchestrates the E2E test execution
type Runner struct {
	opts    *TestOptions
	profile Profile
	cluster *cluster.KindCluster
	builder *docker.Builder
}

// NewRunner creates a new test runner
func NewRunner(opts *TestOptions, profile Profile) *Runner {
	return &Runner{
		opts:    opts,
		profile: profile,
		cluster: cluster.NewKindCluster(opts.ClusterName, opts.Verbose),
		builder: docker.NewBuilder(opts.Verbose),
	}
}

// Run executes the E2E tests
func (r *Runner) Run(ctx context.Context) error {
	r.log("Starting E2E tests for profile: %s", r.profile.Name())
	r.log("Description: %s", r.profile.Description())

	// Step 1: Setup cluster
	if !r.opts.UseExistingCluster {
		if err := r.setupCluster(ctx); err != nil {
			return fmt.Errorf("failed to setup cluster: %w", err)
		}

		if !r.opts.KeepCluster {
			defer r.cleanupCluster(ctx)
		}
	}

	// Step 2: Build and load Docker images
	if err := r.buildAndLoadImages(ctx); err != nil {
		return fmt.Errorf("failed to build and load images: %w", err)
	}

	// Step 3: Get kubeconfig and create Kubernetes client
	kubeConfig, err := r.cluster.GetKubeConfig(ctx)
	if err != nil {
		return fmt.Errorf("failed to get kubeconfig: %w", err)
	}

	config, err := clientcmd.BuildConfigFromFlags("", kubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	kubeClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create Kubernetes client: %w", err)
	}

	// Step 4: Setup profile (deploy Helm charts, etc.)
	setupOpts := &SetupOptions{
		KubeClient:  kubeClient,
		KubeConfig:  kubeConfig,
		ClusterName: r.opts.ClusterName,
		ImageTag:    r.opts.ImageTag,
		Verbose:     r.opts.Verbose,
	}

	if err := r.profile.Setup(ctx, setupOpts); err != nil {
		return fmt.Errorf("failed to setup profile: %w", err)
	}

	defer func() {
		teardownOpts := &TeardownOptions{
			KubeClient:  kubeClient,
			KubeConfig:  kubeConfig,
			ClusterName: r.opts.ClusterName,
			Verbose:     r.opts.Verbose,
		}
		r.profile.Teardown(context.Background(), teardownOpts)
	}()

	// Step 5: Run tests
	results, err := r.runTests(ctx, kubeClient)
	if err != nil {
		return fmt.Errorf("failed to run tests: %w", err)
	}

	// Step 6: Print results
	r.printResults(results)

	// Check if any tests failed
	for _, result := range results {
		if !result.Passed {
			return fmt.Errorf("some tests failed")
		}
	}

	r.log("✅ All tests passed!")
	return nil
}

func (r *Runner) setupCluster(ctx context.Context) error {
	r.log("Setting up Kind cluster: %s", r.opts.ClusterName)
	return r.cluster.Create(ctx)
}

func (r *Runner) cleanupCluster(ctx context.Context) {
	r.log("Cleaning up Kind cluster: %s", r.opts.ClusterName)
	if err := r.cluster.Delete(ctx); err != nil {
		r.log("Warning: failed to delete cluster: %v", err)
	}
}

func (r *Runner) buildAndLoadImages(ctx context.Context) error {
	r.log("Building and loading Docker images")

	buildOpts := docker.BuildOptions{
		Dockerfile:   "Dockerfile.extproc",
		Tag:          fmt.Sprintf("ghcr.io/vllm-project/semantic-router/extproc:%s", r.opts.ImageTag),
		BuildContext: ".",
	}

	return r.builder.BuildAndLoad(ctx, r.opts.ClusterName, buildOpts)
}

func (r *Runner) runTests(ctx context.Context, kubeClient *kubernetes.Clientset) ([]TestResult, error) {
	r.log("Running tests")

	// Get test cases to run
	var testCasesToRun []testcases.TestCase
	var err error

	if len(r.opts.TestCases) > 0 {
		// Run specific test cases
		testCasesToRun, err = testcases.ListByNames(r.opts.TestCases...)
		if err != nil {
			return nil, err
		}
	} else {
		// Run all test cases for the profile
		profileTestCases := r.profile.GetTestCases()
		testCasesToRun, err = testcases.ListByNames(profileTestCases...)
		if err != nil {
			return nil, err
		}
	}

	r.log("Running %d test cases", len(testCasesToRun))

	results := make([]TestResult, 0, len(testCasesToRun))
	resultsMu := sync.Mutex{}

	if r.opts.Parallel {
		// Run tests in parallel
		var wg sync.WaitGroup
		for _, tc := range testCasesToRun {
			wg.Add(1)
			go func(tc testcases.TestCase) {
				defer wg.Done()
				result := r.runSingleTest(ctx, kubeClient, tc)
				resultsMu.Lock()
				results = append(results, result)
				resultsMu.Unlock()
			}(tc)
		}
		wg.Wait()
	} else {
		// Run tests sequentially
		for _, tc := range testCasesToRun {
			result := r.runSingleTest(ctx, kubeClient, tc)
			results = append(results, result)
		}
	}

	return results, nil
}

func (r *Runner) runSingleTest(ctx context.Context, kubeClient *kubernetes.Clientset, tc testcases.TestCase) TestResult {
	r.log("Running test: %s", tc.Name)

	start := time.Now()

	opts := testcases.TestCaseOptions{
		Verbose:   r.opts.Verbose,
		Namespace: "default",
		Timeout:   "5m",
	}

	err := tc.Fn(ctx, kubeClient, opts)
	duration := time.Since(start)

	result := TestResult{
		Name:     tc.Name,
		Passed:   err == nil,
		Error:    err,
		Duration: duration.String(),
	}

	if err != nil {
		r.log("❌ Test %s failed: %v", tc.Name, err)
	} else {
		r.log("✅ Test %s passed (%s)", tc.Name, duration)
	}

	return result
}

func (r *Runner) printResults(results []TestResult) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("TEST RESULTS")
	fmt.Println(strings.Repeat("=", 80))

	passed := 0
	failed := 0

	for _, result := range results {
		status := "✅ PASSED"
		if !result.Passed {
			status = "❌ FAILED"
			failed++
		} else {
			passed++
		}

		fmt.Printf("%s - %s (%s)\n", status, result.Name, result.Duration)
		if result.Error != nil {
			fmt.Printf("  Error: %v\n", result.Error)
		}
	}

	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("Total: %d | Passed: %d | Failed: %d\n", len(results), passed, failed)
	fmt.Println(strings.Repeat("=", 80))
}

func (r *Runner) log(format string, args ...interface{}) {
	if r.opts.Verbose {
		fmt.Printf("[Runner] "+format+"\n", args...)
	}
}
