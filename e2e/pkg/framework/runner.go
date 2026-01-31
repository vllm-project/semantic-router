package framework

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/cluster"
	"github.com/vllm-project/semantic-router/e2e/pkg/docker"
	"github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// Runner orchestrates the E2E test execution
type Runner struct {
	opts       *TestOptions
	profile    Profile
	cluster    *cluster.KindCluster
	builder    *docker.Builder
	restConfig *rest.Config
	reporter   *ReportGenerator
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

	// Initialize report generator
	r.reporter = NewReportGenerator(r.opts.Profile, r.opts.ClusterName)
	r.reporter.SetEnvironment("go_version", "1.24")
	r.reporter.SetEnvironment("verbose", fmt.Sprintf("%v", r.opts.Verbose))
	r.reporter.SetEnvironment("parallel", fmt.Sprintf("%v", r.opts.Parallel))

	exitCode := 0

	// Defer report generation
	defer func() {
		r.reporter.Finalize(exitCode)

		// Write reports
		if err := r.reporter.WriteJSON("test-report.json"); err != nil {
			r.log("Warning: failed to write JSON report: %v", err)
		} else {
			r.log("Test report written to: test-report.json")
		}

		if err := r.reporter.WriteMarkdown("test-report.md"); err != nil {
			r.log("Warning: failed to write Markdown report: %v", err)
		} else {
			r.log("Test report written to: test-report.md")
		}
	}()

	// Step 1: Setup cluster
	if !r.opts.UseExistingCluster {
		if err := r.setupCluster(ctx); err != nil {
			exitCode = 1
			return fmt.Errorf("failed to setup cluster: %w", err)
		}

		if !r.opts.KeepCluster {
			defer r.cleanupCluster(ctx)
		}
	}

	// Step 2: Build and load Docker images
	if err := r.buildAndLoadImages(ctx); err != nil {
		exitCode = 1
		return fmt.Errorf("failed to build and load images: %w", err)
	}

	// Step 3: Get kubeconfig and create Kubernetes client
	kubeConfig, err := r.cluster.GetKubeConfig(ctx)
	if err != nil {
		exitCode = 1
		return fmt.Errorf("failed to get kubeconfig: %w", err)
	}

	config, err := clientcmd.BuildConfigFromFlags("", kubeConfig)
	if err != nil {
		exitCode = 1
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	// Store rest config for test cases
	r.restConfig = config

	kubeClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		exitCode = 1
		return fmt.Errorf("failed to create Kubernetes client: %w", err)
	}

	// Set Kubernetes client for report generator
	r.reporter.SetKubeClient(kubeClient)

	// Create HF_TOKEN secret if available (for gated model downloads)
	if os.Getenv("HF_TOKEN") != "" {
		if err := r.createHFTokenSecret(ctx, kubeClient); err != nil {
			r.log("⚠️  Warning: Failed to create HF_TOKEN secret: %v", err)
			r.log("   Model downloads may fail if gated models (e.g., embeddinggemma-300m) are required")
		} else {
			r.log("✅ Created HF_TOKEN secret for gated model downloads")
		}
	} else {
		r.log("ℹ️  HF_TOKEN not set - gated models (e.g., embeddinggemma-300m) may not be downloadable")
	}

	// Step 4: Setup profile (deploy Helm charts, etc.)
	if !r.opts.SkipSetup {
		setupOpts := &SetupOptions{
			KubeClient:  kubeClient,
			KubeConfig:  kubeConfig,
			ClusterName: r.opts.ClusterName,
			ImageTag:    r.opts.ImageTag,
			Verbose:     r.opts.Verbose,
		}

		if err := r.profile.Setup(ctx, setupOpts); err != nil {
			exitCode = 1
			return fmt.Errorf("failed to setup profile: %w", err)
		}

		// Only register teardown if not in setup-only mode
		if !r.opts.SetupOnly {
			defer func() {
				teardownOpts := &TeardownOptions{
					KubeClient:  kubeClient,
					KubeConfig:  kubeConfig,
					ClusterName: r.opts.ClusterName,
					Verbose:     r.opts.Verbose,
				}
				r.profile.Teardown(context.Background(), teardownOpts)
			}()
		}
	} else {
		r.log("⏭️  Skipping profile setup (--skip-setup enabled)")
	}

	// If setup-only mode, stop here
	if r.opts.SetupOnly {
		r.log("✅ Profile setup complete (--setup-only mode)")
		r.log("💡 Cluster is ready. You can now:")
		r.log("   - Run tests manually: ./bin/e2e -profile %s -skip-setup -use-existing-cluster", r.opts.Profile)
		r.log("   - Inspect the cluster: kubectl --context kind-%s get pods -A", r.opts.ClusterName)
		r.log("   - Clean up when done: make e2e-cleanup")
		return nil
	}

	// Step 5: Run tests
	results, err := r.runTests(ctx, kubeClient)
	if err != nil {
		exitCode = 1
		return fmt.Errorf("failed to run tests: %w", err)
	}

	// Add test results to report
	r.reporter.AddTestResults(results)

	// Collect cluster information for report
	if err := r.reporter.CollectClusterInfo(ctx); err != nil {
		r.log("Warning: failed to collect cluster info: %v", err)
	}

	// Step 6: Print results
	r.printResults(results)

	// Check if any tests failed
	hasFailures := false
	for _, result := range results {
		if !result.Passed {
			hasFailures = true
			break
		}
	}

	// Step 7: Collect semantic-router logs (always, regardless of test result)
	r.log("📝 Collecting semantic-router logs...")
	if err := r.collectSemanticRouterLogs(ctx, kubeClient); err != nil {
		r.log("Warning: failed to collect semantic-router logs: %v", err)
	}

	if hasFailures {
		exitCode = 1
		r.log("❌ Some tests failed, printing all pods status for debugging...")

		// Print summary of all pods
		r.printAllPodsDebugInfo(ctx, kubeClient)

		// Print detailed status and logs for all pods
		if r.opts.Verbose {
			PrintAllPodsStatus(ctx, kubeClient)
		}

		return fmt.Errorf("some tests failed")
	}

	r.log("✅ All tests passed!")
	return nil
}

func (r *Runner) setupCluster(ctx context.Context) error {
	r.log("Setting up Kind cluster: %s", r.opts.ClusterName)

	// Enable GPU support for dynamo profile
	if r.profile.Name() == "dynamo" {
		r.log("Enabling GPU support for Dynamo profile")
		r.cluster.SetGPUEnabled(true)
	}

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
		Dockerfile:   "tools/docker/Dockerfile.extproc",
		Tag:          fmt.Sprintf("ghcr.io/vllm-project/semantic-router/extproc:%s", r.opts.ImageTag),
		BuildContext: ".",
	}

	if err := r.builder.BuildAndLoad(ctx, r.opts.ClusterName, buildOpts); err != nil {
		return err
	}

	// Profiles may require additional local images beyond extproc.
	switch r.profile.Name() {
	case "response-api", "response-api-redis", "response-api-redis-cluster", "ml-model-selection":
		mockOpts := docker.BuildOptions{
			Dockerfile:   "tools/mock-vllm/Dockerfile",
			Tag:          "ghcr.io/vllm-project/semantic-router/mock-vllm:latest",
			BuildContext: "tools/mock-vllm",
		}
		return r.builder.BuildAndLoad(ctx, r.opts.ClusterName, mockOpts)
	default:
		return nil
	}
}

func (r *Runner) runTests(ctx context.Context, kubeClient *kubernetes.Clientset) ([]TestResult, error) {
	r.log("Running tests")

	// Debug: List all registered test cases
	if r.opts.Verbose {
		r.log("All registered test cases:")
		for _, tc := range testcases.List() {
			r.log("  - %s: %s", tc.Name, tc.Description)
		}
	}

	// Get test cases to run
	var testCasesToRun []testcases.TestCase
	var err error

	if len(r.opts.TestCases) > 0 {
		// Run specific test cases
		r.log("Requested test cases: %v", r.opts.TestCases)
		testCasesToRun, err = testcases.ListByNames(r.opts.TestCases...)
		if err != nil {
			return nil, err
		}
	} else {
		// Run all test cases for the profile
		profileTestCases := r.profile.GetTestCases()
		r.log("Profile test cases: %v", profileTestCases)
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

	// Get service configuration from profile
	svcConfig := r.profile.GetServiceConfig()

	// Create result to capture details
	result := TestResult{
		Name:    tc.Name,
		Details: make(map[string]interface{}),
	}

	opts := testcases.TestCaseOptions{
		Verbose:    r.opts.Verbose,
		Namespace:  "default",
		Timeout:    "5m",
		RestConfig: r.restConfig,
		ServiceConfig: testcases.ServiceConfig{
			LabelSelector: svcConfig.LabelSelector,
			Namespace:     svcConfig.Namespace,
			Name:          svcConfig.Name,
			PortMapping:   svcConfig.PortMapping,
		},
		SetDetails: func(details map[string]interface{}) {
			result.Details = details
		},
	}

	err := tc.Fn(ctx, kubeClient, opts)
	duration := time.Since(start)

	result.Passed = err == nil
	result.Error = err
	result.Duration = duration.String()

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

func (r *Runner) printAllPodsDebugInfo(ctx context.Context, client *kubernetes.Clientset) {
	fmt.Printf("\n")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("DEBUGGING INFORMATION - ALL PODS STATUS")
	fmt.Println(strings.Repeat("=", 80))

	// Get all pods from all namespaces
	pods, err := client.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		fmt.Printf("Failed to list pods: %v\n", err)
		return
	}

	fmt.Printf("\nTotal pods across all namespaces: %d\n", len(pods.Items))

	// Group pods by namespace
	podsByNamespace := make(map[string][]string)
	for _, pod := range pods.Items {
		status := fmt.Sprintf("%s (Phase: %s, Ready: %s)",
			pod.Name,
			pod.Status.Phase,
			getPodReadyStatus(pod))
		podsByNamespace[pod.Namespace] = append(podsByNamespace[pod.Namespace], status)
	}

	// Print summary by namespace
	fmt.Printf("\nPods by namespace:\n")
	for ns, podList := range podsByNamespace {
		fmt.Printf("\n  Namespace: %s (%d pods)\n", ns, len(podList))
		for _, podStatus := range podList {
			fmt.Printf("    - %s\n", podStatus)
		}
	}

	fmt.Printf("\n")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("\n")
}

// collectSemanticRouterLogs collects logs from semantic-router pods and saves to file
func (r *Runner) collectSemanticRouterLogs(ctx context.Context, client *kubernetes.Clientset) error {
	// Find semantic-router pods
	pods, err := client.CoreV1().Pods("vllm-semantic-router-system").List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to list semantic-router pods: %w", err)
	}

	if len(pods.Items) == 0 {
		r.log("Warning: no semantic-router pods found")
		return nil
	}

	// Collect logs from all semantic-router pods
	var allLogs strings.Builder
	allLogs.WriteString("========================================\n")
	allLogs.WriteString("Semantic Router Logs\n")
	allLogs.WriteString("========================================\n\n")

	for _, pod := range pods.Items {
		allLogs.WriteString(fmt.Sprintf("=== Pod: %s (Namespace: %s) ===\n", pod.Name, pod.Namespace))
		allLogs.WriteString(fmt.Sprintf("Status: %s\n", pod.Status.Phase))
		allLogs.WriteString(fmt.Sprintf("Node: %s\n", pod.Spec.NodeName))
		if pod.Status.StartTime != nil {
			allLogs.WriteString(fmt.Sprintf("Started: %s\n", pod.Status.StartTime.Format(time.RFC3339)))
		}
		allLogs.WriteString("\n")

		// Collect logs from all containers in the pod
		for _, container := range pod.Spec.Containers {
			allLogs.WriteString(fmt.Sprintf("--- Container: %s ---\n", container.Name))

			logOptions := &corev1.PodLogOptions{
				Container: container.Name,
			}

			req := client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, logOptions)
			logs, err := req.Stream(ctx)
			if err != nil {
				allLogs.WriteString(fmt.Sprintf("Error getting logs: %v\n", err))
				continue
			}

			logBytes, err := io.ReadAll(logs)
			logs.Close()
			if err != nil {
				allLogs.WriteString(fmt.Sprintf("Error reading logs: %v\n", err))
				continue
			}

			if len(logBytes) == 0 {
				allLogs.WriteString("(no logs available)\n")
			} else {
				allLogs.Write(logBytes)
				allLogs.WriteString("\n")
			}

			allLogs.WriteString("\n")
		}

		allLogs.WriteString("\n")
	}

	// Write logs to file
	logFilename := "semantic-router-logs.txt"
	if err := os.WriteFile(logFilename, []byte(allLogs.String()), 0644); err != nil {
		return fmt.Errorf("failed to write log file: %w", err)
	}

	r.log("✅ Semantic router logs saved to: %s", logFilename)
	return nil
}

// createHFTokenSecret creates a Kubernetes secret for HF_TOKEN if it's available in the environment
// This is required for semantic-router to auto-download gated models like google/embeddinggemma-300m
// The secret must be in the same namespace as the semantic-router deployment (vllm-semantic-router-system)
// because Kubernetes secrets are namespace-scoped
func (r *Runner) createHFTokenSecret(ctx context.Context, kubeClient *kubernetes.Clientset) error {
	hfToken := os.Getenv("HF_TOKEN")
	if hfToken == "" {
		return nil // No token to create
	}

	// All E2E profiles deploy semantic-router to this namespace
	nsName := "vllm-semantic-router-system"

	// First, ensure the namespace exists
	_, err := kubeClient.CoreV1().Namespaces().Get(ctx, nsName, metav1.GetOptions{})
	if err != nil {
		// Namespace doesn't exist, create it
		ns := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: nsName,
			},
		}
		_, err = kubeClient.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
		if err != nil && !strings.Contains(err.Error(), "already exists") {
			// If we can't create the namespace, that's okay - the profile will create it
			r.log("⚠️  Could not create namespace %s (will be created by profile): %v", nsName, err)
		}
	}

	// Create the secret in the namespace where semantic-router is deployed
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "hf-token-secret",
			Namespace: nsName,
		},
		Type: corev1.SecretTypeOpaque,
		StringData: map[string]string{
			"token": hfToken,
		},
	}

	_, err = kubeClient.CoreV1().Secrets(nsName).Create(ctx, secret, metav1.CreateOptions{})
	if err != nil {
		// If secret already exists, update it
		if strings.Contains(err.Error(), "already exists") {
			_, err = kubeClient.CoreV1().Secrets(nsName).Update(ctx, secret, metav1.UpdateOptions{})
			if err != nil {
				return fmt.Errorf("failed to update existing HF_TOKEN secret in %s: %w", nsName, err)
			}
			return nil
		}
		// If namespace still doesn't exist, that's okay - it will be created by Helm
		if strings.Contains(err.Error(), "not found") {
			r.log("⚠️  Namespace %s not found yet (will be created by profile)", nsName)
			return nil
		}
		return fmt.Errorf("failed to create HF_TOKEN secret in %s: %w", nsName, err)
	}

	return nil
}

func getPodReadyStatus(pod corev1.Pod) string {
	readyCount := 0
	totalCount := len(pod.Status.ContainerStatuses)
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Ready {
			readyCount++
		}
	}
	return fmt.Sprintf("%d/%d", readyCount, totalCount)
}
