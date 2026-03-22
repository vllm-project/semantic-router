package framework

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"github.com/vllm-project/semantic-router/e2e/pkg/cluster"
	"github.com/vllm-project/semantic-router/e2e/pkg/docker"
	"github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// Runner orchestrates the E2E test execution
type Runner struct {
	opts                *TestOptions
	profile             Profile
	profileCapabilities ProfileCapabilities
	cluster             *cluster.KindCluster
	builder             *docker.Builder
	restConfig          *rest.Config
	reporter            *ReportGenerator
}

// NewRunner creates a new test runner
func NewRunner(opts *TestOptions, profile Profile) *Runner {
	capabilities := ProfileCapabilities{}
	if reg, ok := LookupProfileRegistration(profile.Name()); ok {
		capabilities = reg.Capabilities
	}

	return &Runner{
		opts:                opts,
		profile:             profile,
		profileCapabilities: capabilities,
		cluster:             cluster.NewKindCluster(opts.ClusterName, opts.Verbose),
		builder:             docker.NewBuilder(opts.Verbose),
	}
}

func (r *Runner) setupCluster(ctx context.Context) error {
	r.log("Setting up Kind cluster: %s", r.opts.ClusterName)

	if r.profileCapabilities.RequiresGPU {
		r.log("Enabling GPU support for profile %s", r.profile.Name())
		r.cluster.SetGPUEnabled(true)
	}

	if r.opts.UseWorkspaceModels {
		workspaceModelsDir, err := resolveWorkspaceModelsDir()
		if err != nil {
			return err
		}
		r.log("Using workspace models directory: %s", workspaceModelsDir)
		r.cluster.SetWorkspaceModelsDir(workspaceModelsDir)
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
		BuildArgs:    localDockerBuildArgs(),
	}

	if err := r.builder.BuildAndLoad(ctx, r.opts.ClusterName, buildOpts); err != nil {
		return err
	}

	for _, image := range r.profileCapabilities.LocalImages {
		buildOpts := docker.BuildOptions{
			Dockerfile:   image.Dockerfile,
			Tag:          image.Tag,
			BuildContext: image.BuildContext,
		}
		if err := r.builder.BuildAndLoad(ctx, r.opts.ClusterName, buildOpts); err != nil {
			return err
		}
	}

	return nil
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
		Verbose:       r.opts.Verbose,
		Profile:       r.profile.Name(),
		Namespace:     "default",
		Timeout:       "5m",
		RestConfig:    r.restConfig,
		ServiceConfig: svcConfig,
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

func resolveWorkspaceModelsDir() (string, error) {
	workingDir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("resolve workspace models directory: %w", err)
	}

	modelsDir := filepath.Join(workingDir, "models")
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		return "", fmt.Errorf("create workspace models directory %s: %w", modelsDir, err)
	}

	return modelsDir, nil
}

func localDockerBuildArgs() map[string]string {
	return map[string]string{
		"BUILDPLATFORM": "linux/" + runtime.GOARCH,
		"TARGETARCH":    runtime.GOARCH,
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
		r.appendPodLogs(ctx, client, &allLogs, pod)
	}

	// Write logs to file
	logFilename := "semantic-router-logs.txt"
	if err := os.WriteFile(logFilename, []byte(allLogs.String()), 0644); err != nil {
		return fmt.Errorf("failed to write log file: %w", err)
	}

	r.log("✅ Semantic router logs saved to: %s", logFilename)
	return nil
}

func (r *Runner) appendPodLogs(
	ctx context.Context,
	client *kubernetes.Clientset,
	allLogs *strings.Builder,
	pod corev1.Pod,
) {
	fmt.Fprintf(allLogs, "=== Pod: %s (Namespace: %s) ===\n", pod.Name, pod.Namespace)
	fmt.Fprintf(allLogs, "Status: %s\n", pod.Status.Phase)
	fmt.Fprintf(allLogs, "Node: %s\n", pod.Spec.NodeName)
	if pod.Status.StartTime != nil {
		fmt.Fprintf(allLogs, "Started: %s\n", pod.Status.StartTime.Format(time.RFC3339))
	}
	allLogs.WriteString("\n")

	for _, container := range pod.Spec.Containers {
		r.appendContainerLogs(ctx, client, allLogs, pod, container.Name)
	}

	allLogs.WriteString("\n")
}

func (r *Runner) appendContainerLogs(
	ctx context.Context,
	client *kubernetes.Clientset,
	allLogs *strings.Builder,
	pod corev1.Pod,
	containerName string,
) {
	fmt.Fprintf(allLogs, "--- Container: %s ---\n", containerName)

	logOptions := &corev1.PodLogOptions{Container: containerName}
	req := client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, logOptions)
	logs, err := req.Stream(ctx)
	if err != nil {
		fmt.Fprintf(allLogs, "Error getting logs: %v\n\n", err)
		return
	}

	logBytes, readErr := io.ReadAll(logs)
	if closeErr := logs.Close(); closeErr != nil {
		r.log("Warning: failed to close log stream for %s/%s: %v", pod.Name, containerName, closeErr)
	}
	if readErr != nil {
		fmt.Fprintf(allLogs, "Error reading logs: %v\n\n", readErr)
		return
	}

	if len(logBytes) == 0 {
		allLogs.WriteString("(no logs available)\n\n")
		return
	}

	allLogs.Write(logBytes)
	allLogs.WriteString("\n\n")
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
