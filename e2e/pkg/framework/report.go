package framework

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// TestReport represents the complete test report
type TestReport struct {
	// Metadata
	Profile     string    `json:"profile"`
	StartTime   time.Time `json:"start_time"`
	EndTime     time.Time `json:"end_time"`
	Duration    string    `json:"duration"`
	Status      string    `json:"status"` // "PASSED" or "FAILED"
	ExitCode    int       `json:"exit_code"`
	ClusterName string    `json:"cluster_name"`
	GoVersion   string    `json:"go_version,omitempty"`
	KindVersion string    `json:"kind_version,omitempty"`

	// Test Results
	ExpectedCases []string     `json:"expected_cases"`
	TestResults   []TestResult `json:"test_results"`
	TotalTests    int          `json:"total_tests"`
	PassedTests   int          `json:"passed_tests"`
	FailedTests   int          `json:"failed_tests"`
	FlakyTests    int          `json:"flaky_tests"`

	// Cluster Information
	ClusterInfo ClusterInfo `json:"cluster_info"`

	// Environment
	Environment map[string]string `json:"environment,omitempty"`
}

// ClusterInfo contains information about the Kubernetes cluster
type ClusterInfo struct {
	TotalPods   int      `json:"total_pods"`
	RunningPods int      `json:"running_pods"`
	PendingPods int      `json:"pending_pods"`
	FailedPods  int      `json:"failed_pods"`
	Namespaces  []string `json:"namespaces"`

	// Pod details by namespace
	PodsByNamespace map[string][]PodInfo `json:"pods_by_namespace"`

	// Gateway resources
	Gateways   []string `json:"gateways,omitempty"`
	HTTPRoutes []string `json:"http_routes,omitempty"`
}

// PodInfo contains information about a pod
type PodInfo struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	Phase     string `json:"phase"`
	Ready     string `json:"ready"`
	Restarts  int32  `json:"restarts"`
	Age       string `json:"age"`
	Node      string `json:"node,omitempty"`
}

// ReportGenerator generates test reports
type ReportGenerator struct {
	client    *kubernetes.Clientset
	report    *TestReport
	startTime time.Time
}

// NewReportGenerator creates a new report generator
func NewReportGenerator(profile, clusterName string) *ReportGenerator {
	return &ReportGenerator{
		report: &TestReport{
			Profile:     profile,
			ClusterName: clusterName,
			StartTime:   time.Now(),
			Environment: make(map[string]string),
		},
		startTime: time.Now(),
	}
}

// SetKubeClient sets the Kubernetes client for collecting cluster info
func (rg *ReportGenerator) SetKubeClient(client *kubernetes.Clientset) {
	rg.client = client
}

// AddTestResults adds test results to the report
func (rg *ReportGenerator) AddTestResults(results []TestResult) {
	rg.report.TestResults = results
	rg.report.TotalTests = len(results)

	passedCount := 0
	flakyCount := 0
	for _, result := range results {
		if result.Passed {
			passedCount++
		}
		if result.Flaked {
			flakyCount++
		}
	}
	rg.report.PassedTests = passedCount
	rg.report.FailedTests = rg.report.TotalTests - passedCount
	rg.report.FlakyTests = flakyCount
}

// flakyTestNames lists the tests that only passed after a retry.
func flakyTestNames(results []TestResult) []string {
	names := make([]string, 0)
	for _, result := range results {
		if result.Flaked {
			names = append(names, result.Name)
		}
	}
	return names
}

// SetEnvironment sets environment variables in the report
func (rg *ReportGenerator) SetEnvironment(key, value string) {
	rg.report.Environment[key] = value
}

// CollectClusterInfo collects information about the Kubernetes cluster
func (rg *ReportGenerator) CollectClusterInfo(ctx context.Context) error {
	if rg.client == nil {
		return fmt.Errorf("kubernetes client not set")
	}

	clusterInfo := ClusterInfo{
		PodsByNamespace: make(map[string][]PodInfo),
	}

	// Get all namespaces
	namespaces, err := rg.client.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to list namespaces: %w", err)
	}

	for _, ns := range namespaces.Items {
		clusterInfo.Namespaces = append(clusterInfo.Namespaces, ns.Name)
	}

	// Get all pods
	pods, err := rg.client.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to list pods: %w", err)
	}

	clusterInfo.TotalPods = len(pods.Items)

	for _, pod := range pods.Items {
		// Count by phase
		switch pod.Status.Phase {
		case "Running":
			clusterInfo.RunningPods++
		case "Pending":
			clusterInfo.PendingPods++
		case "Failed":
			clusterInfo.FailedPods++
		}

		// Calculate ready containers
		readyCount := 0
		totalCount := len(pod.Status.ContainerStatuses)
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Ready {
				readyCount++
			}
		}

		// Calculate restart count
		restartCount := int32(0)
		for _, cs := range pod.Status.ContainerStatuses {
			restartCount += cs.RestartCount
		}

		// Calculate age
		age := ""
		if pod.Status.StartTime != nil {
			age = time.Since(pod.Status.StartTime.Time).Round(time.Second).String()
		}

		podInfo := PodInfo{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Phase:     string(pod.Status.Phase),
			Ready:     fmt.Sprintf("%d/%d", readyCount, totalCount),
			Restarts:  restartCount,
			Age:       age,
			Node:      pod.Spec.NodeName,
		}

		clusterInfo.PodsByNamespace[pod.Namespace] = append(
			clusterInfo.PodsByNamespace[pod.Namespace],
			podInfo,
		)
	}

	rg.report.ClusterInfo = clusterInfo
	return nil
}

// Finalize finalizes the report with end time and status
func (rg *ReportGenerator) Finalize(exitCode int) {
	rg.report.EndTime = time.Now()
	rg.report.Duration = rg.report.EndTime.Sub(rg.report.StartTime).Round(time.Second).String()
	rg.report.ExitCode = exitCode

	if exitCode == 0 {
		rg.report.Status = "PASSED"
	} else {
		rg.report.Status = "FAILED"
	}
}

// WriteJSON writes the report to a JSON file
func (rg *ReportGenerator) WriteJSON(filename string) error {
	data, err := json.MarshalIndent(rg.report, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal report: %w", err)
	}

	if err := os.WriteFile(filename, data, 0o644); err != nil {
		return fmt.Errorf("failed to write report file: %w", err)
	}

	return nil
}

// WriteMarkdown writes the report to a Markdown file
func (rg *ReportGenerator) WriteMarkdown(filename string) error {
	md := rg.generateMarkdown()

	if err := os.WriteFile(filename, []byte(md), 0o644); err != nil {
		return fmt.Errorf("failed to write markdown report: %w", err)
	}

	return nil
}

// generateMarkdown generates a Markdown report
func (rg *ReportGenerator) generateMarkdown() string {
	r := rg.report

	statusEmoji := "✅"
	statusColor := "🟢"
	if r.Status == "FAILED" {
		statusEmoji = "❌"
		statusColor = "🔴"
	}

	md := fmt.Sprintf(`## %s E2E Integration Test Report - %s

**Status:** %s **%s**
**Profile:** `+"`%s`"+`
**Duration:** %s
**Cluster:** `+"`%s`"+`

### 📊 Test Results

| Metric | Value |
|--------|-------|
| Exit Code | `+"`%d`"+` |
| Total Tests | %d |
| Passed Tests | %d |
| Failed Tests | %d |
| Flaky Tests | %d |
| Success Rate | %.1f%% |

### 🔧 Cluster Statistics

| Metric | Value |
|--------|-------|
| Total Pods | %d |
| Running Pods | %d |
| Pending Pods | %d |
| Failed Pods | %d |
| Namespaces | %d |

`,
		statusEmoji, r.Profile,
		statusColor, r.Status, r.Profile,
		r.Duration, r.ClusterName,
		r.ExitCode,
		r.TotalTests, r.PassedTests, r.FailedTests, r.FlakyTests,
		float64(r.PassedTests)/float64(r.TotalTests)*100,
		r.ClusterInfo.TotalPods,
		r.ClusterInfo.RunningPods,
		r.ClusterInfo.PendingPods,
		r.ClusterInfo.FailedPods,
		len(r.ClusterInfo.Namespaces),
	)

	// Add test case results
	if flaky := flakyTestNames(r.TestResults); len(flaky) > 0 {
		md += fmt.Sprintf(
			"\n> ⚠️ **%d test(s) passed only after a retry and are flaky:** %s\n",
			len(flaky), strings.Join(flaky, ", "),
		)
	}

	md += failedAndSlowestMarkdown(r.TestResults)

	md += "\n### 📝 Test Cases\n\n"
	for _, result := range r.TestResults {
		status := "✅"
		switch {
		case !result.Passed:
			status = "❌"
		case result.Flaked:
			status = "⚠️"
		}
		errorMsg := ""
		if result.Error != nil {
			errorMsg = fmt.Sprintf(" - Error: `%s`", result.Error.Error())
		}
		flakyNote := ""
		if result.Flaked {
			flakyNote = fmt.Sprintf(" - passed on attempt %d", result.Attempts)
		}
		priority := ""
		if result.Priority != "" {
			priority = strings.ToUpper(result.Priority) + ", "
		}
		md += fmt.Sprintf("- %s **%s** (%s%s)%s%s\n", status, result.Name, priority, result.Duration, flakyNote, errorMsg)

		// Add details if available
		if len(result.Details) > 0 {
			md += "  <details>\n  <summary>Details</summary>\n\n"
			md += "  | Metric | Value |\n"
			md += "  |--------|-------|\n"
			for _, key := range slices.Sorted(maps.Keys(result.Details)) {
				md += fmt.Sprintf("  | %s | %v |\n", key, result.Details[key])
			}
			md += "\n  </details>\n"
		}
	}

	// Add pod details by namespace
	md += "\n### 📦 Deployed Components\n\n"
	for _, ns := range r.ClusterInfo.Namespaces {
		pods, ok := r.ClusterInfo.PodsByNamespace[ns]
		if !ok || len(pods) == 0 {
			continue
		}

		md += fmt.Sprintf("<details>\n<summary>Namespace: %s (%d pods)</summary>\n\n", ns, len(pods))
		md += "| Pod | Phase | Ready | Restarts | Age |\n"
		md += "|-----|-------|-------|----------|-----|\n"

		for _, pod := range pods {
			md += fmt.Sprintf("| %s | %s | %s | %d | %s |\n",
				pod.Name, pod.Phase, pod.Ready, pod.Restarts, pod.Age)
		}

		md += "\n</details>\n\n"
	}

	// Add debugging section
	md += "### 🔍 Debugging\n\n"
	if r.Status == "FAILED" {
		md += "**Test failed!** Check the details above:\n\n"
		md += "- Review failed test cases in the Test Cases section\n"
		md += "- Check pod status in the Deployed Components section\n"
		md += "- Verify all pods are in Running state\n\n"
	} else {
		md += "**All tests passed!** 🎉\n\n"
		md += fmt.Sprintf("The %s integration is working correctly with all components deployed and healthy.\n\n", r.Profile)
	}

	return md
}

// GetReport returns the current report
func (rg *ReportGenerator) GetReport() *TestReport {
	return rg.report
}

// failedAndSlowestMarkdown lists the failed cases with their errors, then the
// five slowest cases, above the full list, so a reader finds both without
// scanning every case.
func failedAndSlowestMarkdown(results []TestResult) string {
	md := ""
	for _, result := range results {
		if result.Passed {
			continue
		}
		if md == "" {
			md = "\n### ❌ Failed\n\n"
		}
		message := "failed"
		if result.Error != nil {
			message = result.Error.Error()
		}
		md += fmt.Sprintf("- **%s**: `%s`\n", result.Name, message)
	}
	slowest := slices.Clone(results)
	slices.SortStableFunc(slowest, func(a, b TestResult) int {
		da, _ := time.ParseDuration(a.Duration)
		db, _ := time.ParseDuration(b.Duration)
		return cmp.Compare(db, da)
	})
	if len(slowest) > 5 {
		slowest = slowest[:5]
	}
	if len(slowest) > 1 {
		md += "\n### ⏱ Slowest\n\n"
		for _, result := range slowest {
			md += fmt.Sprintf("- %s (%s)\n", result.Name, result.Duration)
		}
	}
	return md
}
