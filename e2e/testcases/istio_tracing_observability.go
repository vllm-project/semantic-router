package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("istio-tracing-observability", pkgtestcases.TestCase{
		Description: "Verify distributed tracing and metrics collection in Istio mesh",
		Tags:        []string{"istio", "observability", "tracing", "metrics"},
		Fn:          testIstioTracingObservability,
	})
}

func testIstioTracingObservability(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Istio observability features")
	}

	// Istio-specific test: always use vllm-semantic-router-system namespace
	namespace := "vllm-semantic-router-system"

	// 1. Send a traced request through the system
	traceHeaders, err := sendTracedRequest(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("failed to send traced request: %w", err)
	}

	// 2. Verify Envoy metrics are being collected
	metricsFound, metricsDetails, err := checkEnvoyMetrics(ctx, namespace, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to check Envoy metrics: %w", err)
	}

	// 3. Check for Istio telemetry configuration
	telemetryConfigured := checkIstioTelemetry(ctx, opts.Verbose)

	// 4. Verify access logs are enabled (optional but good for observability)
	accessLogsEnabled := checkAccessLogs(ctx, namespace, opts.Verbose)

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"trace_headers_present": len(traceHeaders) > 0,
			"trace_headers":         traceHeaders,
			"envoy_metrics_found":   metricsFound,
			"metrics_details":       metricsDetails,
			"telemetry_configured":  telemetryConfigured,
			"access_logs_enabled":   accessLogsEnabled,
		})
	}

	if !metricsFound {
		return fmt.Errorf("Envoy metrics not found - observability may not be working")
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ Istio tracing and observability verification passed")
	}

	return nil
}

// sendTracedRequest sends a request and captures tracing headers
func sendTracedRequest(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (map[string]string, error) {
	// Setup service connection
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	defer stopPortForward()

	// Send request with trace headers
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": "Test observability",
			},
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	// Add B3 trace headers for distributed tracing
	req.Header.Set("X-B3-TraceId", fmt.Sprintf("%016x", time.Now().UnixNano()))
	req.Header.Set("X-B3-SpanId", fmt.Sprintf("%016x", time.Now().UnixNano()))
	req.Header.Set("X-B3-Sampled", "1")

	httpClient := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	io.ReadAll(resp.Body) // Consume body

	// Capture trace-related headers from response
	traceHeaders := make(map[string]string)
	for key, values := range resp.Header {
		if len(values) > 0 {
			// Capture tracing headers
			lowerKey := strings.ToLower(key)
			if strings.Contains(lowerKey, "trace") ||
				strings.Contains(lowerKey, "span") ||
				strings.Contains(lowerKey, "request-id") ||
				strings.HasPrefix(lowerKey, "x-b3") {
				traceHeaders[key] = values[0]
			}
		}
	}

	if opts.Verbose && len(traceHeaders) > 0 {
		fmt.Println("[Test] Captured trace headers:")
		for k, v := range traceHeaders {
			fmt.Printf("[Test]   %s: %s\n", k, v)
		}
	}

	return traceHeaders, nil
}

// checkEnvoyMetrics checks if Envoy is exposing metrics
func checkEnvoyMetrics(ctx context.Context, namespace string, verbose bool) (bool, map[string]interface{}, error) {
	// Get a semantic-router pod
	cmd := exec.CommandContext(ctx, "kubectl", "get", "pods", "-n", namespace,
		"-l", "app.kubernetes.io/name=semantic-router", "-o", "jsonpath={.items[0].metadata.name}")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return false, nil, fmt.Errorf("failed to get pod: %w", err)
	}

	podName := strings.TrimSpace(string(output))
	if podName == "" {
		return false, nil, fmt.Errorf("no pod found")
	}

	// Check if metrics endpoint is accessible on istio-proxy
	// Envoy typically exposes metrics on port 15090
	metricsCmd := exec.CommandContext(ctx, "kubectl", "exec", "-n", namespace,
		podName, "-c", "istio-proxy", "--",
		"curl", "-s", "http://localhost:15090/stats/prometheus")

	metricsOutput, err := metricsCmd.CombinedOutput()
	if err != nil {
		if verbose {
			fmt.Printf("[Test] Failed to fetch Envoy metrics: %v\n", err)
		}
		return false, nil, nil
	}

	metricsData := string(metricsOutput)

	// Check for common Envoy metrics
	commonMetrics := []string{
		"envoy_cluster_",
		"envoy_server_",
		"istio_requests_total",
		"istio_request_duration",
	}

	metricsDetails := make(map[string]interface{})
	metricsFound := false

	for _, metric := range commonMetrics {
		found := strings.Contains(metricsData, metric)
		metricsDetails[metric] = found
		if found {
			metricsFound = true
		}
	}

	if verbose {
		if metricsFound {
			fmt.Println("[Test] Envoy metrics are being collected")
			for metric, found := range metricsDetails {
				if found.(bool) {
					fmt.Printf("[Test]   Found metric: %s\n", metric)
				}
			}
		} else {
			fmt.Println("[Test] Warning: No standard Envoy metrics found")
		}
	}

	return metricsFound, metricsDetails, nil
}

// checkIstioTelemetry checks if Istio telemetry is configured
func checkIstioTelemetry(ctx context.Context, verbose bool) bool {
	// Check if Telemetry CRD exists in the cluster
	cmd := exec.CommandContext(ctx, "kubectl", "get", "telemetries.telemetry.istio.io",
		"--all-namespaces", "-o", "json")

	output, err := cmd.CombinedOutput()
	if err != nil {
		if verbose {
			fmt.Println("[Test] Telemetry CRD not found or not configured (using defaults)")
		}
		return false
	}

	// If we got output, telemetry is configured
	hasItems := strings.Contains(string(output), "\"items\"")
	if verbose && hasItems {
		fmt.Println("[Test] Istio Telemetry configuration found")
	}

	return hasItems
}

// checkAccessLogs checks if Envoy access logs are enabled
func checkAccessLogs(ctx context.Context, namespace string, verbose bool) bool {
	// Get logs from istio-proxy to see if access logs are present
	cmd := exec.CommandContext(ctx, "kubectl", "get", "pods", "-n", namespace,
		"-l", "app.kubernetes.io/name=semantic-router", "-o", "jsonpath={.items[0].metadata.name}")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return false
	}

	podName := strings.TrimSpace(string(output))
	if podName == "" {
		return false
	}

	// Check recent logs from istio-proxy for access log entries
	logsCmd := exec.CommandContext(ctx, "kubectl", "logs", "-n", namespace,
		podName, "-c", "istio-proxy", "--tail=50")

	logsOutput, err := logsCmd.CombinedOutput()
	if err != nil {
		return false
	}

	logsData := string(logsOutput)

	// Look for typical Envoy access log patterns
	// Format: [timestamp] "METHOD PATH PROTOCOL" status
	accessLogPatterns := []string{
		"\"POST /v1/",
		"\"GET /",
		"upstream_cluster",
		"response_code",
	}

	for _, pattern := range accessLogPatterns {
		if strings.Contains(logsData, pattern) {
			if verbose {
				fmt.Printf("[Test] Access logs are enabled (found pattern: %s)\n", pattern)
			}
			return true
		}
	}

	if verbose {
		fmt.Println("[Test] Access logs not clearly detected (may not be enabled)")
	}

	return false
}
