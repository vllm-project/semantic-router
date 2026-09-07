package testcases

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("reliability-timeouts", pkgtestcases.TestCase{
		Description: "Verify runtime enforcement of distinct request deadlines, stalled stream idle timeouts, and short connect timeouts (issue #3210)",
		Tags:        []string{"reliability", "timeout", "streaming", "connect", "kubernetes"},
		Fn:          testReliabilityTimeouts,
	})
	pkgtestcases.Register("reliability-distinct-deadlines", pkgtestcases.TestCase{
		Description: "Verify per-model request timeouts enforce distinct deadlines across models",
		Tags:        []string{"reliability", "timeout", "deadline"},
		Fn:          testReliabilityDistinctDeadlines,
	})
	pkgtestcases.Register("reliability-stalled-streams", pkgtestcases.TestCase{
		Description: "Verify stream idle timeout terminates stalled streams between chunks",
		Tags:        []string{"reliability", "timeout", "streaming"},
		Fn:          testReliabilityStalledStreams,
	})
	pkgtestcases.Register("reliability-short-connect-failures", pkgtestcases.TestCase{
		Description: "Verify short connect timeout fails fast on unreachable backend endpoints",
		Tags:        []string{"reliability", "timeout", "connect"},
		Fn:          testReliabilityShortConnectFailures,
	})
}

// TimeoutProbeModel names used for deadline, streaming, and connect verification
const (
	timeoutProbeFastModel        = "timeout-probe-fast"
	timeoutProbeSlowModel        = "timeout-probe-slow"
	timeoutProbeUnreachableModel = "timeout-probe-unreachable"

	// Expected timeouts configured in e2e/profiles/ai-gateway/values.yaml
	expectedFastDeadline         = 2 * time.Second
	expectedFastDeadlineMinBound = 1800 * time.Millisecond
	expectedFastDeadlineMaxBound = 3500 * time.Millisecond

	expectedSlowDeadline         = 15 * time.Second
	expectedSlowProbeDelay       = 4 * time.Second
	expectedSlowDeadlineMinBound = 3800 * time.Millisecond
	expectedSlowDeadlineMaxBound = 6000 * time.Millisecond

	expectedStreamIdleTimeout  = 2 * time.Second
	expectedStreamIdleMinBound = 1800 * time.Millisecond
	expectedStreamIdleMaxBound = 4500 * time.Millisecond

	expectedConnectTimeout  = 1 * time.Second
	expectedConnectMinBound = 900 * time.Millisecond
	expectedConnectMaxBound = 3000 * time.Millisecond
)

func testReliabilityTimeouts(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Running comprehensive reliability timeouts test suite")
	}

	var failures []error

	if err := testReliabilityDistinctDeadlines(ctx, client, opts); err != nil {
		failures = append(failures, fmt.Errorf("distinct-deadlines: %w", err))
	}
	if err := testReliabilityStalledStreams(ctx, client, opts); err != nil {
		failures = append(failures, fmt.Errorf("stalled-streams: %w", err))
	}
	if err := testReliabilityShortConnectFailures(ctx, client, opts); err != nil {
		failures = append(failures, fmt.Errorf("short-connect-failures: %w", err))
	}
	if err := verifyTimeoutMetrics(ctx, client, opts); err != nil {
		failures = append(failures, fmt.Errorf("timeout-metrics: %w", err))
	}

	return errors.Join(failures...)
}

func probeModelDeadline(ctx context.Context, localPort, model, prompt string, timeout time.Duration) (time.Duration, int, error) {
	start := time.Now()
	resp, err := sendLocalChatCompletion(ctx, localPort, model, prompt, timeout)
	elapsed := time.Since(start)

	if err != nil && !isTimeoutOrConnectionError(err) {
		return elapsed, 0, fmt.Errorf("unexpected error on %s: %w", model, err)
	}
	statusCode := 0
	if resp != nil {
		statusCode = resp.StatusCode
	}
	return elapsed, statusCode, nil
}

func testReliabilityDistinctDeadlines(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing distinct deadlines per model with controllable header delay")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	// Fast model has request_timeout: 2s. Sending a 4s header delay forces Envoy's
	// per-model deadline to trigger, returning HTTP 504 Gateway Timeout in ~2s.
	fastPrompt := "__mock_header_delay_4s__ hello fast deadline probe"
	fastElapsed, fastStatus, err := probeModelDeadline(ctx, localPort, timeoutProbeFastModel, fastPrompt, 15*time.Second)
	if err != nil {
		return fmt.Errorf("fast model probe failed: %w", err)
	}
	if fastStatus != http.StatusGatewayTimeout {
		return fmt.Errorf("expected status %d (Gateway Timeout) for fast model exceeding deadline, got %d",
			http.StatusGatewayTimeout, fastStatus)
	}
	if err := evaluateTimeoutBounds("fast model deadline", fastElapsed, expectedFastDeadlineMinBound, expectedFastDeadlineMaxBound); err != nil {
		return err
	}

	// Slow model has request_timeout: 15s. Sending a 4s header delay forces the request
	// to take 4s, which exceeds the fast model's 2s deadline but completes well within
	// the slow model's 15s deadline, returning HTTP 200 OK after ~4s.
	slowPrompt := "__mock_header_delay_4s__ hello slow deadline probe"
	slowElapsed, slowStatus, slowErr := probeModelDeadline(ctx, localPort, timeoutProbeSlowModel, slowPrompt, 20*time.Second)
	if slowErr != nil {
		return fmt.Errorf("slow model probe failed: %w", slowErr)
	}
	if slowStatus != http.StatusOK {
		return fmt.Errorf("expected status %d (OK) for slow model within deadline, got %d",
			http.StatusOK, slowStatus)
	}
	if err := evaluateTimeoutBounds("slow model deadline", slowElapsed, expectedSlowDeadlineMinBound, expectedSlowDeadlineMaxBound); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"distinct_deadlines_verified": true,
			"fast_model_status":           fastStatus,
			"fast_model_elapsed_ms":       fastElapsed.Milliseconds(),
			"slow_model_status":           slowStatus,
			"slow_model_elapsed_ms":       slowElapsed.Milliseconds(),
		})
	}
	return nil
}

func sendStalledStreamRequest(ctx context.Context, localPort, model, prompt string) (*http.Response, time.Time, error) {
	reqBody := map[string]interface{}{
		"model":  model,
		"stream": true,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	start := time.Now()
	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	return resp, start, err
}

func scanStreamForCompletion(body io.Reader) (int, bool) {
	chunks := 0
	hasDone := false
	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		chunks++
		if strings.Contains(scanner.Text(), "[DONE]") {
			hasDone = true
		}
	}
	return chunks, hasDone
}

func testReliabilityStalledStreams(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing stalled stream idle timeout with mid-stream frame stall")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	stalledPrompt := "__mock_frame_stall_15s__ hello stalled stream probe"
	resp, start, err := sendStalledStreamRequest(ctx, localPort, timeoutProbeFastModel, stalledPrompt)
	if err != nil {
		return fmt.Errorf("failed to initiate stalled stream request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected initial status 200 for stalled stream probe, got %d", resp.StatusCode)
	}

	chunksReceived, streamHasDone := scanStreamForCompletion(resp.Body)
	elapsed := time.Since(start)

	if streamHasDone {
		return fmt.Errorf("stalled stream path unexpectedly emitted [DONE]")
	}
	if chunksReceived < 1 {
		return fmt.Errorf("stalled stream expected at least 1 initial chunk before frame stall, got %d", chunksReceived)
	}
	if err := evaluateTimeoutBounds("stalled stream idle timeout", elapsed, expectedStreamIdleMinBound, expectedStreamIdleMaxBound); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"stalled_streams_verified": true,
			"chunks_received":          chunksReceived,
			"elapsed_ms":               elapsed.Milliseconds(),
			"prematurely_terminated":   !streamHasDone,
			"typed_reset_outcome":      "stream_idle_timeout",
		})
	}
	return nil
}

func testReliabilityShortConnectFailures(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing short connect timeout failure through Envoy cluster")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	// Send request through Envoy for unreachable model cluster with short connect_timeout (1s).
	start := time.Now()
	resp, err := sendLocalChatCompletion(ctx, localPort, timeoutProbeUnreachableModel, "connect failure probe", 10*time.Second)
	elapsed := time.Since(start)

	// An upstream connect failure through Envoy must return an HTTP failure or transport error
	if resp != nil && resp.StatusCode == http.StatusOK {
		return fmt.Errorf("expected connect failure through Envoy cluster, but got 200 OK")
	}

	if err != nil && !isTimeoutOrConnectionError(err) {
		return fmt.Errorf("unexpected error on connect failure probe: %w", err)
	}

	// Verify exact status: Envoy returns 503 Service Unavailable on cluster connect timeout (UF)
	if resp != nil && resp.StatusCode != http.StatusServiceUnavailable {
		return fmt.Errorf("expected status %d (Service Unavailable) on connect timeout, got %d",
			http.StatusServiceUnavailable, resp.StatusCode)
	}

	// Verify the failure occurred within bounded duration reflecting the short connect timeout
	if err := evaluateTimeoutBounds("connect timeout", elapsed, expectedConnectMinBound, expectedConnectMaxBound); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		statusCode := 0
		if resp != nil {
			statusCode = resp.StatusCode
		}
		opts.SetDetails(map[string]interface{}{
			"short_connect_verified": true,
			"elapsed_ms":             elapsed.Milliseconds(),
			"status_code":            statusCode,
			"typed_connect_outcome":  "connect_timeout",
		})
	}
	return nil
}

// evaluateTimeoutBounds verifies that elapsed time is within [minBound, maxBound].
func evaluateTimeoutBounds(name string, elapsed, minBound, maxBound time.Duration) error {
	if elapsed < minBound {
		return fmt.Errorf("%s took %v, below minimum bound %v", name, elapsed, minBound)
	}
	if elapsed > maxBound {
		return fmt.Errorf("%s took %v, exceeded maximum bound %v", name, elapsed, maxBound)
	}
	return nil
}

// verifyTimeoutMetrics verifies that the router's Prometheus /metrics endpoint records
// timeout errors in llm_request_errors_total.
func verifyTimeoutMetrics(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if client == nil || opts.RestConfig == nil {
		return nil
	}
	metricsSession, err := fixtures.OpenSemanticRouterMetricsSession(ctx, client, opts)
	if err != nil {
		if opts.Verbose {
			fmt.Printf("[Test] Note: metrics session not accessible: %v\n", err)
		}
		return nil
	}
	defer metricsSession.Close()

	metricsHTTP := metricsSession.HTTPClient(10 * time.Second)
	metricsResp, err := fixtures.DoGETRequest(ctx, metricsHTTP, metricsSession.URL("/metrics"))
	if err != nil {
		return fmt.Errorf("fetch /metrics: %w", err)
	}
	if metricsResp.StatusCode != http.StatusOK {
		return fmt.Errorf("/metrics: expected 200, got %d", metricsResp.StatusCode)
	}

	body := string(metricsResp.Body)
	if !strings.Contains(body, "llm_request_errors_total") || !strings.Contains(body, `reason="timeout"`) {
		return fmt.Errorf("metrics body missing llm_request_errors_total with reason=\"timeout\": %s", body)
	}
	return nil
}

// isTimeoutOrConnectionError checks if an error indicates a network timeout or connection reset.
func isTimeoutOrConnectionError(err error) bool {
	if err == nil {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "timeout") ||
		strings.Contains(msg, "deadline exceeded") ||
		strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "reset by peer") ||
		strings.Contains(msg, "eof")
}
