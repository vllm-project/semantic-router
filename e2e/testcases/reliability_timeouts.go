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

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
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
	expectedFastDeadline       = 2 * time.Second
	expectedFastDeadlineSlack  = 500 * time.Millisecond
	expectedSlowDeadline       = 15 * time.Second
	expectedSlowDeadlineSlack  = 1 * time.Second
	expectedStreamIdleTimeout  = 2 * time.Second
	expectedStreamIdleMaxSlack = 6 * time.Second
	expectedConnectTimeout     = 1 * time.Second
	expectedConnectSlack       = 4 * time.Second
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

	return errors.Join(failures...)
}

func probeModelDeadline(ctx context.Context, localPort, model, prompt string, timeout time.Duration) (time.Duration, error) {
	start := time.Now()
	resp, err := sendLocalChatCompletion(ctx, localPort, model, prompt, timeout)
	elapsed := time.Since(start)

	if err != nil && !isTimeoutOrConnectionError(err) {
		return elapsed, fmt.Errorf("unexpected error on %s: %w", model, err)
	}
	if resp != nil && !isAllowedProbeStatusCode(resp.StatusCode) {
		return elapsed, fmt.Errorf("expected 200, 408, or 504 for %s, got status %d", model, resp.StatusCode)
	}
	return elapsed, nil
}

func isAllowedProbeStatusCode(statusCode int) bool {
	return statusCode == http.StatusOK ||
		statusCode == http.StatusGatewayTimeout ||
		statusCode == http.StatusRequestTimeout
}

func testReliabilityDistinctDeadlines(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing distinct deadlines per model")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	fastElapsed, err := probeModelDeadline(ctx, localPort, timeoutProbeFastModel, "hello fast deadline probe", 15*time.Second)
	if err != nil {
		return err
	}

	slowElapsed, err := probeModelDeadline(ctx, localPort, timeoutProbeSlowModel, "hello slow deadline probe", 20*time.Second)
	if err != nil {
		return err
	}

	if err := evaluateTimeoutBound("fast model deadline", fastElapsed, expectedFastDeadline, expectedFastDeadlineSlack); err != nil {
		return err
	}
	if err := evaluateTimeoutBound("slow model deadline", slowElapsed, expectedSlowDeadline, expectedSlowDeadlineSlack); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"distinct_deadlines_verified": true,
			"fast_model_elapsed_ms":       fastElapsed.Milliseconds(),
			"slow_model_elapsed_ms":       slowElapsed.Milliseconds(),
		})
	}
	return nil
}

func sendStalledStreamRequest(ctx context.Context, localPort, model string) (*http.Response, time.Time, error) {
	reqBody := map[string]interface{}{
		"model":  model,
		"stream": true,
		"messages": []map[string]string{
			{"role": "user", "content": "__mock_incomplete_stream__"},
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
		fmt.Println("[Test] Testing stalled stream idle timeout")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	resp, start, err := sendStalledStreamRequest(ctx, localPort, timeoutProbeFastModel)
	if err != nil {
		return fmt.Errorf("failed to initiate stalled stream request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected status 200 for stalled stream probe, got %d", resp.StatusCode)
	}

	chunksReceived, streamHasDone := scanStreamForCompletion(resp.Body)
	elapsed := time.Since(start)

	if streamHasDone {
		return fmt.Errorf("stalled stream path unexpectedly emitted [DONE]")
	}
	if err := evaluateTimeoutBound("stalled stream idle timeout", elapsed, expectedStreamIdleTimeout, expectedStreamIdleMaxSlack); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"stalled_streams_verified": true,
			"chunks_received":          chunksReceived,
			"elapsed_ms":               elapsed.Milliseconds(),
			"prematurely_terminated":   !streamHasDone,
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

	// Send request through Envoy for unreachable model cluster with short connect_timeout (1s)
	start := time.Now()
	resp, err := sendLocalChatCompletion(ctx, localPort, timeoutProbeUnreachableModel, "connect failure probe", 10*time.Second)
	elapsed := time.Since(start)

	// An upstream connect failure through Envoy must return an HTTP failure (503/504) or transport error
	if resp != nil && resp.StatusCode == http.StatusOK {
		return fmt.Errorf("expected connect failure through Envoy cluster, but got 200 OK")
	}

	if err != nil && !isTimeoutOrConnectionError(err) {
		return fmt.Errorf("unexpected error on connect failure probe: %w", err)
	}

	// Verify the failure occurred within bounded duration reflecting the short connect timeout
	if err := evaluateTimeoutBound("connect timeout", elapsed, expectedConnectTimeout, expectedConnectSlack); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"short_connect_verified": true,
			"elapsed_ms":             elapsed.Milliseconds(),
		})
	}
	return nil
}

// evaluateTimeoutBound verifies that an operation completed within the expected timeout plus allowable slack.
func evaluateTimeoutBound(name string, elapsed, expectedTimeout, allowableSlack time.Duration) error {
	maxAllowed := expectedTimeout + allowableSlack
	if elapsed > maxAllowed {
		return fmt.Errorf("%s took %v, exceeded max allowed %v (expected ~%v)", name, elapsed, maxAllowed, expectedTimeout)
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
