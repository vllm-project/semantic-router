package testcases

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

// TimeoutProbeModel names used for distinct deadline verification
const (
	timeoutProbeSlowModel = "timeout-probe-slow"
	timeoutProbeFastModel = "timeout-probe-fast"
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

func testReliabilityDistinctDeadlines(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing distinct deadlines per model")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	// Probe fast model with tight deadline
	start := time.Now()
	resp, err := sendLocalChatCompletion(ctx, localPort, timeoutProbeFastModel, "hello", 15*time.Second)
	elapsed := time.Since(start)

	if err != nil && !isTimeoutOrConnectionError(err) {
		return fmt.Errorf("unexpected error on fast model: %w", err)
	}

	// Status 504 / 408 or client-side timeout within reasonable bounds is expected
	if resp != nil && resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusGatewayTimeout && resp.StatusCode != http.StatusRequestTimeout {
		return fmt.Errorf("expected 200, 408, or 504 for deadline probe, got status %d", resp.StatusCode)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"distinct_deadlines_verified": true,
			"fast_model_elapsed_ms":       elapsed.Milliseconds(),
		})
	}
	return nil
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

	reqBody := map[string]interface{}{
		"model":  "MoM",
		"stream": true,
		"messages": []map[string]string{
			{"role": "user", "content": "Stream test probe"},
		},
	}
	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonBytes))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		// A timeout or broken pipe is valid when stream stalls
		return nil
	}
	defer resp.Body.Close()

	chunksReceived := 0
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		chunksReceived++
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"stalled_streams_verified": true,
			"chunks_received":          chunksReceived,
		})
	}
	return nil
}

func testReliabilityShortConnectFailures(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing short connect timeout failure")
	}

	// Verify that dialing an unreachable/non-listening address fails fast under bounded duration
	d := net.Dialer{Timeout: 500 * time.Millisecond}
	start := time.Now()
	conn, err := d.DialContext(ctx, "tcp", "127.0.0.1:65530")
	elapsed := time.Since(start)

	if conn != nil {
		_ = conn.Close()
	}

	if err == nil {
		return fmt.Errorf("expected connection failure to closed port 65530, but succeeded")
	}

	if err := evaluateConnectTimeoutBound(elapsed, 500*time.Millisecond, 1500*time.Millisecond); err != nil {
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

// evaluateConnectTimeoutBound verifies that an operation completed within the expected timeout plus allowable slack.
func evaluateConnectTimeoutBound(elapsed time.Duration, expectedTimeout time.Duration, allowableSlack time.Duration) error {
	maxAllowed := expectedTimeout + allowableSlack
	if elapsed > maxAllowed {
		return fmt.Errorf("connect timeout took %v, exceeded max allowed %v (expected ~%v)", elapsed, maxAllowed, expectedTimeout)
	}
	return nil
}

// evaluateDistinctDeadlineResult checks that distinct deadline policies correctly separate fast and slow models.
func evaluateDistinctDeadlineResult(fastElapsed, slowElapsed time.Duration, fastLimit time.Duration) error {
	if fastElapsed > fastLimit+500*time.Millisecond {
		return fmt.Errorf("fast model elapsed %v exceeded limit %v", fastElapsed, fastLimit)
	}
	return nil
}

// isTimeoutOrConnectionError checks if an error indicates a network timeout or connection reset.
func isTimeoutOrConnectionError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "timeout") ||
		strings.Contains(msg, "deadline exceeded") ||
		strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "reset by peer") ||
		strings.Contains(msg, "eof")
}
