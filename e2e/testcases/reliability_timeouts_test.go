package testcases

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestEvaluateTimeoutBounds(t *testing.T) {
	t.Parallel()

	minBound := 1 * time.Second
	maxBound := 3 * time.Second

	// Case 1: comfortably within bounds
	if err := evaluateTimeoutBounds("test within", 2*time.Second, minBound, maxBound); err != nil {
		t.Fatalf("expected nil for within bounds, got %v", err)
	}

	// Case 2: exactly at lower bound
	if err := evaluateTimeoutBounds("test min boundary", 1*time.Second, minBound, maxBound); err != nil {
		t.Fatalf("expected nil for lower boundary, got %v", err)
	}

	// Case 3: exactly at upper bound
	if err := evaluateTimeoutBounds("test max boundary", 3*time.Second, minBound, maxBound); err != nil {
		t.Fatalf("expected nil for upper boundary, got %v", err)
	}

	// Case 4: below lower bound
	errBelow := evaluateTimeoutBounds("test below", 500*time.Millisecond, minBound, maxBound)
	if errBelow == nil {
		t.Fatalf("expected error for below lower bound, got nil")
	}
	if !strings.Contains(errBelow.Error(), "below minimum bound") {
		t.Errorf("expected error message to contain 'below minimum bound', got: %v", errBelow)
	}

	// Case 5: exceeding upper bound
	errAbove := evaluateTimeoutBounds("test above", 4*time.Second, minBound, maxBound)
	if errAbove == nil {
		t.Fatalf("expected error for exceeding upper bound, got nil")
	}
	if !strings.Contains(errAbove.Error(), "exceeded maximum bound") {
		t.Errorf("expected error message to contain 'exceeded maximum bound', got: %v", errAbove)
	}
}

func TestScanStreamForCompletion(t *testing.T) {
	t.Parallel()

	// Complete SSE stream with [DONE]
	completeStream := strings.NewReader("data: {\"choices\":[...]}\n\ndata: [DONE]\n\n")
	chunks, hasDone := scanStreamForCompletion(completeStream)
	if !hasDone {
		t.Errorf("expected complete stream to have hasDone=true")
	}
	if chunks == 0 {
		t.Errorf("expected chunks > 0, got %d", chunks)
	}

	// Stalled/truncated SSE stream without [DONE]
	stalledStream := strings.NewReader("data: {\"choices\":[...]}\n\n")
	chunksStalled, hasDoneStalled := scanStreamForCompletion(stalledStream)
	if hasDoneStalled {
		t.Errorf("expected stalled stream to have hasDone=false")
	}
	if chunksStalled == 0 {
		t.Errorf("expected chunks > 0, got %d", chunksStalled)
	}

	// Empty body
	emptyStream := strings.NewReader("")
	chunksEmpty, hasDoneEmpty := scanStreamForCompletion(emptyStream)
	if hasDoneEmpty || chunksEmpty != 0 {
		t.Errorf("expected (0, false) for empty body, got (%d, %t)", chunksEmpty, hasDoneEmpty)
	}
}

func TestIsTimeoutOrConnectionError(t *testing.T) {
	t.Parallel()

	// Case 1: nil error
	if isTimeoutOrConnectionError(nil) {
		t.Errorf("expected nil error to return false")
	}

	// Case 2: Matched network/timeout error patterns
	matchErrors := []error{
		errors.New("context deadline exceeded"),
		errors.New("dial tcp 198.51.100.1:81: i/o timeout"),
		errors.New("dial tcp 127.0.0.1:19999: connect: connection refused"),
		errors.New("read tcp 127.0.0.1:8000: connection reset by peer"),
		errors.New("unexpected EOF"),
		errors.New("i/o timeout"),
	}
	for _, err := range matchErrors {
		if !isTimeoutOrConnectionError(err) {
			t.Errorf("expected error %q to be recognized as timeout or connection error", err)
		}
	}

	// Case 3: Unrelated error
	unrelatedErr := errors.New("invalid JSON syntax at position 42")
	if isTimeoutOrConnectionError(unrelatedErr) {
		t.Errorf("expected unrelated error %q to return false", unrelatedErr)
	}
}

func TestConfiguredTimeoutBoundsConstants(t *testing.T) {
	t.Parallel()

	// Fast deadline: 2s configured timeout, triggered around 2s. Lower bound < 2s, upper bound > 2s.
	if expectedFastDeadlineMinBound >= expectedFastDeadline {
		t.Errorf("expectedFastDeadlineMinBound (%v) should be < expectedFastDeadline (%v)",
			expectedFastDeadlineMinBound, expectedFastDeadline)
	}
	if expectedFastDeadlineMaxBound <= expectedFastDeadline {
		t.Errorf("expectedFastDeadlineMaxBound (%v) should be > expectedFastDeadline (%v)",
			expectedFastDeadlineMaxBound, expectedFastDeadline)
	}

	// Slow deadline: 4s probe delay within 15s deadline. Lower bound < 4s, upper bound > 4s, upper bound < 15s.
	if expectedSlowDeadlineMinBound >= expectedSlowProbeDelay {
		t.Errorf("expectedSlowDeadlineMinBound (%v) should be < expectedSlowProbeDelay (%v)",
			expectedSlowDeadlineMinBound, expectedSlowProbeDelay)
	}
	if expectedSlowDeadlineMaxBound <= expectedSlowProbeDelay {
		t.Errorf("expectedSlowDeadlineMaxBound (%v) should be > expectedSlowProbeDelay (%v)",
			expectedSlowDeadlineMaxBound, expectedSlowProbeDelay)
	}
	if expectedSlowDeadlineMaxBound >= expectedSlowDeadline {
		t.Errorf("expectedSlowDeadlineMaxBound (%v) should be < expectedSlowDeadline (%v)",
			expectedSlowDeadlineMaxBound, expectedSlowDeadline)
	}

	// Stream idle timeout: 2s configured timeout. Lower bound < 2s, upper bound > 2s.
	if expectedStreamIdleMinBound >= expectedStreamIdleTimeout {
		t.Errorf("expectedStreamIdleMinBound (%v) should be < expectedStreamIdleTimeout (%v)",
			expectedStreamIdleMinBound, expectedStreamIdleTimeout)
	}
	if expectedStreamIdleMaxBound <= expectedStreamIdleTimeout {
		t.Errorf("expectedStreamIdleMaxBound (%v) should be > expectedStreamIdleTimeout (%v)",
			expectedStreamIdleMaxBound, expectedStreamIdleTimeout)
	}

	// Connect timeout: 1s configured timeout. Lower bound < 1s, upper bound > 1s.
	if expectedConnectMinBound >= expectedConnectTimeout {
		t.Errorf("expectedConnectMinBound (%v) should be < expectedConnectTimeout (%v)",
			expectedConnectMinBound, expectedConnectTimeout)
	}
	if expectedConnectMaxBound <= expectedConnectTimeout {
		t.Errorf("expectedConnectMaxBound (%v) should be > expectedConnectTimeout (%v)",
			expectedConnectMaxBound, expectedConnectTimeout)
	}
}

func TestParseRequestErrorCount(t *testing.T) {
	t.Parallel()

	metricsExposition := `# HELP llm_request_errors_total Total number of request errors
# TYPE llm_request_errors_total counter
llm_request_errors_total{model="timeout-probe-fast",reason="timeout"} 1
llm_request_errors_total{model="timeout-probe-slow",reason="timeout"} 0
llm_request_errors_total{model="timeout-probe-unreachable",reason="timeout"} 1
llm_request_errors_total{model="other-model",reason="invalid_request"} 5
`

	// Match exact model and reason
	fastCount := parseRequestErrorCount(metricsExposition, "timeout-probe-fast", "timeout")
	if fastCount != 1 {
		t.Errorf("expected fast timeout count 1, got %v", fastCount)
	}

	slowCount := parseRequestErrorCount(metricsExposition, "timeout-probe-slow", "timeout")
	if slowCount != 0 {
		t.Errorf("expected slow timeout count 0, got %v", slowCount)
	}

	unreachableTimeout := parseRequestErrorCount(metricsExposition, "timeout-probe-unreachable", "timeout")
	if unreachableTimeout != 1 {
		t.Errorf("expected unreachable timeout count 1, got %v", unreachableTimeout)
	}

	// Model not present or reason not matched returns 0
	nonExistent := parseRequestErrorCount(metricsExposition, "non-existent-model", "timeout")
	if nonExistent != 0 {
		t.Errorf("expected non-existent metric to return 0, got %v", nonExistent)
	}

	// Empty body returns 0
	empty := parseRequestErrorCount("", "timeout-probe-fast", "timeout")
	if empty != 0 {
		t.Errorf("expected empty body to return 0, got %v", empty)
	}
}
