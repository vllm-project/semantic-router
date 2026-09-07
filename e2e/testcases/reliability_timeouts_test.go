package testcases

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestEvaluateTimeoutBound(t *testing.T) {
	t.Parallel()

	// Normal case: completed well within timeout + slack
	if err := evaluateTimeoutBound("test fast", 300*time.Millisecond, 500*time.Millisecond, 1*time.Second); err != nil {
		t.Fatalf("expected nil for within bound, got %v", err)
	}

	// Boundary case: completed exactly at timeout + slack
	if err := evaluateTimeoutBound("test boundary", 1*time.Second, 500*time.Millisecond, 500*time.Millisecond); err != nil {
		t.Fatalf("expected nil for exact boundary, got %v", err)
	}

	// Exceeded case: completed past timeout + slack
	if err := evaluateTimeoutBound("test exceeded", 2*time.Second, 500*time.Millisecond, 500*time.Millisecond); err == nil {
		t.Fatalf("expected error for exceeded bound, got nil")
	}
}

func TestIsAllowedProbeStatusCode(t *testing.T) {
	t.Parallel()

	allowed := []int{http.StatusOK, http.StatusRequestTimeout, http.StatusGatewayTimeout}
	for _, code := range allowed {
		if !isAllowedProbeStatusCode(code) {
			t.Errorf("expected status %d to be allowed", code)
		}
	}

	rejected := []int{
		http.StatusBadRequest,
		http.StatusNotFound,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
	}
	for _, code := range rejected {
		if isAllowedProbeStatusCode(code) {
			t.Errorf("expected status %d to be rejected", code)
		}
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
