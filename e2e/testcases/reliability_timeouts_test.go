package testcases

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestEvaluateConnectTimeoutBound(t *testing.T) {
	t.Parallel()

	// Normal case: completed within timeout + slack
	if err := evaluateConnectTimeoutBound(300*time.Millisecond, 500*time.Millisecond, 1*time.Second); err != nil {
		t.Fatalf("expected nil for within bound, got %v", err)
	}

	// Exceeded case
	if err := evaluateConnectTimeoutBound(2*time.Second, 500*time.Millisecond, 500*time.Millisecond); err == nil {
		t.Fatalf("expected error for exceeded bound, got nil")
	}
}

func TestEvaluateDistinctDeadlineResult(t *testing.T) {
	t.Parallel()

	// Normal: fastElapsed is within limit + slack
	if err := evaluateDistinctDeadlineResult(800*time.Millisecond, 2*time.Second, 1*time.Second); err != nil {
		t.Fatalf("expected nil for within limit, got %v", err)
	}

	// Fast elapsed exceeds limit + slack
	if err := evaluateDistinctDeadlineResult(2*time.Second, 5*time.Second, 1*time.Second); err == nil {
		t.Fatalf("expected error for fast model exceeding limit, got nil")
	}
}

func TestDistinctDeadlinesWithMockServer(t *testing.T) {
	t.Parallel()

	// Mock server that sleeps 150ms before responding
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(150 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"result":"ok"}`))
	}))
	defer server.Close()

	// Client with short deadline (50ms) must fail
	shortCtx, shortCancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer shortCancel()
	reqShort, _ := http.NewRequestWithContext(shortCtx, http.MethodGet, server.URL, nil)
	_, errShort := http.DefaultClient.Do(reqShort)
	if errShort == nil {
		t.Fatalf("expected short deadline request to fail, but succeeded")
	}

	// Client with longer deadline (500ms) must succeed
	longCtx, longCancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer longCancel()
	reqLong, _ := http.NewRequestWithContext(longCtx, http.MethodGet, server.URL, nil)
	respLong, errLong := http.DefaultClient.Do(reqLong)
	if errLong != nil {
		t.Fatalf("expected long deadline request to succeed, got %v", errLong)
	}
	defer respLong.Body.Close()
	if respLong.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", respLong.StatusCode)
	}
}

func TestStalledStreamWithMockServer(t *testing.T) {
	t.Parallel()

	// Mock server that streams first chunk, then closes or stalls
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintf(w, "data: chunk 1\n\n")
		flusher.Flush()
		// Server stalls / returns
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, server.URL, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("failed initial stream request: %v", err)
	}
	defer resp.Body.Close()

	buf := make([]byte, 1024)
	n, err := resp.Body.Read(buf)
	if err != nil && n == 0 {
		t.Fatalf("expected to read initial chunk: %v", err)
	}
}

func TestShortConnectFailuresWithClosedPort(t *testing.T) {
	t.Parallel()

	d := net.Dialer{Timeout: 100 * time.Millisecond}
	start := time.Now()
	conn, err := d.DialContext(context.Background(), "tcp", "127.0.0.1:65531")
	elapsed := time.Since(start)

	if conn != nil {
		_ = conn.Close()
	}

	if err == nil {
		t.Fatalf("expected dial to fail on closed port")
	}

	if elapsed > 1*time.Second {
		t.Fatalf("connect took %v, want fast failure (< 1s)", elapsed)
	}
}
