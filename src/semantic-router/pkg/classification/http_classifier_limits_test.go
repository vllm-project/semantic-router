package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func inputForRequestBytes(t *testing.T, n int) string {
	t.Helper()
	envelope, err := json.Marshal(httpClassifyRequest{Inputs: ""})
	if err != nil {
		t.Fatalf("failed to marshal the empty request envelope: %v", err)
	}
	if n < len(envelope) {
		t.Fatalf("cannot build a %d-byte body: the empty envelope is already %d bytes", n, len(envelope))
	}
	return strings.Repeat("a", n-len(envelope))
}

func newLimitedTestInference(t *testing.T, server *httptest.Server, requestBytes, responseBytes int64) *HTTPClassifierInference {
	t.Helper()
	inf, err := NewHTTPClassifierInference(&config.ExternalModelConfig{
		ModelEndpoint:    endpointForTestServer(t, server),
		ModelName:        "custom-classifier",
		MaxRequestBytes:  requestBytes,
		MaxResponseBytes: responseBytes,
	}, testJailbreakMapping())
	if err != nil {
		t.Fatalf("failed to construct inference: %v", err)
	}
	return inf
}

func TestHTTPClassifierInferenceClassify_RequestAtLimitIsSent(t *testing.T) {
	var received int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received = r.ContentLength
		_ = json.NewEncoder(w).Encode([]httpClassifyLabelScore{
			{Label: "safe", Score: 0.9},
			{Label: "jailbreak", Score: 0.1},
		})
	}))
	defer server.Close()

	const limit = 1024
	inf := newLimitedTestInference(t, server, 1024, 0)
	if _, err := inf.Classify(context.Background(), inputForRequestBytes(t, limit)); err != nil {
		t.Fatalf("unexpected error for a request exactly at the cap: %v", err)
	}
	if received != limit {
		t.Errorf("server received %d bytes, want %d", received, limit)
	}
}

func TestHTTPClassifierInferenceClassify_RequestOneByteOverLimitIsRejected(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("the endpoint was called for an over-limit request body")
	}))
	defer server.Close()

	inf := newLimitedTestInference(t, server, 1024, 0)
	if _, err := inf.Classify(context.Background(), inputForRequestBytes(t, 1025)); err == nil {
		t.Fatal("expected an error for a request body of exactly cap+1 bytes, got nil")
	}
}

func serveBodyOfSize(t *testing.T, n int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(strings.Repeat("x", n))); err != nil {
			t.Errorf("failed to write the test response body: %v", err)
		}
	}))
}

func TestHTTPClassifierInferenceClassify_ResponseOneByteOverLimitIsRejected(t *testing.T) {
	server := serveBodyOfSize(t, 1025)
	defer server.Close()

	inf := newLimitedTestInference(t, server, 0, 1024)
	_, err := inf.Classify(context.Background(), "some text")
	if err == nil {
		t.Fatal("expected an error for a response body of exactly cap+1 bytes, got nil")
	}
	if !strings.Contains(err.Error(), "exceeds limit") {
		t.Errorf("error = %v, want it to report the exceeded read limit", err)
	}
}

func TestHTTPClassifierInferenceClassify_ResponseAtLimitIsParsed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		scores := []map[string]interface{}{
			{"label": "safe", "score": 0.9},
			{"label": "jailbreak", "score": 0.1},
		}
		body, err := json.Marshal(scores)
		if err != nil {
			t.Errorf("failed to marshal the test response: %v", err)
			return
		}
		scores[0]["padding"] = strings.Repeat("p", 1024-len(body)-len(`,"padding":""`))
		body, err = json.Marshal(scores)
		if err != nil {
			t.Errorf("failed to marshal the padded test response: %v", err)
			return
		}
		if len(body) != 1024 {
			t.Errorf("padded response is %d bytes, want exactly 1024", len(body))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write(body); err != nil {
			t.Errorf("failed to write the test response body: %v", err)
		}
	}))
	defer server.Close()

	inf := newLimitedTestInference(t, server, 0, 1024)
	result, err := inf.Classify(context.Background(), "some text")
	if err != nil {
		t.Fatalf("unexpected error for a response exactly at the cap: %v", err)
	}
	if len(result.Probabilities) != 2 {
		t.Errorf("got %d probabilities, want 2", len(result.Probabilities))
	}
}

func TestHTTPClassifierInferenceClassify_ErrorBodyIsTruncatedNotDropped(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		if _, err := w.Write([]byte(strings.Repeat("e", int(maxClassifyErrorBodyBytes)*2))); err != nil {
			t.Errorf("failed to write the test error body: %v", err)
		}
	}))
	defer server.Close()

	inf := newLimitedTestInference(t, server, 0, 0)
	_, err := inf.Classify(context.Background(), "some text")
	if err == nil {
		t.Fatal("expected an error for a 500 response, got nil")
	}
	if !strings.Contains(err.Error(), "status 500") {
		t.Errorf("error = %v, want it to report the endpoint status", err)
	}
	if !strings.Contains(err.Error(), "truncated=true") {
		t.Errorf("error = %v, want it to report that the body was truncated", err)
	}
	if len(err.Error()) > 2*int(maxClassifyErrorBodyBytes) {
		t.Errorf("error message is %d bytes, want the body bounded near %d", len(err.Error()), maxClassifyErrorBodyBytes)
	}
}

func TestHTTPClassifierInferenceClassify_CallerCancellationStopsTheRequest(t *testing.T) {
	released := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-released:
		}
	}))
	defer server.Close()
	defer close(released)

	inf := newLimitedTestInference(t, server, 0, 0)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := inf.Classify(ctx, "some text")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected an error when the caller cancels, got nil")
	}
	if elapsed >= inf.timeout {
		t.Errorf("Classify took %v, want it to return on cancellation well before the %v timeout", elapsed, inf.timeout)
	}
}
