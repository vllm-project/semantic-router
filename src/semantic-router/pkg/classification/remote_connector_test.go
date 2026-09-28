package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func endpointForTestServer(t *testing.T, server *httptest.Server) config.ClassifierVLLMEndpoint {
	t.Helper()
	parsed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse test server URL: %v", err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatalf("parse test server port: %v", err)
	}
	return config.ClassifierVLLMEndpoint{
		Protocol: parsed.Scheme,
		Address:  parsed.Hostname(),
		Port:     port,
	}
}

func TestHTTPClassifierRetriesTransientStatus(t *testing.T) {
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if attempts.Add(1) == 1 {
			http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
			return
		}
		_ = json.NewEncoder(w).Encode([]httpClassifyLabelScore{
			{Label: "safe", Score: 0.9},
			{Label: "jailbreak", Score: 0.1},
		})
	}))
	defer server.Close()

	inference := newTestHTTPClassifierInference(t, server, testJailbreakMapping())
	if _, err := inference.Classify(context.Background(), "hello"); err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("attempts = %d, want 2", got)
	}
}
