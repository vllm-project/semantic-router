package classification

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestVLLMClientTimeoutHonorsConfig verifies that a configured
// llm_timeout_seconds bounds the HTTP round-trip via http.Client.Timeout.
// Before the fix, newVLLMClient hardcoded 30s and a configured timeout above
// 30s was silently ignored; below 30s the caller's context.WithTimeout
// happened to fire first, masking the bug. This test isolates the client
// timeout by passing context.Background() (no caller deadline) so only
// http.Client.Timeout can interrupt the slow server.
func TestVLLMClientTimeoutHonorsConfig(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(2 * time.Second)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"safe"}}]}`))
	}))
	defer server.Close()

	client := newVLLMClientFromConfig(&config.ExternalModelConfig{
		ModelEndpoint:  config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1},
		TimeoutSeconds: 1,
	})
	client.baseURL = server.URL

	start := time.Now()
	_, err := client.Generate(context.Background(), "classifier", "test", nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a timeout error, got nil")
	}
	// ponytail: loose bound (1.5x the configured timeout) — CI jitter + server
	// shutdown slop. The point is "config-driven timeout fired", not "exactly
	// 1s". The pre-fix behavior would have waited the full 2s server sleep.
	if elapsed > 1500*time.Millisecond {
		t.Fatalf("elapsed = %v, want < 1.5s (configured 1s timeout); client timeout did not fire", elapsed)
	}
	if elapsed < 800*time.Millisecond {
		t.Fatalf("elapsed = %v, want > 0.8s; timed out too early, check test setup", elapsed)
	}
}

// TestVLLMClientTimeoutDefaultsWhenUnset verifies the 30s default still
// applies when llm_timeout_seconds is unset, so existing configs without an
// explicit timeout keep the prior behavior.
func TestVLLMClientTimeoutDefaultsWhenUnset(t *testing.T) {
	client := newVLLMClientFromConfig(&config.ExternalModelConfig{
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1},
	})
	if got := client.httpClient.Timeout; got != 30*time.Second {
		t.Fatalf("default client timeout = %v, want 30s", got)
	}
}

// TestGetTimeoutCoversCallers is the lazy self-check: ExternalModelConfig.GetTimeout
// resolves the config-driven timeout used by newVLLMClientFromConfig (the client
// timeout) and by the vLLM jailbreak and external preference callers. The
// generic LLM label caller keeps its own 5s default for the caller-side
// context.WithTimeout, but still gets the config-driven client timeout via
// newVLLMClientFromConfig, so a configured llm_timeout_seconds is honored on
// both chains. This guards against a future caller reintroducing a hardcoded
// client default that disagrees with config.
func TestGetTimeoutCoversCallers(t *testing.T) {
	cases := []struct {
		name    string
		seconds int
		want    time.Duration
	}{
		{"unset_uses_default", 0, 30 * time.Second},
		{"configured_passes_through", 60, 60 * time.Second},
		{"negative_uses_default", -1, 30 * time.Second},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &config.ExternalModelConfig{TimeoutSeconds: tc.seconds}
			if got := cfg.GetTimeout(); got != tc.want {
				t.Fatalf("GetTimeout() = %v, want %v", got, tc.want)
			}
		})
	}
}
