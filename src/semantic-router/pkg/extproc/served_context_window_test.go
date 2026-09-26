package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const servedWindowEvent = "served_context_window_below_model_card"

// vllmModelsHandler answers like vLLM's /v1/models: base models carry the
// engine's max_model_len, which is null when unset.
func vllmModelsHandler(t *testing.T, calls *atomic.Int32, statuses []int, id string, maxModelLen *int) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		call := int(calls.Add(1))
		assert.Equal(t, http.MethodGet, req.Method)
		assert.Equal(t, "/v1/models", req.URL.Path)
		assert.Equal(t, "Bearer backend-secret", req.Header.Get("Authorization"))
		if call <= len(statuses) {
			w.WriteHeader(statuses[call-1])
			return
		}
		card := map[string]any{"id": id, "object": "model", "owned_by": "vllm", "root": id, "parent": nil, "max_model_len": maxModelLen, "permission": []any{}}
		assert.NoError(t, json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{card}}))
	}
}

func servedWindowConfig(t *testing.T, backendURL string, cardOverride int) *config.RouterConfig {
	t.Helper()
	backend, err := url.Parse(backendURL)
	require.NoError(t, err)
	override := "routing: {}\n"
	if cardOverride > 0 {
		override = fmt.Sprintf("routing:\n  modelCards:\n    - name: qwen/qwen3.6-27b\n      context_window_size: %d\n", cardOverride)
	}
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.3
listeners: []
providers:
  defaults:
    model: qwen-long
  models:
    - name: qwen-long
      catalog: qwen/qwen3.6-27b
      backend_refs:
        - name: qwen-vllm
          provider: vllm
          endpoint: %s
          protocol: http
          api_key: backend-secret
%sglobal: {}
`, backend.Host, override)))
	require.NoError(t, err)
	return cfg
}

func TestServedContextWindowWarnsOnlyWhenTheBackendServesLess(t *testing.T) {
	for _, tt := range []struct {
		name         string
		servedID     string
		served       *int
		statuses     []int
		cardOverride int
		wantCalls    int32
		wantWindow   int
		wantSource   string
	}{
		{name: "built-in card above served", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), wantCalls: 1, wantWindow: 262_144, wantSource: "builtin"},
		{name: "operator override above served", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), cardOverride: 65_536, wantCalls: 1, wantWindow: 65_536, wantSource: "operator"},
		{name: "retried until the backend loads", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), statuses: []int{http.StatusServiceUnavailable, http.StatusBadGateway}, wantCalls: 3, wantWindow: 262_144, wantSource: "builtin"},
		{name: "aligned override", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), cardOverride: 32_768, wantCalls: 1},
		{name: "served above card", servedID: "Qwen/Qwen3.6-27B", served: intPtr(1_048_576), wantCalls: 1},
		{name: "served length unset", servedID: "Qwen/Qwen3.6-27B", wantCalls: 1},
		{name: "model not listed", servedID: "Qwen/Qwen3.6-35B-A3B", served: intPtr(32_768), wantCalls: 1},
		{name: "unauthorized is not retried", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), statuses: []int{http.StatusUnauthorized}, wantCalls: 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			logs := newObservedEventLogger(t)
			interval := servedContextWindowRetryInterval
			servedContextWindowRetryInterval = time.Millisecond
			t.Cleanup(func() { servedContextWindowRetryInterval = interval })
			var calls atomic.Int32
			backend := httptest.NewServer(vllmModelsHandler(t, &calls, tt.statuses, tt.servedID, tt.served))
			t.Cleanup(backend.Close)
			cfg := servedWindowConfig(t, backend.URL, tt.cardOverride)
			targets := servedContextWindowTargets(cfg)
			require.Len(t, targets, 1)

			(&OpenAIRouter{Config: cfg}).checkServedContextWindow(context.Background(), targets[0])

			assert.Equal(t, tt.wantCalls, calls.Load())
			warnings := logs.FilterMessage(servedWindowEvent).All()
			if tt.wantWindow == 0 {
				assert.Empty(t, warnings)
				return
			}
			require.Len(t, warnings, 1)
			fields := warnings[0].ContextMap()
			assert.Equal(t, "qwen-long", fields["model"])
			assert.Equal(t, "qwen/qwen3.6-27b", fields["model_card"])
			assert.Equal(t, "Qwen/Qwen3.6-27B", fields["upstream_model"])
			assert.EqualValues(t, tt.wantWindow, fields["context_window_size"])
			assert.Equal(t, tt.wantSource, fields["context_window_source"])
			assert.EqualValues(t, 32_768, fields["served_max_model_len"])
			assert.Contains(t, fields["message"], `context_window_size: 32768 on the routing.modelCards entry named "qwen/qwen3.6-27b"`)
		})
	}
}

func TestServedContextWindowTargetsOnlyVLLMBackendsWithKnownWindows(t *testing.T) {
	cfg := servedWindowConfig(t, "http://127.0.0.1:8000", 0)
	require.Len(t, servedContextWindowTargets(cfg), 1)

	params := cfg.ModelConfig["qwen-long"]
	params.ContextWindowSize = 0
	cfg.ModelConfig["qwen-long"] = params
	assert.Empty(t, servedContextWindowTargets(cfg))

	cfg = servedWindowConfig(t, "http://127.0.0.1:8000", 0)
	for name, profile := range cfg.ProviderProfiles {
		profile.Type = "sglang"
		cfg.ProviderProfiles[name] = profile
	}
	assert.Empty(t, servedContextWindowTargets(cfg))
}

func TestPublishRouterStateChecksServedContextWindowUntilClose(t *testing.T) {
	restoreProcessGlobals(t)
	logs := newObservedEventLogger(t)
	interval, attempts := servedContextWindowRetryInterval, servedContextWindowAttempts
	servedContextWindowRetryInterval, servedContextWindowAttempts = 5*time.Millisecond, 1_000
	t.Cleanup(func() { servedContextWindowRetryInterval, servedContextWindowAttempts = interval, attempts })

	var served atomic.Int32
	servedBackend := httptest.NewServer(vllmModelsHandler(t, &served, nil, "Qwen/Qwen3.6-27B", intPtr(32_768)))
	t.Cleanup(servedBackend.Close)
	cfg := servedWindowConfig(t, servedBackend.URL, 0)
	router := &OpenAIRouter{Config: cfg, resources: newResourceScope()}
	publishRouterState(cfg, router, nil, nil)
	require.Eventually(t, func() bool { return len(logs.FilterMessage(servedWindowEvent).All()) == 1 }, 5*time.Second, 5*time.Millisecond)
	require.NoError(t, router.Close())

	var loading atomic.Int32
	loadingStatuses := make([]int, 1_000)
	for i := range loadingStatuses {
		loadingStatuses[i] = http.StatusServiceUnavailable
	}
	loadingBackend := httptest.NewServer(vllmModelsHandler(t, &loading, loadingStatuses, "Qwen/Qwen3.6-27B", intPtr(32_768)))
	t.Cleanup(loadingBackend.Close)
	cfg = servedWindowConfig(t, loadingBackend.URL, 0)
	router = &OpenAIRouter{Config: cfg, resources: newResourceScope()}
	publishRouterState(cfg, router, nil, nil)
	require.Eventually(t, func() bool { return loading.Load() >= 2 }, 5*time.Second, 5*time.Millisecond)
	require.NoError(t, router.Close())
	time.Sleep(20 * time.Millisecond)
	afterClose := loading.Load()
	time.Sleep(100 * time.Millisecond)
	assert.Equal(t, afterClose, loading.Load(), "a retired generation kept probing its backend")
}
