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
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	servedWindowEvent                  = "served_context_window_below_model_card"
	servedContextWindowNotServedEvent  = "served_context_window_model_not_served"
	servedContextWindowUnverifiedEvent = "served_context_window_unverified"
)

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
		name          string
		servedID      string
		served        *int
		statuses      []int
		cardOverride  int
		wantCalls     int32
		wantWindow    int
		wantSource    string
		wantNotServed bool
		wantUnread    string
	}{
		{name: "built-in card above served", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), wantCalls: 1, wantWindow: 262_144, wantSource: "builtin"},
		{name: "operator override above served", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), cardOverride: 65_536, wantCalls: 1, wantWindow: 65_536, wantSource: "operator"},
		{name: "retried until the backend loads", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), statuses: []int{http.StatusServiceUnavailable, http.StatusBadGateway}, wantCalls: 3, wantWindow: 262_144, wantSource: "builtin"},
		{name: "aligned override", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), cardOverride: 32_768, wantCalls: 1},
		{name: "served above card", servedID: "Qwen/Qwen3.6-27B", served: intPtr(1_048_576), wantCalls: 1},
		{name: "served length unset", servedID: "Qwen/Qwen3.6-27B", wantCalls: 1, wantUnread: "reports no max_model_len"},
		{name: "model not listed", servedID: "Qwen/Qwen3.6-35B-A3B", served: intPtr(32_768), wantCalls: 1, wantNotServed: true},
		{name: "unauthorized is not retried", servedID: "Qwen/Qwen3.6-27B", served: intPtr(32_768), statuses: []int{http.StatusUnauthorized}, wantCalls: 1, wantUnread: "HTTP status 401"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			core, logs := observer.New(zapcore.DebugLevel)
			t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
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
			unread := logs.FilterMessage(servedContextWindowUnverifiedEvent).All()
			if tt.wantUnread != "" {
				require.Len(t, unread, 1)
				fields := unread[0].ContextMap()
				assert.Equal(t, "qwen-long", fields["model"])
				assert.Contains(t, fields["error"], tt.wantUnread)
			} else {
				assert.Empty(t, unread)
			}
			notServed := logs.FilterMessage(servedContextWindowNotServedEvent).All()
			if tt.wantNotServed {
				require.Len(t, notServed, 1)
				fields := notServed[0].ContextMap()
				assert.Equal(t, "Qwen/Qwen3.6-27B", fields["upstream_model"])
				assert.Equal(t, "qwen-long", fields["model"])
				assert.Contains(t, fields["message"], "--served-model-name Qwen/Qwen3.6-27B")
			} else {
				assert.Empty(t, notServed)
			}
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

// An in-scope backend that cannot be resolved still has to leave a trace,
// otherwise "never probed" and "probed and aligned" read the same in the log.
func TestServedContextWindowRecordsAnUnresolvableVLLMBackend(t *testing.T) {
	core, logs := observer.New(zapcore.DebugLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))

	cfg := servedWindowConfig(t, "http://127.0.0.1:8000", 0)
	endpoint := cfg.VLLMEndpoints[0].Name
	for name, profile := range cfg.ProviderProfiles {
		profile.BaseURL = "://not a url"
		cfg.ProviderProfiles[name] = profile
	}

	assert.Empty(t, servedContextWindowTargets(cfg))

	events := logs.FilterMessage(servedContextWindowUnverifiedEvent).All()
	require.Len(t, events, 1)
	fields := events[0].ContextMap()
	assert.Equal(t, "qwen-long", fields["model"])
	assert.Equal(t, endpoint, fields["endpoint"])
	assert.Contains(t, fields["error"], "base URL is not parseable")
}

// A backend that keeps answering "still loading" must not be mistaken for one
// the Router gave up on, so there is no attempt ceiling to reach.
func TestServedContextWindowKeepsProbingARetryableBackendPastAnyCeiling(t *testing.T) {
	logs := newObservedEventLogger(t)
	interval := servedContextWindowRetryInterval
	servedContextWindowRetryInterval = time.Millisecond
	t.Cleanup(func() { servedContextWindowRetryInterval = interval })

	var calls atomic.Int32
	backend := httptest.NewServer(vllmModelsHandler(t, &calls, nil, "Qwen/Qwen3.6-27B", intPtr(32_768)))
	t.Cleanup(backend.Close)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	// Fail every attempt for a while, then let the read through. The old
	// ceiling gave up after twenty tries and logged nothing.
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) < 40 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"id":"Qwen/Qwen3.6-27B","max_model_len":32768}]}`))
	}))
	t.Cleanup(failing.Close)

	cfg := servedWindowConfig(t, failing.URL, 0)
	targets := servedContextWindowTargets(cfg)
	require.Len(t, targets, 1)

	go (&OpenAIRouter{Config: cfg}).checkServedContextWindow(ctx, targets[0])

	require.Eventually(t, func() bool {
		return len(logs.FilterMessage(servedWindowEvent).All()) == 1
	}, 10*time.Second, 5*time.Millisecond, "the probe gave up before the backend answered")
	assert.Empty(t, logs.FilterMessage(servedContextWindowUnverifiedEvent).All())
}

func TestPublishRouterStateChecksServedContextWindowUntilClose(t *testing.T) {
	restoreProcessGlobals(t)
	logs := newObservedEventLogger(t)
	interval := servedContextWindowRetryInterval
	servedContextWindowRetryInterval = 5 * time.Millisecond
	t.Cleanup(func() { servedContextWindowRetryInterval = interval })

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
