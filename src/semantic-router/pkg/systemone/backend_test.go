package systemone

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

func TestNativeRemoteAdapterPreservesTaskAndUsesProviderAuth(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/custom/v1/systemone" || r.Header.Get("Authorization") != "Bearer private-test-key" {
			t.Errorf("wrong native path or configured credential")
		}
		if r.Header.Get(BackendRequestHeader) != "1" {
			t.Error("native backend request lost its recursion guard")
		}
		var body map[string]json.RawMessage
		if json.NewDecoder(r.Body).Decode(&body) != nil || string(body["model"]) != `"native-model"` || string(body["state"]) != `"hello"` {
			t.Error("native envelope incorrectly rewritten")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(certainResponse))
	}))
	defer server.Close()
	t.Setenv("SYSTEMONE_BACKEND_TEST_KEY", "private-test-key")
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.3
routing: {}
providers:
  models:
    - name: native
      api_format: systemone
      provider_model_id: native-model
      backend_refs:
        - provider: systemone-compatible
          base_url: %s/custom/v1
          api_key_env: SYSTEMONE_BACKEND_TEST_KEY
`, server.URL)))
	if err != nil {
		t.Fatal(err)
	}
	remote, err := upstream.Build(cfg, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer remote.Close(context.Background())
	invoke := BackendInvoker(cfg, "", nil, remote)
	code, body, err := invoke(context.Background(), "native", json.RawMessage(singleRequest))
	if err != nil || code != 200 || string(body) != certainResponse {
		t.Fatalf("status=%d error=%v", code, err)
	}
	_, _, err = invoke(context.Background(), "unknown", json.RawMessage(singleRequest))
	if err == nil || calls.Load() != 1 {
		t.Fatal("unknown native alias reached a backend")
	}
	if remote.Topology().DefaultCluster != "" {
		t.Fatal("native backend became default Chat route")
	}
}

func TestForwardedNativeActionCannotEnterAnotherRoutingChain(t *testing.T) {
	cfg := &config.RouterConfig{
		Entrypoints: []config.EntrypointMapping{{API: config.SystemOneAPI, ModelNames: []string{"auto"}, Recipe: "native"}},
	}
	cfg.ModelConfig = map[string]config.ModelParams{
		"remote": {APIFormat: config.APIFormatSystemOne},
		"local":  {APIFormat: config.APIFormatSystemOne, Deployment: "primary"},
	}
	listener := &config.Listener{APIKeys: []string{"key"}, SystemOne: &config.ListenerSystemOne{Models: []string{"auto", "remote", "local"}}}
	for _, tc := range []struct {
		model, key string
		status     int
	}{
		{"auto", "key", http.StatusConflict},
		{"remote", "key", http.StatusConflict},
		{"local", "key", http.StatusOK},
		{"local", "", http.StatusUnauthorized},
	} {
		t.Run(tc.model+tc.key, func(t *testing.T) {
			calls := 0
			invoke := func(context.Context, string, json.RawMessage) (int, []byte, error) {
				calls++
				return http.StatusOK, []byte(certainResponse), nil
			}
			r := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"`+tc.model+`"}`))
			r.Header.Set(BackendRequestHeader, "1")
			r.Header.Set("Api-Key", tc.key)
			w := httptest.NewRecorder()
			Handler(cfg, listener, invoke)(w, r)
			if w.Code != tc.status || (calls != 0) != (tc.status == http.StatusOK) {
				t.Fatalf("status=%d calls=%d body=%s", w.Code, calls, w.Body.String())
			}
		})
	}
}

func TestPublicNativeEntrypointNeedsItsOwnListenerGrant(t *testing.T) {
	cfg := &config.RouterConfig{Entrypoints: []config.EntrypointMapping{{API: config.SystemOneAPI, ModelNames: []string{"vllm-sr/auto"}, Recipe: "native"}}}
	listener := &config.Listener{APIKeys: []string{"key"}, Models: []string{"vllm-sr/auto"}, SystemOne: &config.ListenerSystemOne{Models: []string{"another"}}}
	calls := 0
	invoke := func(_ context.Context, target string, _ json.RawMessage) (int, []byte, error) {
		calls++
		if target != "vllm-sr/auto" {
			t.Fatal("auto converted to a deployment")
		}
		return 200, []byte(certainResponse), nil
	}
	send := func() int {
		r := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(singleRequest))
		r.Header.Set("Authorization", "Bearer key")
		w := httptest.NewRecorder()
		Handler(cfg, listener, invoke)(w, r)
		return w.Code
	}
	if send() != http.StatusForbidden || calls != 0 {
		t.Fatal("Chat grant authorized native auto")
	}
	listener.SystemOne.Models = []string{"vllm-sr/auto"}
	if send() != http.StatusOK || calls != 1 {
		t.Fatal("authorized native entrypoint did not dispatch")
	}
	r := httptest.NewRequest(http.MethodGet, "/v1/systemone/models", nil)
	r.Header.Set("Authorization", "Bearer key")
	w := httptest.NewRecorder()
	Handler(cfg, listener, invoke)(w, r)
	if !strings.Contains(w.Body.String(), `"routing":true`) {
		t.Fatal("virtual discovery lost native routing identity")
	}
}

func TestPublicAutoErrorsDistinguishUnresolvedFromDeadline(t *testing.T) {
	cfg := &config.RouterConfig{Entrypoints: []config.EntrypointMapping{{API: config.SystemOneAPI, ModelNames: []string{"vllm-sr/auto"}, Recipe: "native"}}}
	listener := &config.Listener{SystemOne: &config.ListenerSystemOne{Models: []string{"vllm-sr/auto"}}}
	for _, tc := range []struct {
		err    error
		status int
		code   string
	}{
		{ErrUnresolved, 503, "systemone_unresolved"},
		{context.DeadlineExceeded, 504, "systemone_deadline_exceeded"},
	} {
		t.Run(tc.code, func(t *testing.T) {
			invoke := func(context.Context, string, json.RawMessage) (int, []byte, error) {
				return tc.status, nil, fmt.Errorf("internal diagnostic: %w", tc.err)
			}
			request := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(singleRequest))
			response := httptest.NewRecorder()
			Handler(cfg, listener, invoke)(response, request)
			if response.Code != tc.status || !strings.Contains(response.Body.String(), tc.code) || strings.Contains(response.Body.String(), "internal diagnostic") {
				t.Fatalf("unexpected public error: %d %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestEngineModeDoesNotPublishPreservedRoutingRecipes(t *testing.T) {
	disabled := false
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{RouterEnabled: &disabled},
		Entrypoints:   []config.EntrypointMapping{{API: config.SystemOneAPI, ModelNames: []string{"auto"}, Recipe: "native"}},
	}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/test"}}
	listener := &config.Listener{SystemOne: &config.ListenerSystemOne{Models: []string{"auto", "vllm-sr/test"}}}
	invoked := false
	handler := Handler(cfg, listener, func(context.Context, string, json.RawMessage) (int, []byte, error) {
		invoked = true
		return http.StatusOK, []byte(certainResponse), nil
	})
	w := httptest.NewRecorder()
	handler(w, httptest.NewRequest(http.MethodGet, "/v1/systemone/models", nil))
	if w.Code != http.StatusOK || strings.Contains(w.Body.String(), `"auto"`) || !strings.Contains(w.Body.String(), "vllm-sr/test") {
		t.Fatalf("Engine discovery: %d %s", w.Code, w.Body.String())
	}
	w = httptest.NewRecorder()
	handler(w, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"auto"}`)))
	if w.Code != http.StatusNotFound || invoked || !strings.Contains(w.Body.String(), "systemone_routing_disabled") {
		t.Fatal("Engine mode dispatched a saved routing plan")
	}
}
