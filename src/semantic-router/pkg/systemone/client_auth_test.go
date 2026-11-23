package systemone

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/clientauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEntrypointsOnlySystemOneGrantAndDiscovery(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{EntrypointsOnly: true},
		Entrypoints: []config.EntrypointMapping{
			{API: config.ChatAPI, ModelNames: []string{"vllm-sr/test"}, Recipe: "chat"},
			{API: config.SystemOneAPI, ModelNames: []string{"native/public"}, Recipe: "native"},
		},
	}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"worker": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/test", Model: "private-served"}}
	listener := &config.Listener{APIs: config.ListenerAPIs{"systemone": {Schema: "vllm-sr/4.0", Models: []string{"vllm-sr/test", "native/public"}}}}
	policy, problems := clientauth.Compile(cfg.Entrypoints, true, os.LookupEnv)
	require.Empty(t, problems)
	var calls []string
	handler := Handler(cfg, listener, func(_ context.Context, target string, _ json.RawMessage) (int, []byte, error) {
		calls = append(calls, target)
		return http.StatusOK, []byte(certainResponse), nil
	}, ClientAuth{Served: policy})
	for _, path := range []string{"/v1/systemone/models", "/v1/models"} {
		w := httptest.NewRecorder()
		handler(w, httptest.NewRequest(http.MethodGet, path, nil))
		require.Equal(t, http.StatusOK, w.Code)
		require.Contains(t, w.Body.String(), `"native/public"`)
		require.NotContains(t, w.Body.String(), `"vllm-sr/test"`)
	}
	for _, tc := range []struct {
		model  string
		status int
	}{{"vllm-sr/test", http.StatusForbidden}, {"native/public", http.StatusOK}} {
		w := httptest.NewRecorder()
		handler(w, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(strings.Replace(singleRequest, "vllm-sr/auto", tc.model, 1))))
		require.Equal(t, tc.status, w.Code, "%s", w.Body.String())
	}
	require.Equal(t, []string{"native/public"}, calls)
}

func nativePolicy(t *testing.T, envs ...string) *clientauth.Policy {
	t.Helper()
	entrypoint := config.EntrypointMapping{API: config.SystemOneAPI, ModelNames: []string{"vllm-sr/auto"}, Recipe: "native"}
	if len(envs) > 0 {
		entrypoint.ClientAuth = &config.EntrypointClientAuth{Type: config.ClientAuthBearerAPIKey, APIKeyEnvs: envs}
	}
	policy, problems := clientauth.Compile([]config.EntrypointMapping{entrypoint}, false, func(name string) (string, bool) {
		return map[string]string{"NATIVE_KEY": "native-1", "NATIVE_KEY_NEXT": "native-2"}[name], true
	})
	if problems != nil {
		t.Fatal(problems)
	}
	return policy
}

// A System One entrypoint with client keys admits a request only with one of
// them, as a Bearer token, before its recipe or any model runs, and refuses a
// request whose entrypoint a reload protected differently while it arrived.
func TestANativeEntrypointAdmitsOnlyItsKey(t *testing.T) {
	cfg := &config.RouterConfig{Entrypoints: []config.EntrypointMapping{{API: config.SystemOneAPI, ModelNames: []string{"vllm-sr/auto"}, Recipe: "native"}}}
	listener := &config.Listener{APIs: config.ListenerAPIs{"systemone": {Schema: "vllm-sr/4.0", Models: []string{"vllm-sr/auto"}}}}
	protected := nativePolicy(t, "NATIVE_KEY")
	for _, tc := range []struct {
		name    string
		clients ClientAuth
		header  map[string]string
		status  int
		code    string
	}{
		{"no key", ClientAuth{Served: protected}, nil, http.StatusUnauthorized, "invalid_api_key"},
		{"another key", ClientAuth{Served: protected}, map[string]string{"Authorization": "Bearer native-2"}, http.StatusUnauthorized, "invalid_api_key"},
		{"the key as Api-Key", ClientAuth{Served: protected}, map[string]string{"Api-Key": "native-1"}, http.StatusUnauthorized, "invalid_api_key"},
		{"the key", ClientAuth{Served: protected}, map[string]string{"Authorization": "Bearer native-1"}, http.StatusOK, ""},
		{"a public entrypoint", ClientAuth{Served: nativePolicy(t)}, nil, http.StatusOK, ""},
		{
			"protected while arriving",
			ClientAuth{Served: nativePolicy(t), Active: func() *clientauth.Policy { return protected }},
			map[string]string{"Authorization": "Bearer native-1"},
			http.StatusConflict, "configuration_changed",
		},
		{
			"keys rotated while arriving",
			ClientAuth{Served: protected, Active: func() *clientauth.Policy { return nativePolicy(t, "NATIVE_KEY_NEXT") }},
			map[string]string{"Authorization": "Bearer native-1"},
			http.StatusConflict, "configuration_changed",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			invoke := func(context.Context, string, json.RawMessage) (int, []byte, error) {
				calls++
				return http.StatusOK, []byte(certainResponse), nil
			}
			request := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(singleRequest))
			for name, value := range tc.header {
				request.Header.Set(name, value)
			}
			response := httptest.NewRecorder()
			Handler(cfg, listener, invoke, tc.clients)(response, request)
			if response.Code != tc.status || !strings.Contains(response.Body.String(), tc.code) {
				t.Fatalf("status = %d, body = %s; want %d %s", response.Code, response.Body.String(), tc.status, tc.code)
			}
			if (calls != 0) != (tc.status == http.StatusOK) {
				t.Fatalf("%d calls for status %d", calls, response.Code)
			}
			if tc.status == http.StatusUnauthorized && response.Header().Get("WWW-Authenticate") == "" {
				t.Fatal("a refusal for a missing key carries no challenge")
			}
		})
	}
}
