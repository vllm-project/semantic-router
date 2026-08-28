package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

type replayProxyCredentialProvider struct {
	token string
}

func (provider replayProxyCredentialProvider) ManagementCredential() (string, error) {
	return provider.token, nil
}

func TestDashboardReplayPathUsesAuthenticatedManagementProxyNotEnvoy(t *testing.T) {
	var envoyCalls int
	envoyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		envoyCalls++
		w.WriteHeader(http.StatusTeapot)
	}))
	defer envoyServer.Close()
	envoyTarget, err := url.Parse(envoyServer.URL)
	if err != nil {
		t.Fatalf("parse Envoy URL: %v", err)
	}
	envoyProxy := httputil.NewSingleHostReverseProxy(envoyTarget)

	var routerCalls int
	var upstreamPaths []string
	var upstreamAuthorizations []string
	routerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		routerCalls++
		upstreamPaths = append(upstreamPaths, r.URL.RequestURI())
		upstreamAuthorizations = append(upstreamAuthorizations, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"replay-1","request_body":"private prompt"}`))
	}))
	defer routerServer.Close()

	mux := auth.NewPolicyMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: routerServer.URL},
		envoyProxy,
		replayProxyCredentialProvider{token: "router-service-token"},
	)
	tests := []struct {
		name       string
		context    auth.AuthContext
		wantPrompt string
	}{
		{
			name:       "read-only user receives redacted detail",
			context:    auth.AuthContext{Role: auth.RoleRead, Perms: map[string]bool{auth.PermReplayRead: true}},
			wantPrompt: "",
		},
		{
			name:       "write user receives full detail",
			context:    auth.AuthContext{Role: auth.RoleWrite, Perms: map[string]bool{auth.PermReplayRead: true, auth.PermConfigWrite: true}},
			wantPrompt: "private prompt",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(
				http.MethodGet,
				"/api/router/v1/router_replay/replay-1?source=dashboard",
				nil,
			)
			request.Header.Set("Authorization", "Bearer browser-user-token")
			request = request.WithContext(auth.WithAuthContext(request.Context(), test.context))
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, request)

			if response.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
			var payload map[string]interface{}
			if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if payload["request_body"] != test.wantPrompt {
				t.Fatalf("request_body = %#v, want %q", payload["request_body"], test.wantPrompt)
			}
		})
	}

	if envoyCalls != 0 || routerCalls != len(tests) {
		t.Fatalf("proxy calls: envoy=%d management=%d", envoyCalls, routerCalls)
	}
	for index := range upstreamPaths {
		if upstreamPaths[index] != "/v1/router_replay/replay-1?source=dashboard" {
			t.Fatalf("management path %d = %q", index, upstreamPaths[index])
		}
		if upstreamAuthorizations[index] != "Bearer router-service-token" {
			t.Fatalf("management Authorization %d = %q", index, upstreamAuthorizations[index])
		}
	}
}
