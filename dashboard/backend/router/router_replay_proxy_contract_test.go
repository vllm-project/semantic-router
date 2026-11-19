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

	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: routerServer.URL},
		envoyProxy,
		nil, nil,
		replayProxyCredentialProvider{token: "router-service-token"})
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
				"/api/router/api/v1/observability/replays/replay-1?source=dashboard",
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
		if upstreamPaths[index] != "/api/v1/observability/replays/replay-1?source=dashboard" {
			t.Fatalf("management path %d = %q", index, upstreamPaths[index])
		}
		if upstreamAuthorizations[index] != "Bearer router-service-token" {
			t.Fatalf("management Authorization %d = %q", index, upstreamAuthorizations[index])
		}
	}
}

// The records list reaches the Router with its harness filter as the browser
// sent it, and a read-only user's redacted list keeps the harness counts.
func TestDashboardReplayListKeepsHarnessFilterAndCounts(t *testing.T) {
	var upstream string
	routerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstream = r.URL.RequestURI()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"router_replay.list","total":1,"harness_counts":{"api_client":2,"codex":1},` +
			`"data":[{"id":"replay-1","request_body":"private prompt","agent":{"harness":"codex","id":"agent-hash"}}]}`))
	}))
	defer routerServer.Close()
	mux := http.NewServeMux()
	registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: routerServer.URL}, nil, nil, nil,
		replayProxyCredentialProvider{token: "router-service-token"})

	const target = "/api/v1/observability/replays?harness=api_client%2Ccodex&limit=25"
	request := httptest.NewRequest(http.MethodGet, "/api/router"+target, nil)
	request = request.WithContext(auth.WithAuthContext(request.Context(),
		auth.AuthContext{Role: auth.RoleRead, Perms: map[string]bool{auth.PermReplayRead: true}}))
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)

	if response.Code != http.StatusOK || upstream != target {
		t.Fatalf("status = %d, management target = %q, body=%s", response.Code, upstream, response.Body.String())
	}
	var page struct {
		HarnessCounts map[string]int `json:"harness_counts"`
		Data          []struct {
			RequestBody string            `json:"request_body"`
			Agent       map[string]string `json:"agent"`
		} `json:"data"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if page.HarnessCounts["api_client"] != 2 || page.HarnessCounts["codex"] != 1 || len(page.HarnessCounts) != 2 {
		t.Fatalf("harness counts = %v", page.HarnessCounts)
	}
	if len(page.Data) != 1 || page.Data[0].RequestBody != "" || page.Data[0].Agent["harness"] != "codex" || page.Data[0].Agent["id"] != "" {
		t.Fatalf("redacted record = %+v", page.Data)
	}
}
