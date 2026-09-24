package router

import (
	"bufio"
	"context"
	"database/sql"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
)

type routerProxyCredentialProvider struct {
	token string
}

func TestGrafanaRouteServesAdapterAndRewritesDocument(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/d/router" {
			t.Errorf("unexpected upstream request %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = io.WriteString(w, "<html><head></head><body>Grafana</body></html>")
	}))
	defer upstream.Close()
	mux := http.NewServeMux()
	registerGrafanaRoutes(mux, &config.Config{GrafanaURL: upstream.URL})
	for _, path := range []string{proxy.GrafanaAuthScriptPath, "/embedded/grafana/d/router"} {
		request := httptest.NewRequest(http.MethodGet, path, nil)
		request.Header.Set("Accept", "text/html")
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, request)
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d", path, response.Code)
		}
		if path == proxy.GrafanaAuthScriptPath {
			if !strings.Contains(response.Header().Get("Content-Type"), "javascript") {
				t.Fatal("adapter was not served locally")
			}
		} else if !strings.Contains(response.Body.String(), proxy.GrafanaAuthScriptPath) {
			t.Fatal("document did not install the adapter")
		}
	}
}

func TestRegisterProxyRoutesDoesNotExposeFleetSimAPI(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	registerProxyRoutes(mux, &config.Config{}, nil)

	req := httptest.NewRequest(http.MethodGet, "/api/fleet-sim/api/workloads", nil)
	_, pattern := mux.Handler(req)
	if pattern != "" {
		t.Fatalf("matched route = %q, want no API fallback", pattern)
	}

	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNotFound)
	}
}

func (provider routerProxyCredentialProvider) ManagementCredential() (string, error) {
	return provider.token, nil
}

func TestRouterAPIProxyReplacesBrowserAuthorization(t *testing.T) {
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		nil,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	req := httptest.NewRequest(http.MethodGet, "/api/router/v1/models", nil)
	req.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	recorder := httptest.NewRecorder()

	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
	if authorization != "Bearer router-service-token" {
		t.Fatalf("Authorization = %q", authorization)
	}
}

func TestPlaygroundChatProxyPreservesIdentityAndStripsBrowserCredentials(t *testing.T) {
	t.Setenv("VLLM_SR_PORT_OFFSET", "0")
	type upstreamRequest struct {
		method  string
		path    string
		query   string
		headers http.Header
		body    string
	}
	received := make(chan upstreamRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read proxied request: %v", err)
		}
		received <- upstreamRequest{r.Method, r.URL.Path, r.URL.RawQuery, r.Header.Clone(), string(body)}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	target, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(t.TempDir(), "runtime.yaml")
	runtimeConfig := "version: v0.3\nlisteners:\n  - name: public\n    address: 127.0.0.1\n    port: " + target.Port() + "\n"
	if err := os.WriteFile(configPath, []byte(runtimeConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{EnvoyURL: server.URL, RouterAPIURL: server.URL, AbsConfigPath: configPath}
	mux := http.NewServeMux()
	registerRouterAPIProxy(mux, cfg, configureEnvoyProxy(cfg), nil, routerProxyCredentialProvider{token: "management-only-token"})

	for _, conversation := range []string{"conversation-one", "conversation-one", "conversation-two"} {
		body := `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello"}]}`
		request := httptest.NewRequest(http.MethodPost,
			"/api/router/v1/chat/completions?authToken=browser-query-token&keep=trace", strings.NewReader(body))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("X-Session-ID", "session-one")
		request.Header.Set("X-Conversation-ID", conversation)
		request.Header.Set("Authorization", "Bearer browser-token")
		request.Header.Set("Proxy-Authorization", "Bearer browser-proxy-token")
		request.Header.Set("Cookie", "vsr_session=browser-cookie")
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, request)
		if response.Code != http.StatusNoContent {
			t.Fatalf("chat proxy status = %d: %s", response.Code, response.Body.String())
		}
		got := <-received
		if got.method != http.MethodPost || got.path != "/v1/chat/completions" || got.query != "keep=trace" || got.body != body {
			t.Fatalf("proxied chat request changed: %#v", got)
		}
		if got.headers.Get("X-Session-ID") != "session-one" || got.headers.Get("X-Conversation-ID") != conversation {
			t.Fatalf("conversation identity not preserved: %v", got.headers)
		}
		for _, name := range []string{"Authorization", "Proxy-Authorization", "Cookie"} {
			if got.headers.Get(name) != "" {
				t.Fatalf("browser credential %s reached Envoy", name)
			}
		}
	}
}

func TestInferenceStreamStopsWhenLiveAuthorizationIsRevoked(t *testing.T) {
	for _, revocation := range []string{"permission", "session"} {
		t.Run(revocation, func(t *testing.T) {
			releaseSecond := make(chan struct{})
			upstreamCanceled := make(chan struct{})
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(upstreamCanceled)
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: first\n\n")
				w.(http.Flusher).Flush()
				select {
				case <-releaseSecond:
					_, _ = io.WriteString(w, "data: second\n\n")
					w.(http.Flusher).Flush()
				case <-r.Context().Done():
					return
				}
				<-r.Context().Done()
			}))
			defer upstream.Close()

			dbPath := filepath.Join(t.TempDir(), "auth.db")
			store, err := auth.NewStore(dbPath)
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			svc := auth.NewService(store, "inference-stream-secret", 1)
			const email, password = "stream@example.com", "test-admin-password"
			if err := svc.EnsureBootstrapAdmin(context.Background(), email, password, "Stream Admin"); err != nil {
				t.Fatal(err)
			}
			token, user, err := svc.Login(context.Background(), email, password)
			if err != nil {
				t.Fatal(err)
			}
			claims, err := svc.ParseToken(token)
			if err != nil {
				t.Fatal(err)
			}

			mux := auth.NewPolicyMux()
			envoyProxy, err := proxy.NewReverseProxy(upstream.URL, "", false)
			if err != nil {
				t.Fatal(err)
			}
			registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: upstream.URL}, envoyProxy, nil, nil)
			mux.Seal()
			front := httptest.NewServer(auth.AuthenticateRequest(svc, mux)(mux))
			defer front.Close()
			request, err := http.NewRequest(http.MethodPost, front.URL+"/api/router/v1/chat/completions", strings.NewReader(`{"stream":true,"messages":[{"role":"user","content":"hello"}]}`))
			if err != nil {
				t.Fatal(err)
			}
			request.Header.Set("Authorization", "Bearer "+token)
			request.Header.Set("Content-Type", "application/json")
			client := front.Client()
			client.Timeout = 10 * time.Second
			response, err := client.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			if response.StatusCode != http.StatusOK {
				t.Fatalf("status=%d", response.StatusCode)
			}
			reader := bufio.NewReader(response.Body)
			if line, readErr := reader.ReadString('\n'); readErr != nil || line != "data: first\n" {
				t.Fatalf("first event=%q error=%v", line, readErr)
			}
			if line, readErr := reader.ReadString('\n'); readErr != nil || line != "\n" {
				t.Fatalf("first event terminator=%q error=%v", line, readErr)
			}

			switch revocation {
			case "permission":
				db, openErr := sql.Open("sqlite3", dbPath+"?_busy_timeout=3000")
				if openErr != nil {
					t.Fatal(openErr)
				}
				if _, execErr := db.Exec(`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,0)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=0`, user.ID, auth.PermInferenceRun); execErr != nil {
					_ = db.Close()
					t.Fatal(execErr)
				}
				_ = db.Close()
			case "session":
				if err := store.RevokeSession(context.Background(), claims.ID); err != nil {
					t.Fatal(err)
				}
			}
			close(releaseSecond)
			remaining := make(chan string, 1)
			go func() {
				bytes, _ := io.ReadAll(reader)
				remaining <- string(bytes)
			}()
			select {
			case tail := <-remaining:
				if strings.Contains(tail, "second") {
					t.Fatalf("post-%s-revocation event escaped: %q", revocation, tail)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("revoked inference stream did not close")
			}
			select {
			case <-upstreamCanceled:
			case <-time.After(3 * time.Second):
				t.Fatal("revoked inference upstream did not close")
			}
		})
	}
}

func TestRouterAPIProxyExposesRuntimeDocumentation(t *testing.T) {
	t.Parallel()

	requested := make([]string, 0, 3)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requested = append(requested, r.URL.RequestURI())
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"source":"router"}`))
	}))
	defer server.Close()

	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		nil,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	for _, target := range []string{
		"/api/router/api/v1",
		"/api/router/openapi.json?path=%2Fconfig%2Frouter&method=PATCH",
		"/api/router/docs",
	} {
		recorder := httptest.NewRecorder()
		mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, target, nil))
		if recorder.Code != http.StatusOK {
			t.Fatalf("GET %s returned %d: %s", target, recorder.Code, recorder.Body.String())
		}
	}

	want := []string{
		"/api/v1",
		"/openapi.json?method=PATCH&path=%2Fconfig%2Frouter",
		"/docs",
	}
	if strings.Join(requested, ",") != strings.Join(want, ",") {
		t.Fatalf("Router requests = %v, want %v", requested, want)
	}
}

func TestRouterAPIProxyExposesKnowledgeBaseActivationHash(t *testing.T) {
	t.Parallel()
	const snapshot = `{"activation_status":"pending","active_runtime_hash":"old","generated_runtime_hash":"candidate"}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/v1/config/hash" {
			t.Errorf("unexpected upstream request: %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer router-service-token" {
			t.Error("activation polling did not use the router service credential")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(snapshot))
	}))
	defer upstream.Close()
	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: upstream.URL},
		nil,
		nil,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	request := httptest.NewRequest(http.MethodGet, "/api/router/api/v1/config/hash", nil)
	request.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK || response.Body.String() != snapshot {
		t.Fatalf("activation snapshot = %d %s", response.Code, response.Body.String())
	}
}

func TestRouterOutcomeProxyUsesServiceCredential(t *testing.T) {
	var authorization string
	var proxyAuthorization string
	var cookie string
	var queryCredential string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		proxyAuthorization = r.Header.Get("Proxy-Authorization")
		cookie = r.Header.Get("Cookie")
		queryCredential = r.URL.Query().Get("authToken")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		nil,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	req := httptest.NewRequest(http.MethodPost, "/api/router/api/v1/observability/outcomes?authToken=query-user-jwt", nil)
	req.Header.Set("Authorization", "Bearer dashboard-feedback-user-jwt")
	req.Header.Set("Proxy-Authorization", "Bearer proxy-user-jwt")
	req.Header.Set("Cookie", "vsr_session=cookie-user-jwt")
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
	if authorization != "Bearer router-service-token" {
		t.Fatalf("Authorization = %q", authorization)
	}
	if proxyAuthorization != "" || cookie != "" || queryCredential != "" {
		t.Fatalf("browser credentials leaked: proxy=%q cookie=%q query=%q", proxyAuthorization, cookie, queryCredential)
	}
}

func TestRouterAPIProxyRejectsUnknownManagementMutation(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		nil,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	req := httptest.NewRequest(http.MethodPost, "/api/router/v1/unknown-mutation", nil)
	req.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	recorder := httptest.NewRecorder()

	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNotFound)
	}
	if calls != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls)
	}
}

func TestRouterManagementProxyAllowlistMatchesDashboardSurfaces(t *testing.T) {
	tests := []struct {
		method string
		path   string
		want   bool
	}{
		{method: http.MethodGet, path: "/api/router/api/v1/config/hash", want: true},
		{method: http.MethodPost, path: "/api/router/api/v1/config/hash", want: false},
		{method: http.MethodGet, path: "/api/router/v1/models", want: true},
		{method: http.MethodGet, path: "/api/router/api/v1", want: true},
		{method: http.MethodGet, path: "/api/router/openapi.json", want: true},
		{method: http.MethodHead, path: "/api/router/docs", want: true},
		{method: http.MethodPost, path: "/api/router/openapi.json", want: false},
		{method: http.MethodGet, path: "/api/router/api/v1/observability/replays", want: true},
		{method: http.MethodGet, path: "/api/router/api/v1/observability/replays/replay-1", want: true},
		{method: http.MethodHead, path: "/api/router/api/v1/observability/replays", want: false},
		{method: http.MethodPost, path: "/api/router/api/v1/observability/replays", want: false},
		{method: http.MethodPost, path: "/api/router/api/v1/observability/outcomes", want: true},
		{method: http.MethodGet, path: "/api/router/api/v1/storage/response-cache/stats", want: true},
		{method: http.MethodPost, path: "/api/router/api/v1/storage/response-cache/invalidate", want: true},
		{method: http.MethodPost, path: "/api/router/api/v1/plugins/context_compression/preview", want: true},
		{method: http.MethodDelete, path: "/api/router/api/v1/plugins/context_compression/stats", want: false},
		{method: http.MethodPost, path: "/api/router/api/v1/config", want: false},
		{method: http.MethodPost, path: "/api/router/unknown", want: false},
	}
	for _, test := range tests {
		if got := routerManagementProxyRouteAllowed(test.method, test.path); got != test.want {
			t.Fatalf("routerManagementProxyRouteAllowed(%q, %q) = %v, want %v", test.method, test.path, got, test.want)
		}
	}
}

func TestRedactCredentialParams(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "authToken is redacted and other params survive",
			raw:  "http://localhost:8711/embedded/grafana/x?orgId=1&authToken=secret",
			want: "http://localhost:8711/embedded/grafana/x?authToken=%5BREDACTED%5D&orgId=1",
		},
		{
			name: "token is redacted",
			raw:  "http://localhost:8711/x?token=secret",
			want: "http://localhost:8711/x?token=%5BREDACTED%5D",
		},
		{
			name: "access_token is redacted",
			raw:  "http://localhost:8711/x?access_token=secret",
			want: "http://localhost:8711/x?access_token=%5BREDACTED%5D",
		},
		{
			name: "repeated authToken collapses to one redaction",
			raw:  "http://localhost:8711/x?authToken=a&authToken=b",
			want: "http://localhost:8711/x?authToken=%5BREDACTED%5D",
		},
		{
			name: "fragment is preserved",
			raw:  "http://localhost:8711/x?authToken=secret#gatewayUrl=http://localhost:8080",
			want: "http://localhost:8711/x?authToken=%5BREDACTED%5D#gatewayUrl=http://localhost:8080",
		},
		{
			name: "no credential is returned byte for byte",
			raw:  "http://localhost:8711/embedded/grafana/x?orgId=1&b=2",
			want: "http://localhost:8711/embedded/grafana/x?orgId=1&b=2",
		},
		{name: "empty", raw: "", want: ""},
		{name: "unparsable", raw: "::::", want: "[unparsable]"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := redactCredentialParams(tc.raw); got != tc.want {
				t.Fatalf("redactCredentialParams(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

func TestRedactCredentialParamsRemovesTheTokenFromTheLoggedReferer(t *testing.T) {
	t.Parallel()

	fakeJWT := "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
	logged := redactCredentialParams("http://localhost:8711/embedded/grafana/goto/x?orgId=1&authToken=" + fakeJWT)
	if strings.Contains(logged, fakeJWT) {
		t.Fatalf("the token survived redaction: %q", logged)
	}
}
