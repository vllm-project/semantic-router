package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestModelVerificationSessionRevokedDuringBodyDoesNotCallProvider(t *testing.T) {
	store, err := dashboardauth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	service := dashboardauth.NewService(store, "model-revocation-test-secret", 1)
	const password = "Revocation test passphrase #2026"
	hash, err := service.HashPassword(password)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateUser(t.Context(), "revocation@example.test", "Revocation Test", hash, dashboardauth.RoleWrite, "active"); err != nil {
		t.Fatal(err)
	}
	token, _, err := service.Login(t.Context(), "revocation@example.test", password)
	if err != nil {
		t.Fatal(err)
	}

	var providerCalls atomic.Int32
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", "")
	verify := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client: modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
			providerCalls.Add(1)
			return modelVerificationHTTPResponse(http.StatusOK, `{"choices":[{"message":{"content":"OK"}}]}`), nil
		}),
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	mux := dashboardauth.NewPolicyMux()
	mux.HandlePolicyFunc(dashboardauth.ProtectedDelegatedAuditRoute(
		modelVerificationPath, dashboardauth.PermEvalRun, "model.inference_verify",
		dashboardauth.SensitivitySensitive, dashboardauth.ResourceOwnerInference, 2<<20, http.MethodPost,
	), verify)
	mux.Seal()
	handler := dashboardauth.AuthenticateRequest(service, mux)(mux)

	response, finish := pausedOutboundRequest(t, http.MethodPost, modelVerificationPath, handler,
		`{"model":"`, `logical-model"}`, func(request *http.Request) *http.Request {
			request.Header.Set("Authorization", "Bearer "+token)
			return request
		})
	finish(func() {
		if revokeErr := service.RevokeToken(t.Context(), token); revokeErr != nil {
			t.Fatal(revokeErr)
		}
	})
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("revoked session status = %d, body = %s", response.Code, response.Body.String())
	}
	if got := providerCalls.Load(); got != 0 {
		t.Fatalf("provider calls after session revocation = %d", got)
	}
}

func TestModelVerificationPermissionRevokedBeforeProviderCall(t *testing.T) {
	var providerCalls atomic.Int32
	var permitted atomic.Bool
	permitted.Store(true)
	config := modelVerificationTestConfig(t, "https://provider.example", "openai", "")
	verify := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client: modelVerificationRoundTripper(func(*http.Request) (*http.Response, error) {
			providerCalls.Add(1)
			return modelVerificationHTTPResponse(http.StatusOK, `{"choices":[{"message":{"content":"OK"}}]}`), nil
		}),
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	response, finish := pausedOutboundRequest(t, http.MethodPost, modelVerificationPath, verify,
		`{"model":"`, `logical-model"}`, func(request *http.Request) *http.Request {
			request = request.WithContext(dashboardauth.WithPermissionRevalidator(request.Context(), func(context.Context) error {
				if !permitted.Load() {
					return errors.New("evaluation.run revoked")
				}
				return nil
			}))
			if err := dashboardauth.RevalidateRequest(request); err != nil {
				t.Fatalf("request was not admitted: %v", err)
			}
			return request
		})
	finish(func() { permitted.Store(false) })
	if response.Code != http.StatusForbidden || providerCalls.Load() != 0 {
		t.Fatalf("revoked permission status = %d, provider calls = %d", response.Code, providerCalls.Load())
	}
}

func TestFetchRawPermissionRevokedDuringBodyDoesNotFetch(t *testing.T) {
	var providerCalls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		providerCalls.Add(1)
		_, _ = w.Write([]byte("unexpected"))
	}))
	t.Cleanup(provider.Close)
	var permitted atomic.Bool
	permitted.Store(true)
	response, finish := pausedOutboundRequest(t, http.MethodPost, "/api/tools/fetch-raw", FetchRawHandler(),
		`{"url":"`, provider.URL+`"}`, func(request *http.Request) *http.Request {
			request = request.WithContext(dashboardauth.WithPermissionRevalidator(request.Context(), func(context.Context) error {
				if !permitted.Load() {
					return errors.New("tools.use revoked")
				}
				return nil
			}))
			if err := dashboardauth.RevalidateRequest(request); err != nil {
				t.Fatalf("request was not admitted: %v", err)
			}
			return request
		})
	finish(func() { permitted.Store(false) })
	if response.Code != http.StatusForbidden {
		t.Fatalf("revoked permission status = %d, body = %s", response.Code, response.Body.String())
	}
	if got := providerCalls.Load(); got != 0 {
		t.Fatalf("fetch calls after permission revocation = %d", got)
	}
}

func TestOutboundResultsPermissionRevokedDuringSlowResponse(t *testing.T) {
	setupPayload := mustJSONRaw(t, createValidSetupPatch())
	tests := []struct {
		name         string
		path         string
		handler      func(*testing.T) http.Handler
		requestBody  func(string) any
		responseBody []byte
	}{
		{
			name:    "OpenWeb direct fetch",
			path:    "/api/tools/open-web",
			handler: func(*testing.T) http.Handler { return OpenWebHandler() },
			requestBody: func(target string) any {
				return OpenWebRequest{URL: target}
			},
			responseBody: []byte("<html><body>revoked-page-content</body></html>"),
		},
		{
			name:    "FetchRaw",
			path:    "/api/tools/fetch-raw",
			handler: func(*testing.T) http.Handler { return FetchRawHandler() },
			requestBody: func(target string) any {
				return FetchRawRequest{URL: target}
			},
			responseBody: []byte("revoked-raw-content"),
		},
		{
			name: "setup remote import",
			path: "/api/setup/import-remote",
			handler: func(t *testing.T) http.Handler {
				configPath := createBootstrapSetupConfig(t, t.TempDir())
				return SetupImportRemoteHandler(configPath, setupmode.New(configPath, false))
			},
			requestBody: func(target string) any {
				return SetupImportRemoteRequest{URL: target}
			},
			responseBody: setupPayload,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// These globals are test-only policy inputs. The fixture is reachable
			// through a public-looking hostname whose one allowed answer is loopback.
			allowLoopbackForTest(t)
			withInwardResolver(t, "127.0.0.1")

			upstreamReached := make(chan struct{})
			releaseUpstream := make(chan struct{})
			var releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(releaseUpstream) }) }
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
				close(upstreamReached)
				<-releaseUpstream
				_, _ = w.Write(tt.responseBody)
			}))
			defer func() {
				unblock()
				server.Close()
			}()

			parsed, err := url.Parse(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			target := "http://public-fetch.example:" + parsed.Port() + "/document"
			body, err := json.Marshal(tt.requestBody(target))
			if err != nil {
				t.Fatal(err)
			}

			var permitted atomic.Bool
			permitted.Store(true)
			request := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(string(body)))
			request = request.WithContext(dashboardauth.WithPermissionRevalidator(request.Context(), func(context.Context) error {
				if !permitted.Load() {
					return errors.New("permission revoked during upstream response")
				}
				return nil
			}))
			handler := tt.handler(t)
			response := httptest.NewRecorder()
			done := make(chan struct{})
			go func() {
				handler.ServeHTTP(response, request)
				close(done)
			}()

			select {
			case <-upstreamReached:
			case <-done:
				t.Fatalf("handler returned before reaching the upstream: %d %s", response.Code, response.Body.String())
			case <-time.After(10 * time.Second):
				t.Fatal("handler did not reach the upstream")
			}
			permitted.Store(false)
			unblock()
			select {
			case <-done:
			case <-time.After(10 * time.Second):
				t.Fatal("handler did not finish after the upstream responded")
			}
			if response.Code != http.StatusForbidden {
				t.Fatalf("revoked permission status = %d, body = %s", response.Code, response.Body.String())
			}
			if strings.Contains(response.Body.String(), "revoked-") || strings.Contains(response.Body.String(), "providers") {
				t.Fatalf("revoked request returned upstream content: %s", response.Body.String())
			}
		})
	}
}

func TestWeatherPermissionRevokedAfterGeocodingSkipsForecast(t *testing.T) {
	var permitted atomic.Bool
	permitted.Store(true)
	var forecastCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/search":
			permitted.Store(false)
			_ = json.NewEncoder(w).Encode(map[string]any{"results": []map[string]any{{
				"name": "Chengdu", "latitude": 30.67, "longitude": 104.06, "timezone": "Asia/Shanghai",
			}}})
		case "/v1/forecast":
			forecastCalls.Add(1)
			w.WriteHeader(http.StatusOK)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(upstream.Close)
	originalGeocodingURL := weatherGeocodingBaseURL
	originalForecastURL := weatherForecastBaseURL
	weatherGeocodingBaseURL = upstream.URL
	weatherForecastBaseURL = upstream.URL
	t.Cleanup(func() {
		weatherGeocodingBaseURL = originalGeocodingURL
		weatherForecastBaseURL = originalForecastURL
	})

	request := httptest.NewRequest(http.MethodPost, "/api/tools/weather", strings.NewReader(`{"location":"Chengdu"}`))
	request = request.WithContext(dashboardauth.WithPermissionRevalidator(request.Context(), func(context.Context) error {
		if !permitted.Load() {
			return errors.New("tools.use revoked")
		}
		return nil
	}))
	response := httptest.NewRecorder()
	WeatherHandler().ServeHTTP(response, request)
	if response.Code != http.StatusForbidden || forecastCalls.Load() != 0 {
		t.Fatalf("status = %d, forecast calls = %d", response.Code, forecastCalls.Load())
	}
}

// The first fragment is consumed while authorization is still active. The
// caller then revokes access before releasing the rest of the JSON body.
func pausedOutboundRequest(
	t *testing.T, method, path string, handler http.Handler, first, rest string,
	prepare func(*http.Request) *http.Request,
) (*httptest.ResponseRecorder, func(func())) {
	t.Helper()
	reader, writer := io.Pipe()
	request := httptest.NewRequest(method, path, reader)
	if prepare != nil {
		request = prepare(request)
	}
	response := httptest.NewRecorder()
	return response, func(revoke func()) {
		done := make(chan struct{})
		go func() {
			handler.ServeHTTP(response, request)
			close(done)
		}()
		firstWritten := make(chan error, 1)
		go func() {
			_, writeErr := io.WriteString(writer, first)
			firstWritten <- writeErr
		}()
		select {
		case writeErr := <-firstWritten:
			if writeErr != nil {
				t.Fatalf("write first request fragment: %v", writeErr)
			}
		case <-time.After(10 * time.Second):
			_ = writer.Close()
			t.Fatal("handler did not consume the first request fragment")
		}
		revoke()
		if _, writeErr := io.WriteString(writer, rest); writeErr != nil {
			t.Fatalf("write final request fragment: %v", writeErr)
		}
		if closeErr := writer.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			t.Fatal("handler did not return after request completion")
		}
	}
}
