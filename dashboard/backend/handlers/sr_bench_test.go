package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func srBenchTestRequest(method, path, body string) *http.Request {
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	return r.WithContext(dashboardauth.WithAuthContext(r.Context(), dashboardauth.AuthContext{UserID: "owner-123", Role: dashboardauth.RoleWrite}))
}

func TestSRBenchProxyPreservesRequestsWithTrustedIdentity(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.URL.Path != "/api/sr-bench/v1/runs" || r.Method != http.MethodPost {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL)
		}
		if r.Header.Get("Authorization") != "Bearer service-secret" || r.Header.Get("X-SR-Bench-Actor-ID") != "owner-123" || r.Header.Get("X-SR-Bench-Actor-Role") != dashboardauth.RoleWrite {
			t.Error("server-owned credentials or actor missing")
		}
		if r.Header.Get("Cookie") != "" || r.Header.Get("X-Forwarded-Host") != "" {
			t.Error("browser credentials escaped to service")
		}
		body, _ := io.ReadAll(r.Body)
		if string(body) != `{"manifest":{"name":"run"}}` {
			t.Errorf("body changed: %s", body)
		}
		w.Header().Set("Set-Cookie", "upstream=secret")
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"id":"run-1","status":"pending"}`))
	}))
	defer upstream.Close()
	handler, err := NewSRBenchHandler(upstream.URL, "service-secret", false)
	if err != nil {
		t.Fatal(err)
	}
	request := srBenchTestRequest(http.MethodPost, SRBenchAPIPath+"/runs", `{"manifest":{"name":"run"}}`)
	request.Header.Set("Authorization", "Bearer browser-secret")
	request.Header.Set("X-SR-Bench-Actor-ID", "forged-admin")
	request.Header.Set("X-SR-Bench-Actor-Role", "admin")
	request.Header.Set("Cookie", "session=secret")
	request.Header.Set("X-Forwarded-Host", "attacker.example")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != 202 || calls.Load() != 1 {
		t.Fatalf("response=%d calls=%d body=%s", response.Code, calls.Load(), response.Body.String())
	}
	if response.Header().Get("Set-Cookie") != "" || response.Header().Get("Cache-Control") != "private, no-store" {
		t.Fatalf("unexpected headers: %v", response.Header())
	}
}

func TestSRBenchProxyRejectsUntrustedAndUnsupportedRequests(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("unexpected upstream call") }))
	defer upstream.Close()
	for _, tc := range []struct {
		name, method, path, token string
		readonly, anonymous       bool
		status                    int
	}{
		{name: "anonymous", method: "GET", path: "/catalog", token: "service-secret", anonymous: true, status: 401},
		{name: "unconfigured", method: "GET", path: "/catalog", status: 503},
		{name: "readonly", method: "POST", path: "/runs", token: "service-secret", readonly: true, status: 403},
		{name: "unsupported method", method: "DELETE", path: "/runs", token: "service-secret", status: 405},
		{name: "unknown route", method: "GET", path: "/proxy", token: "service-secret", status: 404},
		{name: "invalid ID", method: "GET", path: "/runs/a.b", token: "service-secret", status: 404},
		{name: "escaped path", method: "GET", path: "/runs/a%2fb", token: "service-secret", status: 404},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler, err := NewSRBenchHandler(upstream.URL, tc.token, tc.readonly)
			if err != nil {
				t.Fatal(err)
			}
			request := srBenchTestRequest(tc.method, SRBenchAPIPath+tc.path, "")
			if tc.anonymous {
				request = httptest.NewRequest(tc.method, SRBenchAPIPath+tc.path, nil)
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != tc.status {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestSRBenchProxyCancellationEventsAndRestartAreIndependent(t *testing.T) {
	var calls []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls = append(calls, r.Method+" "+r.URL.RequestURI())
		_, _ = w.Write([]byte(`{"id":"run-1","status":"cancelled"}`))
	}))
	defer upstream.Close()
	for _, item := range []struct{ method, path string }{
		{"GET", "/runs/run-1/events?after=7"}, {"POST", "/runs/run-1/cancel"}, {"GET", "/runs/run-1"},
	} {
		// Recreating Dashboard transport never reconciles or cancels service workers.
		handler, err := NewSRBenchHandler(upstream.URL, "service-secret", false)
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest(item.method, SRBenchAPIPath+item.path, "{}"))
		if response.Code != 200 {
			t.Fatal(response.Body.String())
		}
	}
	if len(calls) != 3 || calls[0] != "GET /api/sr-bench/v1/runs/run-1/events?after=7" || calls[1] != "POST /api/sr-bench/v1/runs/run-1/cancel" {
		t.Fatalf("unexpected actions: %v", calls)
	}
}

func TestSRBenchProxyDoesNotFollowRedirectsOrLeakTransportErrors(t *testing.T) {
	for _, redirect := range []bool{false, true} {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if redirect {
				http.Redirect(w, r, "https://secret.example/token", http.StatusFound)
				return
			}
			time.Sleep(30 * time.Millisecond)
		}))
		handler, err := NewSRBenchHandler(upstream.URL, "service-secret", false)
		if err != nil {
			t.Fatal(err)
		}
		handler.client.Timeout = 5 * time.Millisecond
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest("GET", SRBenchAPIPath+"/catalog", ""))
		upstream.Close()
		if response.Code != 502 || strings.Contains(response.Body.String(), "secret") || strings.Contains(response.Body.String(), upstream.URL) {
			t.Fatalf("error response: %d %s", response.Code, response.Body.String())
		}
	}
}

func TestSRBenchOfflineRoutes(t *testing.T) {
	for _, path := range []string{"/replays", "/runs/run-1/regrade", "/runs/run-1/export", "/runs/run-1/recover-plan", "/runs/run-1/recover"} {
		if method, found := srBenchRouteMethod(SRBenchAPIPath + path); !found || method != http.MethodPost {
			t.Fatalf("offline route missing: %s", path)
		}
	}
}

func TestSRBenchEvidencePaginationAndCallDetail(t *testing.T) {
	var paths []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.RequestURI())
		_, _ = w.Write([]byte(`{"calls":[],"next_cursor":25}`))
	}))
	defer upstream.Close()
	handler, err := NewSRBenchHandler(upstream.URL, "service-secret", true)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{
		"/runs/run-1/calls?after=5&limit=20",
		"/runs/run-1/results?after=5&limit=20",
		"/runs/run-1/calls/call_2",
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, srBenchTestRequest("GET", SRBenchAPIPath+path, ""))
		if response.Code != http.StatusOK || response.Body.String() != `{"calls":[],"next_cursor":25}` {
			t.Fatalf("evidence response: %d %s", response.Code, response.Body.String())
		}
		if paths[len(paths)-1] != SRBenchAPIPath+path {
			t.Fatalf("pagination changed: %s", paths[len(paths)-1])
		}
	}
	for _, path := range []string{"/runs/run-1/calls/..", "/runs/run-1/calls/", "/runs/run-1/calls/call-1/extra"} {
		if _, known := srBenchRouteMethod(SRBenchAPIPath + path); known {
			t.Fatalf("invalid call route accepted: %s", path)
		}
	}
}
