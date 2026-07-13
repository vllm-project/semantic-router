package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestRouterClassifierProxyHandlerForwardsRouterRequests(t *testing.T) {
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/config/kbs/example" {
			t.Fatalf("unexpected proxied path: %s", r.URL.Path)
		}
		if got := r.Header.Get("X-Test-Header"); got != "present" {
			t.Fatalf("expected forwarded request header, got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"name":"example"}`)
	}))
	defer routerAPI.Close()

	handler := RouterClassifierProxyHandler(routerAPI.URL, false)

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/kbs/example", nil)
	req.Header.Set("X-Test-Header", "present")
	rr := httptest.NewRecorder()
	handler(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}
	if got := strings.TrimSpace(rr.Body.String()); got != `{"name":"example"}` {
		t.Fatalf("unexpected proxy body: %s", got)
	}
}

func TestRouterClassifierProxyHandlerBlocksReadonlyMutations(t *testing.T) {
	handler := RouterClassifierProxyHandler("http://router.internal", true)
	req := httptest.NewRequest(http.MethodDelete, "/api/router/config/kbs/example", nil)
	rr := httptest.NewRecorder()

	handler(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403 Forbidden, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestRouterClassifierProxyStripsCredentialsAndHopHeaders(t *testing.T) {
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, name := range []string{"Authorization", "Cookie", "X-CSRF-Token", "X-VSR-Auth-Mode", "X-Request-Hop"} {
			if got := r.Header.Get(name); got != "" {
				t.Errorf("upstream received protected header %s=%q", name, got)
			}
		}
		w.Header().Set("Connection", "X-Response-Hop")
		w.Header().Set("X-Response-Hop", "secret")
		w.Header().Set("Set-Cookie", "router_session=secret")
		w.Header().Set("X-Safe", "present")
		_, _ = io.WriteString(w, `{}`)
	}))
	defer routerAPI.Close()

	req := httptest.NewRequest(http.MethodGet, "/api/router/config/kbs/example", nil)
	req.Header.Set("Authorization", "Bearer dashboard-secret")
	req.Header.Set("Cookie", "vsr_session=dashboard-secret")
	req.Header.Set("X-CSRF-Token", "dashboard-secret")
	req.Header.Set("X-VSR-Auth-Mode", "bearer")
	req.Header.Set("Connection", "X-Request-Hop")
	req.Header.Set("X-Request-Hop", "secret")
	recorder := httptest.NewRecorder()
	RouterClassifierProxyHandler(routerAPI.URL, false)(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %q", recorder.Code, recorder.Body.String())
	}
	for _, name := range []string{"Set-Cookie", "X-Response-Hop", "Connection"} {
		if got := recorder.Header().Get(name); got != "" {
			t.Fatalf("response leaked protected header %s=%q", name, got)
		}
	}
	if got := recorder.Header().Get("X-Safe"); got != "present" {
		t.Fatalf("safe response header = %q", got)
	}
}

func TestRouterClassifierProxyRejectsKnownOversizedBodyBeforeUpstream(t *testing.T) {
	var hits atomic.Int32
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer routerAPI.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/router/config/kbs", strings.NewReader("{}"))
	req.ContentLength = routerClassifierProxyMaxRequestBodyBytes + 1
	recorder := httptest.NewRecorder()
	RouterClassifierProxyHandler(routerAPI.URL, false)(recorder, req)

	if recorder.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", recorder.Code)
	}
	if hits.Load() != 0 {
		t.Fatalf("upstream hits = %d, want 0", hits.Load())
	}
}

func TestRouterClassifierProxyDoesNotFollowRedirects(t *testing.T) {
	var redirectTargetHits atomic.Int32
	redirectTarget := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirectTargetHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer redirectTarget.Close()

	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Redirect(w, &http.Request{}, redirectTarget.URL, http.StatusFound)
	}))
	defer routerAPI.Close()

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/router/config/kbs", nil)
	RouterClassifierProxyHandler(routerAPI.URL, false)(recorder, req)

	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", recorder.Code)
	}
	if redirectTargetHits.Load() != 0 {
		t.Fatalf("redirect target hits = %d, want 0", redirectTargetHits.Load())
	}
}

func TestRouterClassifierProxyRejectsInvalidTargetAndPath(t *testing.T) {
	tests := []struct {
		name   string
		target string
		path   string
		status int
	}{
		{name: "invalid target", target: "file:///tmp/router", path: "/api/router/config/kbs", status: http.StatusBadGateway},
		{name: "outside path", target: "http://router.internal", path: "/api/router/health", status: http.StatusBadRequest},
		{name: "dot segment", target: "http://router.internal", path: "/api/router/config/kbs/../secret", status: http.StatusBadRequest},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, test.path, nil)
			RouterClassifierProxyHandler(test.target, false)(recorder, req)
			if recorder.Code != test.status {
				t.Fatalf("status = %d, want %d", recorder.Code, test.status)
			}
		})
	}
}
