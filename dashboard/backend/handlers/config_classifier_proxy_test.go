package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRouterClassifierProxyHandlerForwardsRouterRequests(t *testing.T) {
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/storage/knowledge-bases/example" {
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

	req := httptest.NewRequest(http.MethodGet, "/api/router/api/v1/storage/knowledge-bases/example", nil)
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
	req := httptest.NewRequest(http.MethodDelete, "/api/router/api/v1/storage/knowledge-bases/example", nil)
	rr := httptest.NewRecorder()

	handler(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403 Forbidden, got %d: %s", rr.Code, rr.Body.String())
	}
}

type classifierProxyCredentialProvider struct {
	token string
}

func (provider classifierProxyCredentialProvider) ManagementCredential() (string, error) {
	return provider.token, nil
}

func TestRouterClassifierProxyReplacesBrowserAuthorization(t *testing.T) {
	routerAPI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer classifier-service-token" {
			t.Fatalf("Authorization = %q", got)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer routerAPI.Close()

	handler := RouterClassifierProxyHandler(
		routerAPI.URL,
		false,
		classifierProxyCredentialProvider{token: "classifier-service-token"},
	)
	req := httptest.NewRequest(http.MethodGet, "/api/router/api/v1/storage/knowledge-bases/example", nil)
	req.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	rr := httptest.NewRecorder()

	handler(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Fatalf("status = %d, body = %s", rr.Code, rr.Body.String())
	}
}

func TestRouterClassifierProxyRejectsUnknownSubpathAndMethod(t *testing.T) {
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; w.WriteHeader(http.StatusNoContent) }))
	defer upstream.Close()
	handler := RouterClassifierProxyHandler(upstream.URL, false)
	for _, tc := range []struct{ method, path string }{
		{http.MethodPost, "/api/router/api/v1/storage/knowledge-bases/example"},
		{http.MethodGet, "/api/router/api/v1/storage/knowledge-bases/example/unknown"},
	} {
		w := httptest.NewRecorder()
		handler(w, httptest.NewRequest(tc.method, tc.path, nil))
		if w.Code != http.StatusForbidden {
			t.Fatalf("undeclared KB proxy allowed: %+v status=%d", tc, w.Code)
		}
	}
	if calls != 0 {
		t.Fatal("undeclared KB request reached Router")
	}
}
