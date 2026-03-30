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
