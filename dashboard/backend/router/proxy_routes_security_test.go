package router

import (
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
)

func TestSmartAPIRouterAllowsOnlyReadCapabilityGrafanaPaths(t *testing.T) {
	t.Parallel()

	var upstreamCalls atomic.Int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		upstreamCalls.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(upstream.Close)
	target, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parse upstream URL: %v", err)
	}
	mux := http.NewServeMux()
	registerSmartAPIRouter(mux, dashboardProxySet{
		grafanaStatic: httputil.NewSingleHostReverseProxy(target),
	})

	allowed := []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/api/frontend/settings"},
		{method: http.MethodGet, path: "/api/search?query=router"},
		{method: http.MethodGet, path: "/api/dashboards/uid/router"},
		{method: http.MethodPost, path: "/api/ds/query"},
	}
	for _, test := range allowed {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(test.method, test.path, strings.NewReader(`{}`))
		mux.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusNoContent {
			t.Errorf("%s %s status = %d, want 204", test.method, test.path, recorder.Code)
		}
	}
	if calls := upstreamCalls.Load(); calls != int64(len(allowed)) {
		t.Fatalf("upstream calls = %d, want %d", calls, len(allowed))
	}

	blocked := []struct {
		method string
		path   string
	}{
		{method: http.MethodGet, path: "/api/admin/settings"},
		{method: http.MethodGet, path: "/api/users"},
		{method: http.MethodGet, path: "/api/user/auth-tokens"},
		{method: http.MethodPost, path: "/api/admin/pause-all"},
		{method: http.MethodPut, path: "/api/datasources/1"},
		{method: http.MethodDelete, path: "/api/dashboards/uid/router"},
	}
	for _, test := range blocked {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(test.method, test.path, strings.NewReader(`{}`))
		mux.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusNotFound {
			t.Errorf("%s %s status = %d, want 404", test.method, test.path, recorder.Code)
		}
	}
	if calls := upstreamCalls.Load(); calls != int64(len(allowed)) {
		t.Fatalf("blocked request reached Grafana: upstream calls = %d", calls)
	}
}

func TestAllowedGrafanaReadAPIRequestDenyByDefault(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		method  string
		path    string
		allowed bool
	}{
		{method: http.MethodGet, path: "/api/user", allowed: true},
		{method: http.MethodGet, path: "/api/users", allowed: false},
		{method: http.MethodGet, path: "/api/org", allowed: true},
		{method: http.MethodGet, path: "/api/orgs", allowed: false},
		{method: http.MethodPost, path: "/api/ds/query", allowed: true},
		{method: http.MethodPost, path: "/api/ds/query/extra", allowed: false},
		{method: http.MethodPatch, path: "/api/annotations/1", allowed: false},
	} {
		if got := isAllowedGrafanaReadAPIRequest(test.method, test.path); got != test.allowed {
			t.Errorf("isAllowedGrafanaReadAPIRequest(%q, %q) = %v, want %v", test.method, test.path, got, test.allowed)
		}
	}
}
