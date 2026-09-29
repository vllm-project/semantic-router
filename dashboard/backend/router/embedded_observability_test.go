package router

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestEmbeddedObservabilityNativeQueriesUseAuthenticatedProxy(t *testing.T) {
	store, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	svc := auth.NewService(store, "embedded-test-secret", 1)
	const email, password = "observability@example.com", "test-admin-password"
	if bootstrapErr := svc.EnsureBootstrapAdmin(context.Background(), email, password, "Test Admin"); bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	token, user, err := svc.Login(context.Background(), email, password)
	if err != nil {
		t.Fatal(err)
	}

	type received struct{ method, path, body string }
	var grafanaRequests, jaegerRequests []received
	grafana := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		grafanaRequests = append(grafanaRequests, received{r.Method, r.URL.Path, string(body)})
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"results":{"A":{"status":200,"frames":[]}}}`)
	}))
	defer grafana.Close()
	jaeger := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		jaegerRequests = append(jaegerRequests, received{method: r.Method, path: r.URL.Path})
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[]}`)
	}))
	defer jaeger.Close()

	cfg := &config.Config{GrafanaURL: grafana.URL, JaegerURL: jaeger.URL}
	mux := http.NewServeMux()
	proxies := dashboardProxySet{grafanaStatic: registerGrafanaRoutes(mux, cfg)}
	proxies.jaegerAPI, proxies.jaegerStatic = registerJaegerRoutes(mux, cfg)
	registerSmartAPIRouter(mux, proxies)
	handler := wrapWithAuth(mux, svc)

	const query = `{"queries":[{"refId":"A","datasource":{"uid":"metrics"},"expr":"up"}],"from":"1","to":"2"}`
	request := func(method, path, origin string, authenticated bool) *httptest.ResponseRecorder {
		t.Helper()
		r := httptest.NewRequest(method, "http://dashboard.example"+path, strings.NewReader(query))
		r.Header.Set("Origin", origin)
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Sec-Fetch-Site", "same-origin")
		if authenticated {
			r.AddCookie(&http.Cookie{Name: "vsr_session", Value: token})
		}
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		return w
	}
	for _, path := range []string{"/embedded/grafana/api/ds/query", "/api/ds/query"} {
		response := request(http.MethodPost, path, "http://dashboard.example", true)
		if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"frames":[]`) {
			t.Fatalf("native Grafana query %s: %d %s", path, response.Code, response.Body.String())
		}
	}
	if len(grafanaRequests) != 2 {
		t.Fatalf("Grafana requests=%d, want 2", len(grafanaRequests))
	}
	for _, got := range grafanaRequests {
		if got.method != http.MethodPost || got.path != "/api/ds/query" || got.body != query {
			t.Fatalf("query changed at proxy: %+v", got)
		}
	}
	response := request(http.MethodGet, "/embedded/jaeger/api/services", "", true)
	if response.Code != http.StatusOK || len(jaegerRequests) != 1 || jaegerRequests[0].path != "/api/services" {
		t.Fatalf("Jaeger read: status=%d upstream=%v", response.Code, jaegerRequests)
	}

	for _, tc := range []struct {
		path          string
		origin        string
		authenticated bool
		want          int
	}{
		{"/embedded/grafana/api/ds/query", "http://dashboard.example", false, http.StatusUnauthorized},
		{"/embedded/grafana/api/ds/query", "https://other.example", true, http.StatusForbidden},
		{"/embedded/grafana/api/dashboards/db", "http://dashboard.example", true, http.StatusForbidden},
	} {
		denied := request(http.MethodPost, tc.path, tc.origin, tc.authenticated)
		if denied.Code != tc.want || len(grafanaRequests) != 2 {
			t.Fatalf("rejected request %s: status=%d requests=%d", tc.path, denied.Code, len(grafanaRequests))
		}
	}
	if _, err := store.UpdateUserRoleOrStatus(context.Background(), user.ID, auth.RoleRead, ""); err != nil {
		t.Fatal(err)
	}
	response = request(http.MethodPost, "/api/ds/query", "http://dashboard.example", true)
	if response.Code != http.StatusForbidden || len(grafanaRequests) != 2 {
		t.Fatalf("query without logs.read: status=%d requests=%d", response.Code, len(grafanaRequests))
	}
	for _, path := range []string{"/api/services", "/api/traces/example", "/api/operations", "/api/dependencies", "/embedded/jaeger/api/services"} {
		response = request(http.MethodGet, path, "", true)
		if response.Code != http.StatusForbidden || len(jaegerRequests) != 1 {
			t.Fatalf("Jaeger read without logs.read %s: status=%d requests=%d", path, response.Code, len(jaegerRequests))
		}
	}
}

func TestJaegerRootRouteBoundariesMatchPermissionClassification(t *testing.T) {
	jaeger := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Test-Upstream", "jaeger")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer jaeger.Close()
	grafana := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Test-Upstream", "grafana")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer grafana.Close()
	cfg := &config.Config{GrafanaURL: grafana.URL, JaegerURL: jaeger.URL}
	mux := http.NewServeMux()
	proxies := dashboardProxySet{grafanaStatic: registerGrafanaRoutes(mux, cfg)}
	proxies.jaegerAPI, proxies.jaegerStatic = registerJaegerRoutes(mux, cfg)
	registerSmartAPIRouter(mux, proxies)
	for _, tc := range []struct {
		path, upstream, permission string
		status                     int
	}{
		{"/api/services", "jaeger", auth.PermLogsRead, http.StatusNoContent},
		{"/api/services/example/operations", "jaeger", auth.PermLogsRead, http.StatusNoContent},
		{"/api/traces/example", "jaeger", auth.PermLogsRead, http.StatusNoContent},
		{"/api/operations", "jaeger", auth.PermLogsRead, http.StatusNoContent},
		{"/api/dependencies", "jaeger", auth.PermLogsRead, http.StatusNoContent},
		{"/api/services-status", "", "", http.StatusNotFound},
		{"/api/traces-summary", "", "", http.StatusNotFound},
		{"/api/operations-old", "", "", http.StatusNotFound},
		{"/api/dependencies-other", "", "", http.StatusNotFound},
	} {
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, tc.path, nil))
		permissions := auth.RequiredPermissions(http.MethodGet, tc.path)
		if response.Code != tc.status || response.Header().Get("X-Test-Upstream") != tc.upstream ||
			(tc.permission == "" && len(permissions) != 0) ||
			(tc.permission != "" && (len(permissions) != 1 || permissions[0] != tc.permission)) {
			t.Fatalf("path=%s status=%d upstream=%s permissions=%v", tc.path, response.Code, response.Header().Get("X-Test-Upstream"), permissions)
		}
	}
}
