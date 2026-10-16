package router

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

// newMetricsRouteHarness builds the authenticated handler around a metrics
// route registration and returns it with a request helper.
func newMetricsRouteHarness(t *testing.T, cfg *config.Config) (http.Handler, func(method, path string, authenticated bool) *httptest.ResponseRecorder) {
	t.Helper()
	store, err := auth.NewStore(filepath.Join(t.TempDir(), "auth.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	svc := auth.NewService(store, "metrics-route-test-secret", 1)
	const email, password = "metrics@example.com", "test-admin-password"
	if bootstrapErr := svc.EnsureBootstrapAdmin(context.Background(), email, password, "Test Admin"); bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	token, _, err := svc.Login(context.Background(), email, password)
	if err != nil {
		t.Fatal(err)
	}

	mux := auth.NewPolicyMux()
	registerMetricsRoutes(mux, cfg)
	mux.Seal()
	handler := wrapWithAuth(mux, svc, mux)

	request := func(method, path string, authenticated bool) *httptest.ResponseRecorder {
		t.Helper()
		r := httptest.NewRequest(method, "http://dashboard.example"+path, nil)
		r.Header.Set("Sec-Fetch-Site", "same-origin")
		if authenticated {
			r.AddCookie(&http.Cookie{Name: "vsr_session", Value: token})
		}
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		return w
	}
	return handler, request
}

func TestMetricsRouterRouteAuthenticatesBeforeTheRedirect(t *testing.T) {
	const target = "http://127.0.0.1:9190/metrics"
	_, request := newMetricsRouteHarness(t, &config.Config{RouterMetrics: target})

	anonymous := request(http.MethodGet, "/metrics/router", false)
	if anonymous.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous status=%d, want 401", anonymous.Code)
	}
	if location := anonymous.Header().Get("Location"); location != "" {
		t.Fatalf("anonymous Location=%q, want the internal target withheld", location)
	}

	authorized := request(http.MethodGet, "/metrics/router", true)
	if authorized.Code != http.StatusTemporaryRedirect {
		t.Fatalf("authorized status=%d, want 307", authorized.Code)
	}
	if location := authorized.Header().Get("Location"); location != target {
		t.Fatalf("authorized Location=%q, want %q", location, target)
	}
}

func TestMetricsRouterRouteWithoutATargetAnswersServiceUnavailable(t *testing.T) {
	_, request := newMetricsRouteHarness(t, &config.Config{})

	response := request(http.MethodGet, "/metrics/router", true)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d, want 503", response.Code)
	}
	if !strings.Contains(response.Body.String(), "TARGET_ROUTER_METRICS_URL") {
		t.Fatalf("body does not name the setting to configure: %s", response.Body.String())
	}
}
