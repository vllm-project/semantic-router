package router

import (
	"net/http"
	"net/http/httptest"
	"testing"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestSRBenchRoutesUseIndependentServiceAndRetireEvaluation(t *testing.T) {
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte(`{"version":"sr-bench-1.0","benchmarks":[]}`))
	}))
	defer upstream.Close()
	t.Setenv("BENCH_TEST_TOKEN", "service-secret")
	cfg := &config.Config{SRBenchURL: upstream.URL, SRBenchTokenEnv: "BENCH_TEST_TOKEN"}
	mux := http.NewServeMux()
	registerSRBenchRoutes(mux, cfg)
	if !cfg.SRBenchAvailable || cfg.SRBenchUnavailableReason != "" {
		t.Fatal("configured sr-bench was unavailable")
	}
	request := httptest.NewRequest("GET", "/api/sr-bench/v1/catalog", nil)
	request = request.WithContext(dashboardauth.WithAuthContext(request.Context(), dashboardauth.AuthContext{UserID: "user", Role: dashboardauth.RoleAdmin}))
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != 200 || calls != 1 {
		t.Fatalf("catalog response=%d calls=%d", response.Code, calls)
	}
	for _, path := range []string{"/api/evaluation", "/api/evaluation/v1/runs", "/api/sr-bench/v1/unknown"} {
		response = httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest("GET", path, nil))
		if response.Code != 404 {
			t.Fatalf("path=%q status=%d", path, response.Code)
		}
	}
}

func TestSRBenchMissingAuthenticationFailsClosedWithoutLeakingConfig(t *testing.T) {
	t.Setenv("BENCH_TEST_TOKEN", "")
	cfg := &config.Config{SRBenchURL: "http://127.0.0.1:8090", SRBenchTokenEnv: "BENCH_TEST_TOKEN"}
	mux := http.NewServeMux()
	registerSRBenchRoutes(mux, cfg)
	if cfg.SRBenchAvailable || cfg.SRBenchUnavailableReason == "" {
		t.Fatal("missing token did not disable sr-bench")
	}
	request := httptest.NewRequest("GET", "/api/sr-bench/v1/catalog", nil)
	request = request.WithContext(dashboardauth.WithAuthContext(request.Context(), dashboardauth.AuthContext{UserID: "user", Role: dashboardauth.RoleAdmin}))
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != 503 {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}
