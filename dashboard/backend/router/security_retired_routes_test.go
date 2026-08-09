package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// TestRegisterCoreRoutesClaimsRetiredSecurityPaths guards the reason the retired
// /api/security surface stays registered instead of simply disappearing. registerSmartAPIRouter
// owns an /api/ catch-all, so an unclaimed path is proxied to an unrelated backend and answers
// 502 rather than reporting the surface as gone.
//
// This drives registerCoreRoutes itself: a test that builds its own mux would keep passing if
// the registration were dropped.
func TestRegisterCoreRoutesClaimsRetiredSecurityPaths(t *testing.T) {
	dir := t.TempDir()
	cfg := &config.Config{
		AbsConfigPath: dir + "/config.yaml",
		ConfigDir:     dir,
	}

	mux := http.NewServeMux()
	registerCoreRoutes(mux, cfg, setupmode.New(cfg.AbsConfigPath, cfg.SetupMode))
	mux.HandleFunc("/api/", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	})

	for _, tc := range []struct {
		method string
		path   string
		body   string
	}{
		{method: http.MethodGet, path: "/api/security"},
		{method: http.MethodGet, path: "/api/security/policy"},
		{method: http.MethodPut, path: "/api/security/policy", body: `{"role_mappings":[]}`},
		{method: http.MethodPost, path: "/api/security/policy/preview", body: `{"role_mappings":[]}`},
		{method: http.MethodGet, path: "/api/security/anything-else"},
	} {
		t.Run(tc.method+" "+tc.path, func(t *testing.T) {
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body)))

			if rec.Code != http.StatusGone {
				t.Fatalf("%s %s returned %d, want %d; the request reached the generic /api/ handler",
					tc.method, tc.path, rec.Code, http.StatusGone)
			}
		})
	}
}
