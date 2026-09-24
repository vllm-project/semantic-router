package router

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

func TestDashboardRouteInventoryHasCompletePolicies(t *testing.T) {
	server := setupRouteInventoryServer(t)
	contracts := server.routePolicies.Contracts()
	if len(contracts) < 80 {
		t.Fatalf("route contracts = %d, want at least 80", len(contracts))
	}
	for _, contract := range contracts {
		if err := auth.ValidateRouteContract(contract); err != nil {
			t.Fatalf("invalid route contract %q: %v", contract.Pattern, err)
		}
		for _, policy := range contract.Policies {
			if policy.Public {
				continue
			}
			for _, permission := range append([]string{policy.Permission}, policy.AdditionalPermissions...) {
				if !slices.Contains(auth.AllPermissions, permission) {
					t.Errorf("%s %s uses unknown permission %q", policy.Method, contract.Pattern, permission)
				}
			}
			if policy.AuditMode != auth.AuditNone && policy.AuditAction == "" {
				t.Errorf("%s %s has no audit action", policy.Method, contract.Pattern)
			}
		}
	}
}

func TestDashboardRoutePoliciesSeparateSecurityDomains(t *testing.T) {
	server := setupRouteInventoryServer(t)
	for _, test := range []struct{ method, path, permission string }{
		{http.MethodPost, "/api/router/v1/chat/completions", auth.PermInferenceRun},
		{http.MethodPost, "/api/router/api/v1/observability/outcomes", auth.PermFeedbackSubmit},
		{http.MethodGet, "/api/router/api/v1/observability/replays/record-1", auth.PermReplayRead},
		{http.MethodPost, "/api/router/config/deploy", auth.PermConfigDeploy},
		{http.MethodPost, "/api/mcp/tools/execute", auth.PermToolsUse},
		{http.MethodPatch, "/api/admin/users/user-1", auth.PermUsersManage},
		{http.MethodGet, "/api/openclaw/teams", auth.PermOpenClawRead},
		{http.MethodPost, "/api/openclaw/teams", auth.PermOpenClaw},
	} {
		policy, result := server.routePolicies.LookupRoutePolicy(test.method, test.path)
		if result != auth.RouteFound || policy.Permission != test.permission {
			t.Errorf("%s %s: lookup=%v permission=%q, want %q", test.method, test.path, result, policy.Permission, test.permission)
		}
	}
	for _, path := range []string{"/api/unmapped", "/api/router/api/v1/observability/replays/record-1/unmapped", "/api/services-status"} {
		if _, result := server.routePolicies.LookupRoutePolicy(http.MethodGet, path); result != auth.RouteNotFound {
			t.Errorf("unknown route %s lookup=%v", path, result)
		}
		response := httptest.NewRecorder()
		server.Handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
		if response.Code != http.StatusForbidden {
			t.Errorf("unknown route %s status=%d", path, response.Code)
		}
	}
}

func setupRouteInventoryServer(t *testing.T) *Server {
	t.Helper()
	dir := t.TempDir()
	staticDir := filepath.Join(dir, "static")
	if err := os.MkdirAll(staticDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("ok"), 0o644); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{
		Port: "19000", AuthDBPath: filepath.Join(dir, "auth.db"), JWTSecret: "route-inventory-secret",
		JWTExpiryHours: 1, StaticDir: staticDir, ConfigFile: configPath, AbsConfigPath: configPath,
		ConfigDir: dir, RouterAPIURL: "http://127.0.0.1:18080", RouterMetrics: "http://127.0.0.1:19190/metrics",
		MCPEnabled: true, OpenClawEnabled: true, OpenClawDataDir: filepath.Join(dir, "openclaw"),
		WorkflowDBPath:         filepath.Join(dir, "workflow.sqlite"),
		ConfigProjectionDBPath: filepath.Join(dir, "projection.sqlite"),
	}
	server := Setup(cfg, setupmode.New(configPath, false))
	t.Cleanup(func() { _ = server.Close() })
	return server
}
