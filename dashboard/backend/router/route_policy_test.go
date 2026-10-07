package router

import (
	"database/sql"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
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
			if policy.AuditAction == "" {
				t.Errorf("%s %s has no audit action", policy.Method, contract.Pattern)
			}
		}
	}
	for _, route := range []struct{ method, path string }{
		{http.MethodGet, "/api/ml-pipeline/availability"},
		{http.MethodGet, "/api/ml-pipeline/jobs"},
		{http.MethodGet, "/api/ml-pipeline/jobs/job-1"},
		{http.MethodPost, "/api/ml-pipeline/benchmark"},
		{http.MethodPost, "/api/ml-pipeline/train"},
		{http.MethodPost, "/api/ml-pipeline/config"},
		{http.MethodGet, "/api/ml-pipeline/download/job-1"},
		{http.MethodGet, "/api/ml-pipeline/stream/job-1"},
	} {
		policy, lookup := server.routePolicies.LookupRoutePolicy(route.method, route.path)
		if lookup != auth.RouteFound || policy.Permission != auth.PermMlPipeline {
			t.Errorf("ML inventory %s %s: lookup=%v policy=%+v", route.method, route.path, lookup, policy)
		}
	}
}

func TestMLPipelineAvailabilityFollowsRouteRegistration(t *testing.T) {
	disabled, disabledCfg := setupRouteInventoryServerWithConfig(t, func(cfg *config.Config) {
		cfg.MLPipelineEnabled = false
	})
	if disabledCfg.MLPipelineAvailable {
		t.Fatal("MLPipelineAvailable = true, want false while the feature is disabled")
	}
	if disabledCfg.MLPipelineUnavailableReason == "" {
		t.Fatal("MLPipelineUnavailableReason is empty while the feature is disabled")
	}
	if _, lookup := disabled.routePolicies.LookupRoutePolicy(http.MethodGet, "/api/ml-pipeline/jobs"); lookup != auth.RouteNotFound {
		t.Errorf("disabled ML route lookup = %v, want RouteNotFound", lookup)
	}
	if policy, lookup := disabled.routePolicies.LookupRoutePolicy(http.MethodGet, "/api/ml-pipeline/availability"); lookup != auth.RouteFound || policy.Permission != auth.PermMlPipeline {
		t.Errorf("disabled availability route = %v with policy %+v, want RouteFound under PermMlPipeline", lookup, policy)
	}

	_, enabledCfg := setupRouteInventoryServerWithConfig(t)
	if !enabledCfg.MLPipelineAvailable || enabledCfg.MLPipelineUnavailableReason != "" {
		t.Errorf("enabled config = %+v, want available with an empty reason", enabledCfg)
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
		{http.MethodGet, "/api/mcp/servers", auth.PermMcpRead},
		{http.MethodPost, "/api/mcp/servers", auth.PermMcpManage},
	} {
		policy, result := server.routePolicies.LookupRoutePolicy(test.method, test.path)
		if result != auth.RouteFound || policy.Permission != test.permission {
			t.Errorf("%s %s: lookup=%v permission=%q, want %q", test.method, test.path, result, policy.Permission, test.permission)
		}
	}
	rawGlobal := "/api/router/config/global/raw"
	if policy, result := server.routePolicies.LookupRoutePolicy(http.MethodGet, rawGlobal); result != auth.RouteFound ||
		policy.AuditMode != auth.AuditRequired || policy.AuditAction == "" {
		t.Errorf("GET %s audit policy=%+v lookup=%v, want a required audit", rawGlobal, policy, result)
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

func TestOutboundDashboardRoutesRevalidateBeforeUse(t *testing.T) {
	server := setupRouteInventoryServer(t)
	for _, path := range []string{
		"/api/router/api/v1/routing/preview",
		"/api/router/api/v1/plugins/rag/preview",
	} {
		policy, lookup := server.routePolicies.LookupRoutePolicy(http.MethodPost, path)
		if lookup != auth.RouteFound || !policy.Revalidate || policy.MaxBodyBytes == 0 || policy.AuditAction == "" {
			t.Errorf("POST %s policy=%+v lookup=%v, want bounded body, live revalidation, and audit metadata", path, policy, lookup)
		}
	}
	for _, test := range []struct {
		path, action string
	}{
		{"/api/models/discover", "model.discover"},
		{"/api/tools/web-search", "tools.web_search"},
		{"/api/tools/open-web", "tools.open_web"},
		{"/api/tools/weather", "tools.weather"},
		{"/api/tools/fetch-raw", "tools.fetch_raw"},
		{"/api/topology/test-query", "topology.test_query"},
		{"/api/mcp/servers/server-1/test", "mcp.server.test"},
		{"/api/mcp/servers/test", "mcp.server.test"},
	} {
		policy, lookup := server.routePolicies.LookupRoutePolicy(http.MethodPost, test.path)
		if lookup != auth.RouteFound || !policy.Revalidate || policy.AuditMode != auth.AuditRequired || policy.AuditAction != test.action {
			t.Errorf("POST %s policy=%+v lookup=%v, want live revalidation and %q audit", test.path, policy, lookup, test.action)
		}
	}
	policy, lookup := server.routePolicies.LookupRoutePolicy(http.MethodDelete, "/api/mcp/servers/server-1")
	if lookup != auth.RouteFound || !policy.Revalidate || policy.AuditMode != auth.AuditRequired || policy.AuditAction != "mcp.server.delete" {
		t.Errorf("DELETE /api/mcp/servers/server-1 policy=%+v lookup=%v", policy, lookup)
	}
}

func TestDashboardProductionRoutePermissionsGrantAndRevokeIndependently(t *testing.T) {
	server, cfg := setupRouteInventoryServerWithConfig(t)
	store, err := auth.NewStore(cfg.AuthDBPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	service := auth.NewService(store, cfg.JWTSecret, cfg.JWTExpiryHours)
	const password = "production-route-permissions-test"
	hash, err := service.HashPassword(password)
	if err != nil {
		t.Fatal(err)
	}
	user, err := store.CreateUser(t.Context(), "route-permissions@example.test", "Route Permissions", hash, auth.RoleRead, "active")
	if err != nil {
		t.Fatal(err)
	}
	token, _, err := service.Login(t.Context(), user.Email, password)
	if err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite3", cfg.AuthDBPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	routes := []struct{ method, path, permission string }{
		{http.MethodGet, "/api/router/config/all", auth.PermConfigRead},
		{http.MethodGet, "/api/router/api/v1/observability/replays/record-1", auth.PermReplayRead},
		{http.MethodPost, "/api/router/api/v1/observability/outcomes", auth.PermFeedbackSubmit},
		{http.MethodPost, "/api/router/v1/chat/completions", auth.PermInferenceRun},
		{http.MethodPost, "/api/ml-pipeline/train", auth.PermMlPipeline},
		{http.MethodPost, "/api/mcp/servers", auth.PermMcpManage},
	}
	setPermission := func(permission string, allowed bool) {
		t.Helper()
		value := 0
		if allowed {
			value = 1
		}
		if _, err := db.ExecContext(t.Context(), `INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,?)
ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed=excluded.allowed`, user.ID, permission, value); err != nil {
			t.Fatal(err)
		}
	}
	for _, route := range routes {
		setPermission(route.permission, false)
	}
	authorized := auth.AuthenticateRequest(service, server.routePolicies)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	status := func(route struct{ method, path, permission string }) int {
		t.Helper()
		request := httptest.NewRequest(route.method, route.path, strings.NewReader(`{}`))
		request.Header.Set("Authorization", "Bearer "+token)
		response := httptest.NewRecorder()
		authorized.ServeHTTP(response, request)
		return response.Code
	}
	for _, granted := range routes {
		setPermission(granted.permission, true)
		for _, route := range routes {
			want := http.StatusForbidden
			if route.permission == granted.permission {
				want = http.StatusNoContent
			}
			if got := status(route); got != want {
				t.Errorf("grant %s: %s %s status=%d, want %d", granted.permission, route.method, route.path, got, want)
			}
		}
		setPermission(granted.permission, false)
		if got := status(granted); got != http.StatusForbidden {
			t.Errorf("revoke %s: %s %s status=%d", granted.permission, granted.method, granted.path, got)
		}
	}
}

func setupRouteInventoryServer(t *testing.T) *Server {
	server, _ := setupRouteInventoryServerWithConfig(t)
	return server
}

func setupRouteInventoryServerWithConfig(t *testing.T, options ...func(*config.Config)) (*Server, *config.Config) {
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
		MCPEnabled: true, MLPipelineEnabled: true, MLPipelineDataDir: filepath.Join(dir, "ml-pipeline"),
		WorkflowDBPath:         filepath.Join(dir, "workflow.sqlite"),
		ConfigProjectionDBPath: filepath.Join(dir, "projection.sqlite"),
	}
	for _, option := range options {
		option(cfg)
	}
	server := Setup(cfg, setupmode.New(configPath, false))
	t.Cleanup(func() { _ = server.Close() })
	return server, cfg
}
