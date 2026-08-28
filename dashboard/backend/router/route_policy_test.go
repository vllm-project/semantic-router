package router

import (
	"net/http"
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
	defer func() {
		if err := server.Close(); err != nil {
			t.Fatalf("close server: %v", err)
		}
	}()

	contracts := server.routePolicies.Contracts()
	if len(contracts) < 80 {
		t.Fatalf("route contracts = %d, want at least 80", len(contracts))
	}

	for _, contract := range contracts {
		if err := auth.ValidateRouteContract(contract); err != nil {
			t.Fatalf("invalid contract %q: %v", contract.Pattern, err)
		}
		for _, policy := range contract.Policies {
			if policy.Public {
				continue
			}
			if !slices.Contains(auth.AllPermissions, policy.Permission) {
				t.Errorf("%s %s uses unknown permission %q", policy.Method, contract.Pattern, policy.Permission)
			}
			if policy.AuditMode != auth.AuditNone && policy.AuditAction == "" {
				t.Errorf("%s %s has no audit action", policy.Method, contract.Pattern)
			}
		}
	}
}

func TestDashboardRoutePoliciesKeepSecurityDomainsIndependent(t *testing.T) {
	server := setupRouteInventoryServer(t)
	defer func() { _ = server.Close() }()

	tests := []struct {
		method     string
		path       string
		permission string
	}{
		{method: http.MethodPost, path: "/api/router/v1/chat/completions", permission: auth.PermInferenceRun},
		{method: http.MethodPost, path: "/api/router/v1/router/outcomes", permission: auth.PermFeedbackSubmit},
		{method: http.MethodGet, path: "/api/router/v1/router_replay/replay-1", permission: auth.PermReplayRead},
		{method: http.MethodPost, path: "/api/router/config/deploy", permission: auth.PermConfigDeploy},
		{method: http.MethodPost, path: "/api/mcp/tools/execute", permission: auth.PermToolsUse},
		{method: http.MethodPatch, path: "/api/admin/users/user-1", permission: auth.PermUsersManage},
	}
	for _, test := range tests {
		policy, result := server.routePolicies.LookupRoutePolicy(test.method, test.path)
		if result != auth.RouteFound {
			t.Fatalf("%s %s lookup = %v", test.method, test.path, result)
		}
		if policy.Permission != test.permission {
			t.Fatalf("%s %s permission = %q, want %q", test.method, test.path, policy.Permission, test.permission)
		}
	}

	if _, result := server.routePolicies.LookupRoutePolicy(http.MethodPost, "/api/router/v1/unknown"); result != auth.RouteNotFound {
		t.Fatalf("unknown Router API lookup = %v, want %v", result, auth.RouteNotFound)
	}
	if _, result := server.routePolicies.LookupRoutePolicy(http.MethodGet, "/api/unknown"); result != auth.RouteNotFound {
		t.Fatalf("unknown Dashboard API lookup = %v, want %v", result, auth.RouteNotFound)
	}
}

func setupRouteInventoryServer(t *testing.T) *Server {
	t.Helper()

	tempDir := t.TempDir()
	staticDir := filepath.Join(tempDir, "static")
	if err := os.MkdirAll(staticDir, 0o755); err != nil {
		t.Fatalf("mkdir static dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(staticDir, "index.html"), []byte("ok"), 0o644); err != nil {
		t.Fatalf("write index: %v", err)
	}
	configPath := filepath.Join(tempDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg := &config.Config{
		Port:                   "19000",
		AuthDBPath:             filepath.Join(tempDir, "auth.db"),
		JWTSecret:              "route-inventory-secret",
		JWTExpiryHours:         1,
		StaticDir:              staticDir,
		ConfigFile:             configPath,
		AbsConfigPath:          configPath,
		ConfigDir:              tempDir,
		RouterAPIURL:           "http://127.0.0.1:18080",
		RouterMetrics:          "http://127.0.0.1:19190/metrics",
		MCPEnabled:             true,
		OpenClawEnabled:        true,
		OpenClawDataDir:        filepath.Join(tempDir, "openclaw"),
		WorkflowDBPath:         filepath.Join(tempDir, "workflow.sqlite"),
		ConfigProjectionDBPath: filepath.Join(tempDir, "projection.sqlite"),
		EvaluationDBPath:       filepath.Join(tempDir, "evaluation.sqlite"),
		EvaluationEnabled:      true,
		MLPipelineEnabled:      true,
		PythonPath:             "python3",
	}
	return Setup(cfg, setupmode.New(configPath, false))
}
