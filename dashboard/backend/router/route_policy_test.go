package router

import (
	"bytes"
	"encoding/json"
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
		BootstrapAdminEmail:    "inventory-admin@example.com",
		BootstrapAdminPassword: "inventory-password",
		BootstrapAdminName:     "Inventory Admin",
		StaticDir:              staticDir,
		ConfigFile:             configPath,
		AbsConfigPath:          configPath,
		ConfigDir:              tempDir,
		RouterAPIURL:           "http://127.0.0.1:18080",
		EnvoyURL:               "http://127.0.0.1:18801",
		RouterMetrics:          "http://127.0.0.1:19190/metrics",
		GrafanaURL:             "http://127.0.0.1:13000",
		JaegerURL:              "http://127.0.0.1:16686",
		PrometheusURL:          "http://127.0.0.1:19090",
		MCPEnabled:             true,
		OpenClawEnabled:        true,
		OpenClawDataDir:        filepath.Join(tempDir, "openclaw"),
		WorkflowDBPath:         filepath.Join(tempDir, "workflow.sqlite"),
		ConfigProjectionDBPath: filepath.Join(tempDir, "projection.sqlite"),
		StatusDBPath:           filepath.Join(tempDir, "status.sqlite"),
		MLPipelineEnabled:      true,
		MLPipelineDataDir:      filepath.Join(tempDir, "ml"),
		PythonPath:             "python3",
	}
	server := Setup(cfg, setupmode.New(configPath, false))
	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Errorf("close server: %v", err)
		}
	})
	return server
}

func TestDashboardRouteInventoryHasCompletePolicies(t *testing.T) {
	server := setupRouteInventoryServer(t)

	contracts := server.RouteContracts()
	if len(contracts) < 120 {
		t.Fatalf("route contracts = %d, want the full Dashboard surface", len(contracts))
	}

	for _, contract := range contracts {
		if err := auth.ValidateRouteContract(contract); err != nil {
			t.Fatalf("invalid contract %q: %v", contract.Pattern, err)
		}
		for _, policy := range contract.Policies {
			for _, permission := range policy.Permissions {
				if !slices.Contains(auth.AllPermissions, permission) {
					t.Errorf("%s %s uses unknown permission %q", policy.Method, contract.Pattern, permission)
				}
			}
			if policy.Public && (strings.HasPrefix(contract.Pattern, "/api/") || strings.HasPrefix(contract.Pattern, "/embedded/")) {
				if !slices.Contains(publicDashboardRoutes, contract.Pattern) {
					t.Errorf("%s %s is public but not in the reviewed public allowlist", policy.Method, contract.Pattern)
				}
			}
			if policy.Revalidate && policy.AuditMode == auth.AuditNone {
				t.Errorf("%s %s revalidates without declaring an audit action", policy.Method, contract.Pattern)
			}
		}
	}
}

// Every unauthenticated route in the protected namespaces must be listed here
// so a new public registration is a reviewed decision, not an accident.
var publicDashboardRoutes = []string{
	"/api/auth/bootstrap/can-register",
	"/api/auth/bootstrap/can-register/{$}",
	"/api/auth/bootstrap/register",
	"/api/auth/bootstrap/register/{$}",
	"/api/auth/invitations/{token}",
	"/api/auth/invitations/{token}/{$}",
	"/api/auth/invitations/{token}/accept",
	"/api/auth/invitations/{token}/accept/{$}",
	"/api/auth/login",
	"/api/auth/login/{$}",
	"/api/auth/logout",
	"/api/auth/logout/{$}",
	"/api/setup/state",
	"/api/status",
	"/api/status/{$}",
	"/embedded/wizmap/assets/",
}

func TestDashboardRoutePoliciesKeepSecurityDomainsIndependent(t *testing.T) {
	server := setupRouteInventoryServer(t)

	tests := []struct {
		method      string
		path        string
		permissions []string
	}{
		{http.MethodPost, "/api/router/v1/chat/completions", []string{auth.PermInferenceRun}},
		{http.MethodPost, "/api/router/api/v1/observability/outcomes", []string{auth.PermFeedbackSubmit}},
		{http.MethodGet, "/api/router/api/v1/observability/replays/replay-1", []string{auth.PermReplayRead}},
		{http.MethodGet, "/api/router/api/v1/observability/audit", []string{auth.PermReplayRead}},
		{http.MethodPost, "/api/router/config/deploy", []string{auth.PermConfigDeploy}},
		{http.MethodPost, "/api/router/config/update", []string{auth.PermConfigWrite}},
		{http.MethodGet, "/api/router/config/all", []string{auth.PermConfigRead}},
		{http.MethodPost, "/api/router/api/v1/plugins/rag/preview", []string{auth.PermEvalRun, auth.PermConfigRead}},
		{http.MethodPut, "/api/router/api/v1/storage/knowledge-bases/example", []string{auth.PermConfigWrite}},
		{http.MethodPost, "/api/mcp/tools/execute", []string{auth.PermToolsUse}},
		{http.MethodPost, "/api/mcp/servers", []string{auth.PermMcpManage}},
		{http.MethodGet, "/api/mcp/servers/server-1/status", []string{auth.PermMcpRead}},
		{http.MethodPatch, "/api/admin/users/user-1", []string{auth.PermUsersManage}},
		{http.MethodGet, "/api/admin/users", []string{auth.PermUsersView}},
		{http.MethodPost, "/api/ml-pipeline/train", []string{auth.PermMlPipeline}},
		{http.MethodPost, "/api/openclaw/provision", []string{auth.PermOpenClaw}},
		{http.MethodGet, "/api/openclaw/next-port", []string{auth.PermOpenClaw}},
		{http.MethodPost, "/api/recipe/probes/lane/variant/validate/", []string{auth.PermTopologyRead}},
		{http.MethodGet, "/api/status/", nil},
		{http.MethodGet, "/api/ml-pipeline/jobs/job-1/", []string{auth.PermMlPipeline}},
		{http.MethodPut, "/api/mcp/servers/tenant%2Fserver", []string{auth.PermMcpManage}},
		{http.MethodPost, "/api/openclaw/rooms/room-1/messages", []string{auth.PermOpenClawRead}},
		{http.MethodGet, "/api/logs", []string{auth.PermLogsRead}},
		{http.MethodPost, "/api/ds/query", []string{auth.PermLogsRead}},
		{http.MethodGet, "/api/traces/trace-1", []string{auth.PermLogsRead}},
		{http.MethodGet, "/embedded/grafana/api/live/ws", []string{auth.PermLogsRead}},
		{http.MethodGet, "/embedded/wizmap/", []string{auth.PermConfigRead}},
		{http.MethodGet, "/api/tools-db", []string{auth.PermToolsUse}},
		{http.MethodPost, "/api/topology/test-query", []string{auth.PermTopologyRead}},
		{http.MethodPost, "/api/recipe/probes/lane/variant/validate", []string{auth.PermTopologyRead}},
		{http.MethodPost, "/api/recipe/probes/lane/variant/run-plan", []string{auth.PermConfigRead}},
		{http.MethodPost, "/api/recipe/import/anything", []string{auth.PermConfigWrite}},
		{http.MethodPost, "/api/recipe/activate", []string{auth.PermConfigDeploy}},
		{http.MethodPost, "/api/recipe/activate/preview", []string{auth.PermConfigDeploy}},
		{http.MethodPost, "/api/sr-bench/v1/runs", []string{auth.PermEvalWrite, auth.PermEvalRun}},
		{http.MethodGet, "/api/sr-bench/v1/runs", []string{auth.PermEvalRead}},
		{http.MethodPost, "/api/sr-bench/v1/comparisons", []string{auth.PermEvalRead}},
		{http.MethodPost, "/api/sr-bench/v1/runs/run-1/cancel", []string{auth.PermEvalRun}},
		{http.MethodPost, "/api/sr-bench/v1/runs/run-1/recover", []string{auth.PermEvalWrite, auth.PermEvalRun}},
		{http.MethodPost, "/api/sr-bench/v1/runs/run-1/candidate-plan", []string{auth.PermEvalWrite}},
		{http.MethodPost, "/api/sr-bench/v1/experiments/exp-0123456789abcdef0123456789abcdef/runs", []string{auth.PermEvalWrite}},
		{http.MethodGet, "/api/sr-bench/v1/datasets/" + strings.Repeat("a", 64) + "/cases", []string{auth.PermEvalRead}},
		{http.MethodPost, "/api/sr-bench/v1/dataset-preparations", []string{auth.PermEvalWrite}},
		{http.MethodGet, "/api/sr-bench/v1/dataset-preparations", []string{auth.PermEvalRead}},
		{http.MethodGet, "/api/sr-bench/v1/dataset-preparations/options", []string{auth.PermEvalRead}},
		{http.MethodGet, "/api/sr-bench/v1/dataset-preparations/prep-0123456789abcdef0123456789abcdef", []string{auth.PermEvalRead}},
	}
	for _, test := range tests {
		policy, result := server.routes.LookupRoutePolicy(test.method, test.path)
		if result != auth.RouteFound {
			t.Fatalf("%s %s lookup = %v", test.method, test.path, result)
		}
		if !slices.Equal(policy.Permissions, test.permissions) {
			t.Fatalf("%s %s permissions = %v, want %v", test.method, test.path, policy.Permissions, test.permissions)
		}
	}

	for _, test := range []struct {
		path   string
		limit  int64
		stream bool
	}{
		{"/api/ml-pipeline/train", 64 << 20, true},
		{"/api/ml-pipeline/benchmark", 32 << 20, true},
		{"/api/ml-pipeline/config", 2 << 20, false},
		{"/api/sr-bench/v1/plans", 8 << 20, false},
	} {
		policy, result := server.routes.LookupRoutePolicy(http.MethodPost, test.path)
		if result != auth.RouteFound || policy.MaxBodyBytes != test.limit || policy.StreamBody != test.stream || !policy.Revalidate {
			t.Fatalf("%s policy = %+v (%v)", test.path, policy, result)
		}
	}

	for _, test := range []struct{ method, path string }{
		{http.MethodPost, "/api/router/v1/unknown"},
		{http.MethodGet, "/api/unknown"},
		{http.MethodGet, "/api/router/config/unknown"},
		{http.MethodGet, "/api/evaluation"},
		{http.MethodGet, "/api/fleet-sim/api/workloads"},
		{http.MethodGet, "/api/services-status"},
		{http.MethodGet, "/embedded/unknown/"},
	} {
		if _, result := server.routes.LookupRoutePolicy(test.method, test.path); result != auth.RouteNotFound {
			t.Fatalf("%s %s lookup = %v, want %v", test.method, test.path, result, auth.RouteNotFound)
		}
	}
	if _, result := server.routes.LookupRoutePolicy(http.MethodDelete, "/api/router/config/all"); result != auth.RouteMethodNotAllowed {
		t.Fatalf("undeclared method lookup = %v, want %v", result, auth.RouteMethodNotAllowed)
	}
}

func TestDashboardServerDeniesUnregisteredProtectedRoutes(t *testing.T) {
	server := setupRouteInventoryServer(t)

	login := httptest.NewRecorder()
	loginBody := bytes.NewBufferString(`{"email":"inventory-admin@example.com","password":"inventory-password"}`)
	server.Handler.ServeHTTP(login, httptest.NewRequest(http.MethodPost, "/api/auth/login", loginBody))
	if login.Code != http.StatusOK {
		t.Fatalf("login status = %d: %s", login.Code, login.Body.String())
	}
	var session struct {
		Token string `json:"token"`
	}
	if err := json.NewDecoder(login.Body).Decode(&session); err != nil {
		t.Fatalf("decode login: %v", err)
	}

	for _, test := range []struct {
		name         string
		method, path string
		token        bool
		want         int
	}{
		{name: "anonymous unknown api path", method: http.MethodGet, path: "/api/unknown", want: http.StatusForbidden},
		{name: "admin unknown api path", method: http.MethodGet, path: "/api/unknown", token: true, want: http.StatusForbidden},
		{name: "admin unknown router path", method: http.MethodPost, path: "/api/router/v1/unknown", token: true, want: http.StatusForbidden},
		{name: "admin retired evaluation path", method: http.MethodGet, path: "/api/evaluation/v1/runs", token: true, want: http.StatusForbidden},
		{name: "admin unknown embedded path", method: http.MethodGet, path: "/embedded/unknown/", token: true, want: http.StatusForbidden},
		{name: "admin undeclared method", method: http.MethodDelete, path: "/api/router/config/all", token: true, want: http.StatusMethodNotAllowed},
		{name: "anonymous registered route", method: http.MethodGet, path: "/api/router/config/all", want: http.StatusUnauthorized},
		{name: "public setup state", method: http.MethodGet, path: "/api/setup/state", want: http.StatusOK},
		{name: "public bootstrap alias", method: http.MethodGet, path: "/api/auth/bootstrap/can-register/", want: http.StatusOK},
		{name: "public status alias", method: http.MethodGet, path: "/api/status/", want: http.StatusOK},
		{name: "anonymous preflight on protected route", method: http.MethodOptions, path: "/api/router/config/all", want: http.StatusUnauthorized},
		{name: "static frontend", method: http.MethodGet, path: "/", want: http.StatusOK},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(test.method, test.path, nil)
			if test.token {
				request.Header.Set("Authorization", "Bearer "+session.Token)
			}
			recorder := httptest.NewRecorder()
			server.Handler.ServeHTTP(recorder, request)
			if recorder.Code != test.want {
				t.Fatalf("status = %d, want %d: %s", recorder.Code, test.want, recorder.Body.String())
			}
		})
	}
}

func TestRouterManagementContractsCoverEveryGatewayPolicy(t *testing.T) {
	t.Parallel()

	gateway := routerManagementContracts(isGatewayProxyPath)
	knowledgeBases := routerManagementContracts(isKnowledgeBasePath)
	if len(knowledgeBases) == 0 || len(gateway) == 0 {
		t.Fatalf("gateway=%d knowledge-base=%d contracts", len(gateway), len(knowledgeBases))
	}
	for _, contract := range append(append([]auth.RouteContract(nil), gateway...), knowledgeBases...) {
		if err := auth.ValidateRouteContract(contract); err != nil {
			t.Fatalf("%s: %v", contract.Pattern, err)
		}
		for _, policy := range contract.Policies {
			if policy.AuditMode == auth.AuditRequired && !strings.HasPrefix(policy.AuditAction, "router.") {
				t.Fatalf("%s %s audit action %q", policy.Method, contract.Pattern, policy.AuditAction)
			}
		}
	}
	if got := managementAuditAction("/api/router/api/v1/storage/response-cache/flush"); got != "router.storage.response_cache.flush" {
		t.Fatalf("audit action = %q", got)
	}
	if got := managementAuditAction("/api/router/api/v1/storage/knowledge-bases/{name}"); got != "router.storage.knowledge_bases" {
		t.Fatalf("audit action = %q", got)
	}
}
