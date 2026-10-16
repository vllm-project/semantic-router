package router

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/server"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

func newMCPShutdownDashboard(t *testing.T) *Server {
	t.Helper()
	tempDir := t.TempDir()
	t.Setenv("VLLM_SR_RECIPE_STORE_DIR", filepath.Join(tempDir, "recipe-store"))
	t.Setenv("VLLM_SR_ACTIVE_RECIPE_DIR", "")
	configPath := filepath.Join(tempDir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	upstream := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(upstream.Close)
	cfg := &config.Config{
		AuthDBPath:             filepath.Join(tempDir, "auth.db"),
		JWTSecret:              "test-secret",
		JWTExpiryHours:         1,
		BootstrapAdminEmail:    "admin@example.com",
		BootstrapAdminPassword: "secret-password",
		BootstrapAdminName:     "Admin",
		StaticDir:              tempDir,
		ConfigFile:             configPath,
		AbsConfigPath:          configPath,
		ConfigDir:              tempDir,
		RouterAPIURL:           upstream.URL,
		RouterMetrics:          upstream.URL + "/metrics",
		EnvoyURL:               upstream.URL,
		MCPEnabled:             true,
		WorkflowDBPath:         filepath.Join(tempDir, "workflow.sqlite"),
		ConfigProjectionDBPath: filepath.Join(tempDir, "config-projection.sqlite"),
		StatusDBPath:           filepath.Join(tempDir, "status.sqlite"),
	}
	dashboard := Setup(cfg, setupmode.New(cfg.AbsConfigPath, cfg.SetupMode))
	t.Cleanup(func() { _ = dashboard.Close() })
	if dashboard.mcpManager == nil {
		t.Fatal("expected MCP manager")
	}
	return dashboard
}

func connectHTTPMCPForShutdown(t *testing.T, dashboard *Server) (string, <-chan string) {
	t.Helper()
	mcpServer := server.NewMCPServer("http-shutdown-test", "1.0.0", server.WithToolCapabilities(false))
	transport := server.NewStreamableHTTPServer(mcpServer, server.WithStateful(true))
	sessionClosed := make(chan string, 1)
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		transport.ServeHTTP(w, r)
		if r.Method == http.MethodDelete {
			sessionClosed <- r.Header.Get("Mcp-Session-Id")
		}
	}))
	t.Cleanup(httpServer.Close)
	serverConfig := &mcp.ServerConfig{
		ID:         "http-shutdown-test",
		Name:       "HTTP shutdown test",
		Transport:  mcp.TransportStreamableHTTP,
		Connection: mcp.ConnectionConfig{URL: httpServer.URL},
	}
	if err := dashboard.mcpManager.AddServer(serverConfig); err != nil {
		t.Fatalf("add HTTP MCP server: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := dashboard.mcpManager.Connect(ctx, serverConfig.ID); err != nil {
		t.Fatalf("connect HTTP MCP server: %v", err)
	}
	return serverConfig.ID, sessionClosed
}
