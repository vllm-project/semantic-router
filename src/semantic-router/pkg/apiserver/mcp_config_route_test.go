//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcpconfig"
)

func TestMCPConfigRouteDisabledByDefault(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		configPath: t.TempDir() + "/config.yaml",
		config: &config.RouterConfig{
			MCPConfig: config.MCPConfigServerConfig{Enabled: false},
		},
	}
	if handler := apiServer.buildMCPConfigHandler(); handler != nil {
		t.Fatal("expected nil handler when MCP config server is disabled")
	}
}

func TestMCPConfigRouteLoopbackOnlyBlocksNonLoopback(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		configPath: t.TempDir() + "/config.yaml",
		config: &config.RouterConfig{
			MCPConfig: config.MCPConfigServerConfig{
				Enabled:      true,
				LoopbackOnly: true,
			},
		},
	}
	handler := apiServer.buildMCPConfigHandler()
	if handler == nil {
		t.Fatal("expected handler when MCP config server is enabled")
	}

	req := httptest.NewRequest(http.MethodGet, internalMCPConfigPath, nil)
	req.RemoteAddr = "203.0.113.10:12345"
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusForbidden)
	}
}

func TestMCPConfigRouteLoopbackOnlyAllowsLoopback(t *testing.T) {
	// Do not hit the Streamable MCP handler with a plain GET; it waits for MCP protocol traffic.
	handler := mcpconfig.LoopbackOnly(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, internalMCPConfigPath, nil)
	req.RemoteAddr = "127.0.0.1:12345"
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	if rr.Code == http.StatusForbidden {
		t.Fatalf("loopback request should not be forbidden, got %d", rr.Code)
	}
	if rr.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusNoContent)
	}
}
