package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

const unsupportedSecurityCanary = "unsupported-security-secret-canary" //nolint:gosec // Canary verifies rejection messages never echo credentials.

func securedServerConfig(id string) *mcp.ServerConfig {
	return &mcp.ServerConfig{
		ID:         id,
		Name:       "Secured server",
		Transport:  mcp.TransportStreamableHTTP,
		Connection: mcp.ConnectionConfig{URL: "http://127.0.0.1:1/mcp"},
		Security: &mcp.SecurityConfig{OAuth: &mcp.OAuthConfig{
			ClientID:     "dashboard",
			ClientSecret: unsupportedSecurityCanary,
			TokenURL:     "https://auth.example.test/token",
		}},
	}
}

func newSecurityTestHandler(t *testing.T, stored *mcp.ServerConfig) (*MCPHandler, *mcp.Manager) {
	t.Helper()
	manager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	if stored != nil {
		// Stands in for a config persisted before the API rejected security settings.
		if err := manager.AddServer(stored); err != nil {
			t.Fatal(err)
		}
	}
	return NewMCPHandler(manager, false), manager
}

func serveMCPRequest(handler http.HandlerFunc, method, path string, body interface{}) *httptest.ResponseRecorder {
	var payload bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&payload).Encode(body)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(method, path, &payload))
	return recorder
}

func assertUnsupportedSecurityMessage(t *testing.T, message, field string) {
	t.Helper()
	if !strings.Contains(message, field) || strings.Contains(message, unsupportedSecurityCanary) {
		t.Fatalf("message must name %s without echoing credentials: %q", field, message)
	}
}

func TestMCPCreateRejectsUnsupportedSecurity(t *testing.T) {
	t.Parallel()
	handler, manager := newSecurityTestHandler(t, nil)
	config := securedServerConfig("create-secured-server")

	recorder := serveMCPRequest(handler.CreateServerHandler(), http.MethodPost, "/api/mcp/servers", config)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	assertUnsupportedSecurityMessage(t, recorder.Body.String(), "security.oauth")
	if _, ok := manager.GetServer(config.ID); ok {
		t.Fatal("server with unsupported security settings was stored")
	}
}

func TestMCPUpdateRejectsUnsupportedSecurityAndKeepsStoredConfig(t *testing.T) {
	t.Parallel()
	config := securedServerConfig("update-server")
	config.Security = nil
	handler, manager := newSecurityTestHandler(t, config)

	update := *config
	update.Name = "Renamed server"
	update.Security = &mcp.SecurityConfig{LocalOnly: true}
	recorder := serveMCPRequest(handler.UpdateServerHandler(), http.MethodPut, "/api/mcp/servers/"+config.ID, &update)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	assertUnsupportedSecurityMessage(t, recorder.Body.String(), "security.local_only")
	if stored, _ := manager.GetServer(config.ID); stored.Name != config.Name || stored.Security != nil {
		t.Fatalf("rejected update changed the stored server: %#v", stored)
	}
}

func TestMCPConnectRefusesStoredUnsupportedSecurity(t *testing.T) {
	t.Parallel()
	config := securedServerConfig("stored-secured-server")
	handler, _ := newSecurityTestHandler(t, config)

	recorder := serveMCPRequest(handler.ConnectServerHandler(), http.MethodPost, "/api/mcp/servers/"+config.ID+"/connect", nil)
	if recorder.Code != http.StatusConflict {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	assertUnsupportedSecurityMessage(t, recorder.Body.String(), "security.oauth")
}

func TestMCPUpdateCanClearStoredUnsupportedSecurity(t *testing.T) {
	t.Parallel()
	config := securedServerConfig("clear-secured-server")
	handler, manager := newSecurityTestHandler(t, config)

	cleared := mcp.RedactedServerConfig(config)
	cleared.Security = &mcp.SecurityConfig{}
	recorder := serveMCPRequest(handler.UpdateServerHandler(), http.MethodPut, "/api/mcp/servers/"+config.ID, cleared)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if stored, _ := manager.GetServer(config.ID); mcp.ValidateSecurity(stored.Security) != nil {
		t.Fatalf("stored security settings were not cleared: %#v", stored.Security)
	}
}

// The collection test path answers a security setting the client cannot
// enforce the way create and update do: 400 with the field names, not a
// test failure envelope (#4338).
func TestMCPConnectionTestRejectsUnsupportedSecurity(t *testing.T) {
	t.Parallel()
	handler, _ := newSecurityTestHandler(t, nil)
	config := securedServerConfig("")
	config.Security = &mcp.SecurityConfig{AllowedOrigins: []string{"https://dashboard.example.test"}}

	recorder := serveMCPRequest(handler.TestConnectionHandler(), http.MethodPost, "/api/mcp/servers/test", config)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	assertUnsupportedSecurityMessage(t, recorder.Body.String(), "security.allowed_origins")
}
