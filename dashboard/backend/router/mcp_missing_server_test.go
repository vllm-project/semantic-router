package router

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

// newMCPRouteClient serves requests through the dashboard's full handler as the bootstrap admin.
func newMCPRouteClient(t *testing.T) (*Server, func(t *testing.T, method, path, body string) *httptest.ResponseRecorder) {
	t.Helper()
	dashboard := newMCPShutdownDashboard(t)

	//nolint:gosec // G117: the fixture password is only sent to this in-memory login handler.
	login, err := json.Marshal(auth.LoginRequest{Email: "admin@example.com", Password: "secret-password"})
	if err != nil {
		t.Fatal(err)
	}
	loginResponse := httptest.NewRecorder()
	dashboard.Handler.ServeHTTP(loginResponse, httptest.NewRequest(http.MethodPost, "/api/auth/login", bytes.NewReader(login)))
	var session auth.LoginResponse
	if loginResponse.Code != http.StatusOK {
		t.Fatalf("login status = %d, body = %s", loginResponse.Code, loginResponse.Body.String())
	}
	if err = json.Unmarshal(loginResponse.Body.Bytes(), &session); err != nil {
		t.Fatal(err)
	}

	return dashboard, func(t *testing.T, method, path, body string) *httptest.ResponseRecorder {
		t.Helper()
		request := httptest.NewRequest(method, path, strings.NewReader(body))
		request.Header.Set("Authorization", "Bearer "+session.Token)
		response := httptest.NewRecorder()
		dashboard.Handler.ServeHTTP(response, request)
		t.Logf("%s %s -> %d %s", method, path, response.Code, strings.TrimSpace(response.Body.String()))
		return response
	}
}

func TestMCPServerRoutesReturnNotFoundForUnknownID(t *testing.T) {
	_, serve := newMCPRouteClient(t)

	const update = `{"name":"Renamed server","transport":"stdio","connection":{"command":"echo"}}`
	tests := []struct{ method, path, body string }{
		{http.MethodPost, "/api/mcp/servers/missing-server/connect", ""},
		{http.MethodPut, "/api/mcp/servers/missing-server", update},
		{http.MethodDelete, "/api/mcp/servers/missing-server", ""},
		{http.MethodPost, "/api/mcp/servers/missing-server/disconnect", ""},
		{http.MethodGet, "/api/mcp/servers/missing-server/status", ""},
	}
	for _, test := range tests {
		t.Run(test.method+" "+test.path, func(t *testing.T) {
			if response := serve(t, test.method, test.path, test.body); response.Code != http.StatusNotFound {
				t.Fatalf("status = %d, want %d", response.Code, http.StatusNotFound)
			}
		})
	}
}

func TestMCPDisconnectRouteKeepsReportingExistingServer(t *testing.T) {
	dashboard, serve := newMCPRouteClient(t)
	idle := &mcp.ServerConfig{
		ID:         "idle-server",
		Name:       "Idle server",
		Transport:  mcp.TransportStdio,
		Connection: mcp.ConnectionConfig{Command: "echo"},
	}
	if err := dashboard.mcpManager.AddServer(idle); err != nil {
		t.Fatal(err)
	}

	response := serve(t, http.MethodPost, "/api/mcp/servers/"+idle.ID+"/disconnect", "")

	var state mcp.ServerState
	if err := json.Unmarshal(response.Body.Bytes(), &state); err != nil {
		t.Fatalf("status = %d, body = %q: %v", response.Code, response.Body.String(), err)
	}
	if response.Code != http.StatusOK || state.Status != mcp.StatusDisconnected || state.Config == nil || state.Config.ID != idle.ID {
		t.Fatalf("status = %d, state = %+v, want 200 with the disconnected state of %q", response.Code, state, idle.ID)
	}
}
