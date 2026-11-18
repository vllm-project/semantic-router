package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	mcpproto "github.com/mark3labs/mcp-go/mcp"

	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

const (
	knownMCPServerID   = "known-server"
	missingMCPServerID = "missing-server"
)

func newConnectedMCPHandler(t *testing.T) (*MCPHandler, *mcp.Manager) {
	t.Helper()
	manager, err := mcp.NewManagerWithInProcessTool(knownMCPServerID, "echo", func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error) {
		return &mcpproto.CallToolResult{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(manager.Close)
	return NewMCPHandler(manager, false), manager
}

func storedMCPServerConfig(id string) *mcp.ServerConfig {
	return &mcp.ServerConfig{
		ID:         id,
		Name:       "Stored server",
		Transport:  mcp.TransportStdio,
		Connection: mcp.ConnectionConfig{Command: "echo"},
	}
}

func TestMCPUnknownServerIDReturnsNotFound(t *testing.T) {
	t.Parallel()
	renamed := storedMCPServerConfig("")
	renamed.Name = "Renamed server"
	tests := []struct {
		name    string
		method  string
		suffix  string
		body    interface{}
		handler func(*MCPHandler) http.HandlerFunc
	}{
		{name: "connect", method: http.MethodPost, suffix: "/connect", handler: (*MCPHandler).ConnectServerHandler},
		{name: "update", method: http.MethodPut, body: renamed, handler: (*MCPHandler).UpdateServerHandler},
		{name: "delete", method: http.MethodDelete, handler: (*MCPHandler).DeleteServerHandler},
		{name: "disconnect", method: http.MethodPost, suffix: "/disconnect", handler: (*MCPHandler).DisconnectServerHandler},
		{name: "status", method: http.MethodGet, suffix: "/status", handler: (*MCPHandler).GetServerStatusHandler},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			handler, manager := newConnectedMCPHandler(t)

			recorder := serveMCPRequest(test.handler(handler), test.method, "/api/mcp/servers/"+missingMCPServerID+test.suffix, test.body)

			if recorder.Code != http.StatusNotFound || strings.TrimSpace(recorder.Body.String()) != "Server not found" {
				t.Fatalf("status=%d body=%q, want 404 \"Server not found\"", recorder.Code, recorder.Body.String())
			}
			if _, ok := manager.GetServer(missingMCPServerID); ok {
				t.Fatal("request created the missing server")
			}
			if _, ok := manager.GetServer(knownMCPServerID); !ok {
				t.Fatal("request removed an unrelated server")
			}
		})
	}
}

func TestMCPDisconnectReportsStateOfExistingServers(t *testing.T) {
	t.Parallel()
	idleManager, err := mcp.NewManager(nil)
	if err != nil {
		t.Fatal(err)
	}
	idle := storedMCPServerConfig("idle-server")
	if addErr := idleManager.AddServer(idle); addErr != nil {
		t.Fatal(addErr)
	}
	connected, _ := newConnectedMCPHandler(t)

	tests := []struct {
		name    string
		handler *MCPHandler
		id      string
	}{
		{name: "already disconnected", handler: NewMCPHandler(idleManager, false), id: idle.ID},
		{name: "connected", handler: connected, id: knownMCPServerID},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := serveMCPRequest(test.handler.DisconnectServerHandler(), http.MethodPost, "/api/mcp/servers/"+test.id+"/disconnect", nil)

			var state mcp.ServerState
			if decodeErr := json.Unmarshal(recorder.Body.Bytes(), &state); decodeErr != nil {
				t.Fatalf("status=%d body=%q: %v", recorder.Code, recorder.Body.String(), decodeErr)
			}
			if recorder.Code != http.StatusOK || state.Status != mcp.StatusDisconnected || state.Config == nil || state.Config.ID != test.id {
				t.Fatalf("status=%d state=%+v, want 200 with the disconnected state of %q", recorder.Code, state, test.id)
			}
		})
	}
}

func TestMCPManagerFailuresOtherThanMissingServerStayInternalErrors(t *testing.T) {
	t.Parallel()
	// A closed store makes every write fail while the server stays in the list.
	newHandler := func(t *testing.T) (*MCPHandler, *mcp.Manager, *mcp.ServerConfig) {
		t.Helper()
		store, err := workflowstore.Open(filepath.Join(t.TempDir(), "workflow.sqlite"))
		if err != nil {
			t.Fatal(err)
		}
		manager, err := mcp.NewManager(store)
		if err != nil {
			t.Fatal(err)
		}
		config := storedMCPServerConfig("stored-server")
		if addErr := manager.AddServer(config); addErr != nil {
			t.Fatal(addErr)
		}
		if closeErr := store.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
		return NewMCPHandler(manager, false), manager, config
	}

	tests := []struct {
		name    string
		method  string
		suffix  string
		handler func(*MCPHandler) http.HandlerFunc
		prepare func(*mcp.Manager)
		want    string
	}{
		{name: "update", method: http.MethodPut, handler: (*MCPHandler).UpdateServerHandler, want: "Update server failed"},
		{name: "delete", method: http.MethodDelete, handler: (*MCPHandler).DeleteServerHandler, want: "Delete server failed"},
		{
			name: "connect", method: http.MethodPost, suffix: "/connect", handler: (*MCPHandler).ConnectServerHandler,
			prepare: (*mcp.Manager).Close, want: "Connect server failed",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			handler, manager, config := newHandler(t)
			if test.prepare != nil {
				test.prepare(manager)
			}

			recorder := serveMCPRequest(test.handler(handler), test.method, "/api/mcp/servers/"+config.ID+test.suffix, config)

			if recorder.Code != http.StatusInternalServerError || strings.TrimSpace(recorder.Body.String()) != test.want {
				t.Fatalf("status=%d body=%q, want 500 %q", recorder.Code, recorder.Body.String(), test.want)
			}
		})
	}
}
