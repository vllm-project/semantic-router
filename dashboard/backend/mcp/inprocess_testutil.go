package mcp

import (
	"context"

	mcpsdk "github.com/mark3labs/mcp-go/client"
	mcpproto "github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"
)

// NewManagerWithInProcessTool wires an in-process MCP server for dashboard tests.
// It is not used by production request paths; handler and client tests import it.
func NewManagerWithInProcessTool(serverID, toolName string, call func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error)) (*Manager, error) {
	return newManagerWithInProcessTool(serverID, toolName, call)
}

func newManagerWithInProcessTool(serverID, toolName string, call func(context.Context, mcpproto.CallToolRequest) (*mcpproto.CallToolResult, error)) (*Manager, error) {
	srv := mcpserver.NewMCPServer(serverID, "1.0.0", mcpserver.WithToolCapabilities(true))
	srv.AddTool(mcpproto.NewTool(toolName, mcpproto.WithDescription("in-process test tool")), call)

	inProc, err := mcpsdk.NewInProcessClient(srv)
	if err != nil {
		return nil, err
	}
	initReq := mcpproto.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcpproto.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcpproto.Implementation{Name: "dashboard-mcp-test", Version: "1"}
	if _, initErr := inProc.Initialize(context.Background(), initReq); initErr != nil {
		return nil, initErr
	}

	client, err := NewClient(&ServerConfig{ID: serverID, Name: serverID, Transport: TransportStdio})
	if err != nil {
		return nil, err
	}
	client.mu.Lock()
	client.status = StatusConnected
	client.mcpClient = inProc
	client.mu.Unlock()

	manager, err := NewManager(nil)
	if err != nil {
		return nil, err
	}
	manager.mu.Lock()
	manager.clients[serverID] = client
	manager.configs[serverID] = client.config
	manager.mu.Unlock()
	return manager, nil
}
