package classification

import (
	"context"
	"errors"

	"github.com/mark3labs/mcp-go/mcp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	mcpclient "github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcp"
)

type MockMCPClient struct {
	connectError   error
	callToolResult *mcp.CallToolResult
	callToolError  error
	closeError     error
	connected      bool
	getToolsResult []mcp.Tool
}

func (m *MockMCPClient) Connect() error {
	if m.connectError != nil {
		return m.connectError
	}
	m.connected = true
	return nil
}

func (m *MockMCPClient) Close() error {
	if m.closeError != nil {
		return m.closeError
	}
	m.connected = false
	return nil
}

func (m *MockMCPClient) IsConnected() bool { return m.connected }

func (m *MockMCPClient) Ping(ctx context.Context) error { return nil }

func (m *MockMCPClient) GetTools() []mcp.Tool { return m.getToolsResult }

func (m *MockMCPClient) GetResources() []mcp.Resource { return nil }

func (m *MockMCPClient) GetPrompts() []mcp.Prompt { return nil }

func (m *MockMCPClient) RefreshCapabilities(ctx context.Context) error { return nil }

func (m *MockMCPClient) CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.CallToolResult, error) {
	if m.callToolError != nil {
		return nil, m.callToolError
	}
	return m.callToolResult, nil
}

func (m *MockMCPClient) ReadResource(ctx context.Context, uri string) (*mcp.ReadResourceResult, error) {
	return nil, errors.New("not implemented")
}

func (m *MockMCPClient) GetPrompt(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.GetPromptResult, error) {
	return nil, errors.New("not implemented")
}

func (m *MockMCPClient) SetLogHandler(handler func(mcpclient.LoggingLevel, string)) {}

var _ mcpclient.MCPClient = (*MockMCPClient)(nil)

func newTestMCPCategoryClassifier() (*MCPCategoryClassifier, *MockMCPClient, *config.RouterConfig) {
	mockClient := &MockMCPClient{}
	cfg := &config.RouterConfig{}
	cfg.Enabled = true
	cfg.ToolName = "classify_text"
	cfg.TransportType = "stdio"
	cfg.Command = "python"
	cfg.Args = []string{"server_keyword.py"}
	cfg.TimeoutSeconds = 30

	return &MCPCategoryClassifier{}, mockClient, cfg
}

func newMCPTextResult(text string) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		IsError: false,
		Content: []mcp.Content{mcp.TextContent{Type: "text", Text: text}},
	}
}

func newMCPErrorResult(text string) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{mcp.TextContent{Type: "text", Text: text}},
	}
}
