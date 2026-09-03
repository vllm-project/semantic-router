package mcp

import (
	"context"
	"fmt"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
)

func toolCallFailureLogMessage(toolName string, err error) string {
	return fmt.Sprintf("Tool call failed: tool=%s, error_class=%T", toolName, err)
}

// ClientLifecycle manages an MCP client's connection state.
type ClientLifecycle interface {
	Connect() error
	Close() error
	IsConnected() bool
	Ping(ctx context.Context) error
}

// CapabilityProvider exposes and refreshes an MCP server's advertised capabilities.
type CapabilityProvider interface {
	GetTools() []mcp.Tool
	GetResources() []mcp.Resource
	GetPrompts() []mcp.Prompt
	RefreshCapabilities(ctx context.Context) error
}

// RequestExecutor performs MCP tool, resource, and prompt requests.
type RequestExecutor interface {
	CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.CallToolResult, error)
	ReadResource(ctx context.Context, uri string) (*mcp.ReadResourceResult, error)
	GetPrompt(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.GetPromptResult, error)
}

// LogHandlerSetter configures delivery of MCP client log messages.
type LogHandlerSetter interface {
	SetLogHandler(handler func(LoggingLevel, string))
}

// MCPClient defines the complete contract that all MCP client implementations must satisfy.
// The composed interfaces preserve the original public method set while allowing consumers
// to depend on a narrower capability when the complete client contract is unnecessary.
type MCPClient interface {
	ClientLifecycle
	CapabilityProvider
	RequestExecutor
	LogHandlerSetter
}

// BaseClient provides common functionality for all client implementations
type BaseClient struct {
	name       string
	config     ClientConfig
	tools      []mcp.Tool
	resources  []mcp.Resource
	prompts    []mcp.Prompt
	logHandler func(LoggingLevel, string)
	connected  bool
}

// NewBaseClient creates a new base client
func NewBaseClient(name string, config ClientConfig) *BaseClient {
	return &BaseClient{
		name:      name,
		config:    config,
		connected: false,
		logHandler: func(_ LoggingLevel, message string) {
			// Default log handler - can be overridden
		},
	}
}

// GetTools returns the available tools
func (c *BaseClient) GetTools() []mcp.Tool {
	return c.tools
}

// GetResources returns the available resources
func (c *BaseClient) GetResources() []mcp.Resource {
	return c.resources
}

// GetPrompts returns the available prompts
func (c *BaseClient) GetPrompts() []mcp.Prompt {
	return c.prompts
}

// IsConnected returns whether the client is connected
func (c *BaseClient) IsConnected() bool {
	return c.connected
}

// SetLogHandler sets the log handler function
func (c *BaseClient) SetLogHandler(handler func(LoggingLevel, string)) {
	c.logHandler = handler
}

// log writes a log message using the configured handler
func (c *BaseClient) log(level LoggingLevel, message string) {
	if c.logHandler != nil {
		c.logHandler(level, message)
	}
}

// ClientConfig represents client configuration
type ClientConfig struct {
	Command          string            `json:"command,omitempty"`
	Args             []string          `json:"args,omitempty"`
	Env              map[string]string `json:"env,omitempty"`
	URL              string            `json:"url,omitempty"`
	Headers          map[string]string `json:"headers,omitempty"`
	TransportType    string            `json:"transportType,omitempty"`
	Timeout          time.Duration     `json:"timeout,omitempty"`
	MaxResponseBytes int64             `json:"maxResponseBytes,omitempty"`
	Options          ClientOptions     `json:"options"`
}

// ToolFilter represents tool filtering configuration
type ToolFilter struct {
	Mode string   `json:"mode"` // "allow" or "block"
	List []string `json:"list"`
}

// ClientOptions represents client options
type ClientOptions struct {
	PanicIfInvalid bool       `json:"panicIfInvalid"`
	LogEnabled     bool       `json:"logEnabled"`
	AuthTokens     []string   `json:"authTokens"`
	ToolFilter     ToolFilter `json:"toolFilter"`
}

// TransportType represents the transport type for MCP communication.
// Supported values: "stdio" for stdin/stdout and "streamable-http" for HTTP transport.
// Note: "http" is also accepted as an alias for "streamable-http" for convenience.
type TransportType string

const (
	TransportStdio          TransportType = "stdio"
	TransportStreamableHTTP TransportType = "streamable-http"
)
