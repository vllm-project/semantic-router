package mcp

import (
	"context"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
)

// MCPClient defines the interface that all MCP client implementations must satisfy
type MCPClient interface {
	// Connection management
	Connect() error
	Close() error
	IsConnected() bool
	Ping(ctx context.Context) error

	// Capability management
	GetTools() []mcp.Tool
	GetResources() []mcp.Resource
	GetPrompts() []mcp.Prompt
	RefreshCapabilities(ctx context.Context) error

	// Tool operations
	CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.CallToolResult, error)

	// Resource operations
	ReadResource(ctx context.Context, uri string) (*mcp.ReadResourceResult, error)

	// Prompt operations
	GetPrompt(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.GetPromptResult, error)

	// Logging
	SetLogHandler(handler func(LoggingLevel, string))
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
	Command       string            `json:"command,omitempty"`
	Args          []string          `json:"args,omitempty"`
	Env           map[string]string `json:"env,omitempty"`
	URL           string            `json:"url,omitempty"`
	Headers       map[string]string `json:"headers,omitempty"`
	TransportType string            `json:"transportType,omitempty"`
	Timeout       time.Duration     `json:"timeout,omitempty"`
	Options       ClientOptions     `json:"options"`
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
