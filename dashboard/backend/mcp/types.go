// Package mcp provides MCP (Model Context Protocol) client implementation
// using the official mcp-go SDK (github.com/mark3labs/mcp-go).
package mcp

import (
	"encoding/json"
	"time"
)

// TransportType represents the transport protocol type (official spec 2025-06-18)
type TransportType string

const (
	// TransportStdio represents Stdio transport - local command line
	TransportStdio TransportType = "stdio"
	// TransportStreamableHTTP represents Streamable HTTP transport - officially recommended
	TransportStreamableHTTP TransportType = "streamable-http"
)

// ServerStatus represents the server connection status
type ServerStatus string

const (
	StatusDisconnected ServerStatus = "disconnected"
	StatusConnecting   ServerStatus = "connecting"
	StatusConnected    ServerStatus = "connected"
	StatusError        ServerStatus = "error"
)

// ServerConfig represents MCP server configuration
type ServerConfig struct {
	// Unique identifier (UUID)
	ID string `json:"id" yaml:"id"`

	// Display name
	Name string `json:"name" yaml:"name"`

	// Server description
	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	// Transport protocol type
	Transport TransportType `json:"transport" yaml:"transport"`

	// Connection configuration
	Connection ConnectionConfig `json:"connection" yaml:"connection"`

	// Whether enabled
	Enabled bool `json:"enabled" yaml:"enabled"`

	// Security configuration (OAuth 2.1)
	Security *SecurityConfig `json:"security,omitempty" yaml:"security,omitempty"`

	// Advanced options
	Options *ServerOptions `json:"options,omitempty" yaml:"options,omitempty"`
}

// ConnectionConfig represents connection configuration
type ConnectionConfig struct {
	// === Stdio Transport Configuration ===
	// Executable command
	Command string `json:"command,omitempty" yaml:"command,omitempty"`
	// Command arguments
	Args []string `json:"args,omitempty" yaml:"args,omitempty"`
	// Environment variables
	Env map[string]string `json:"env,omitempty" yaml:"env,omitempty"`
	// Working directory
	Cwd string `json:"cwd,omitempty" yaml:"cwd,omitempty"`

	// === Streamable HTTP Transport Configuration ===
	// Server URL (single endpoint)
	URL string `json:"url,omitempty" yaml:"url,omitempty"`
	// Custom request headers
	Headers map[string]string `json:"headers,omitempty" yaml:"headers,omitempty"`
}

// SecurityConfig represents security configuration (official 2025-06-18 spec)
type SecurityConfig struct {
	// OAuth 2.1 authentication configuration
	OAuth *OAuthConfig `json:"oauth,omitempty" yaml:"oauth,omitempty"`

	// Allowed origins (prevent DNS rebinding attacks)
	AllowedOrigins []string `json:"allowed_origins,omitempty" yaml:"allowed_origins,omitempty"`

	// Whether local-only access
	LocalOnly bool `json:"local_only,omitempty" yaml:"local_only,omitempty"`
}

// OAuthConfig represents OAuth 2.1 authentication configuration
type OAuthConfig struct {
	// OAuth client ID
	ClientID string `json:"client_id" yaml:"client_id"`
	// OAuth client secret (confidential clients only)
	ClientSecret string `json:"client_secret,omitempty" yaml:"client_secret,omitempty"`
	// Authorization endpoint
	AuthorizationURL string `json:"authorization_url" yaml:"authorization_url"`
	// Token endpoint
	TokenURL string `json:"token_url" yaml:"token_url"`
	// Requested scopes
	Scopes []string `json:"scopes,omitempty" yaml:"scopes,omitempty"`
	// Whether to use PKCE (required for public clients)
	UsePKCE bool `json:"use_pkce,omitempty" yaml:"use_pkce,omitempty"`
}

// ServerOptions represents advanced options
type ServerOptions struct {
	// Auto reconnect
	AutoReconnect bool `json:"auto_reconnect,omitempty" yaml:"auto_reconnect,omitempty"`
	// Reconnect interval (ms)
	ReconnectInterval int `json:"reconnect_interval,omitempty" yaml:"reconnect_interval,omitempty"`
	// Request timeout (ms)
	Timeout int `json:"timeout,omitempty" yaml:"timeout,omitempty"`
	// Max retry count
	MaxRetries int `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`
}

// ToolDefinition represents MCP tool definition
type ToolDefinition struct {
	// Tool name
	Name string `json:"name"`
	// Tool description
	Description string `json:"description,omitempty"`
	// Input parameter schema (JSON Schema)
	InputSchema json.RawMessage `json:"inputSchema"`
}

// Tool represents complete tool information (including source server)
type Tool struct {
	ToolDefinition
	// Source MCP server ID
	ServerID string `json:"serverId"`
	// Source MCP server name
	ServerName string `json:"serverName"`
}

// ToolExecuteRequest represents tool execution request
type ToolExecuteRequest struct {
	ServerID  string          `json:"server_id"`
	ToolName  string          `json:"tool_name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolResult represents tool execution result
type ToolResult struct {
	// Whether streaming response
	IsStreaming bool `json:"is_streaming"`
	// Whether execution succeeded
	Success bool `json:"success"`
	// Execution result
	Result interface{} `json:"result,omitempty"`
	// Structured content (if tool defines outputSchema)
	StructuredContent interface{} `json:"structured_content,omitempty"`
	// Error message
	Error string `json:"error,omitempty"`
	// Execution time (ms)
	ExecutionTimeMs int64 `json:"execution_time_ms,omitempty"`
}

// StreamChunk represents streaming response data chunk
type StreamChunk struct {
	// Chunk type: progress | partial | complete | error
	Type string `json:"type"`
	// Data content
	Data interface{} `json:"data"`
	// Progress (0-100)
	Progress int `json:"progress,omitempty"`
}

// ServerState represents server runtime state
type ServerState struct {
	Config      *ServerConfig    `json:"config"`
	Status      ServerStatus     `json:"status"`
	Error       string           `json:"error,omitempty"`
	Tools       []ToolDefinition `json:"tools,omitempty"`
	ConnectedAt *time.Time       `json:"connected_at,omitempty"`
}

// ========== Config File Types ==========

// ServersConfigFile represents MCP server configuration file structure
type ServersConfigFile struct {
	Version         string         `yaml:"version"`
	ProtocolVersion string         `yaml:"protocol_version"`
	Servers         []ServerConfig `yaml:"servers"`
}
