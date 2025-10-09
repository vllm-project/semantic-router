package mcp

import (
	"encoding/json"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
)

// JSONRPCVersion represents the JSON-RPC version
const JSONRPCVersion = "2.0"

// JSONRPCRequest represents a JSON-RPC request
type JSONRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id,omitempty"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

// JSONRPCResponse represents a JSON-RPC response
type JSONRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      interface{}   `json:"id,omitempty"`
	Result  interface{}   `json:"result,omitempty"`
	Error   *JSONRPCError `json:"error,omitempty"`
}

// JSONRPCError represents a JSON-RPC error
type JSONRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// ServerInfo represents MCP server information
type ServerInfo struct {
	Name         string             `json:"name"`
	Version      string             `json:"version"`
	Protocol     ProtocolVersion    `json:"protocolVersion"`
	Capabilities ServerCapabilities `json:"capabilities"`
	Instructions string             `json:"instructions,omitempty"`
}

// ProtocolVersion represents the MCP protocol version
type ProtocolVersion struct {
	Version string `json:"version"`
}

// ServerCapabilities represents what the server can do
type ServerCapabilities struct {
	Logging   *LoggingCapability   `json:"logging,omitempty"`
	Prompts   *PromptsCapability   `json:"prompts,omitempty"`
	Resources *ResourcesCapability `json:"resources,omitempty"`
	Tools     *ToolsCapability     `json:"tools,omitempty"`
	Sampling  *SamplingCapability  `json:"sampling,omitempty"`
}

// LoggingCapability represents logging capabilities
type LoggingCapability struct{}

// PromptsCapability represents prompts capabilities
type PromptsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// ResourcesCapability represents resources capabilities
type ResourcesCapability struct {
	Subscribe   bool `json:"subscribe,omitempty"`
	ListChanged bool `json:"listChanged,omitempty"`
}

// ToolsCapability represents tools capabilities
type ToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// SamplingCapability represents sampling capabilities
type SamplingCapability struct{}

// ClientCapabilities represents client capabilities
type ClientCapabilities struct {
	Roots    *RootsCapability    `json:"roots,omitempty"`
	Sampling *SamplingCapability `json:"sampling,omitempty"`
}

// RootsCapability represents roots capabilities
type RootsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// InitializeRequest represents an initialize request
type InitializeRequest struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ClientCapabilities `json:"capabilities"`
	ClientInfo      ClientInfo         `json:"clientInfo"`
}

// ClientInfo represents client information
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// InitializeResult represents the result of initialization
type InitializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	ServerInfo      ServerInfo         `json:"serverInfo"`
	Instructions    string             `json:"instructions,omitempty"`
}

// Tool represents an MCP tool
type Tool struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	InputSchema ToolSchema `json:"inputSchema"`
}

// ToolSchema represents a tool's input schema
type ToolSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Required   []string               `json:"required,omitempty"`
}

// ToolCallRequest represents a tool call request
type ToolCallRequest struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

// ToolResult represents the result of a tool call
type ToolResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

// ContentBlock represents a content block
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// TextContent represents text content
type TextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ImageContent represents image content
type ImageContent struct {
	Type     string `json:"type"`
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

// Resource represents an MCP resource
type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

// ResourceContents represents resource contents
type ResourceContents struct {
	URI      string         `json:"uri"`
	MimeType string         `json:"mimeType,omitempty"`
	Contents []ContentBlock `json:"contents"`
}

// Prompt represents an MCP prompt
type Prompt struct {
	Name        string           `json:"name"`
	Description string           `json:"description,omitempty"`
	Arguments   []PromptArgument `json:"arguments,omitempty"`
}

// PromptArgument represents a prompt argument
type PromptArgument struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
}

// PromptMessage represents a prompt message
type PromptMessage struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

// GetPromptResult represents the result of getting a prompt
type GetPromptResult struct {
	Description string          `json:"description,omitempty"`
	Messages    []PromptMessage `json:"messages"`
}

// ListToolsRequest represents a list tools request
type ListToolsRequest struct {
	Cursor string `json:"cursor,omitempty"`
}

// ListToolsResult represents the result of listing tools
type ListToolsResult struct {
	Tools      []Tool `json:"tools"`
	NextCursor string `json:"nextCursor,omitempty"`
}

// ListResourcesRequest represents a list resources request
type ListResourcesRequest struct {
	Cursor string `json:"cursor,omitempty"`
}

// ListResourcesResult represents the result of listing resources
type ListResourcesResult struct {
	Resources  []Resource `json:"resources"`
	NextCursor string     `json:"nextCursor,omitempty"`
}

// ListPromptsRequest represents a list prompts request
type ListPromptsRequest struct {
	Cursor string `json:"cursor,omitempty"`
}

// ListPromptsResult represents the result of listing prompts
type ListPromptsResult struct {
	Prompts    []Prompt `json:"prompts"`
	NextCursor string   `json:"nextCursor,omitempty"`
}

// ReadResourceRequest represents a read resource request
type ReadResourceRequest struct {
	URI string `json:"uri"`
}

// ReadResourceResult represents the result of reading a resource
type ReadResourceResult struct {
	Contents []ResourceContents `json:"contents"`
}

// GetPromptRequest represents a get prompt request
type GetPromptRequest struct {
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

// LoggingLevel represents logging levels
type LoggingLevel string

const (
	LoggingLevelDebug     LoggingLevel = "debug"
	LoggingLevelInfo      LoggingLevel = "info"
	LoggingLevelNotice    LoggingLevel = "notice"
	LoggingLevelWarning   LoggingLevel = "warning"
	LoggingLevelError     LoggingLevel = "error"
	LoggingLevelCritical  LoggingLevel = "critical"
	LoggingLevelAlert     LoggingLevel = "alert"
	LoggingLevelEmergency LoggingLevel = "emergency"
)

// LoggingMessage represents a logging message
type LoggingMessage struct {
	Level  LoggingLevel `json:"level"`
	Data   interface{}  `json:"data"`
	Logger string       `json:"logger,omitempty"`
}

// Notification represents an MCP notification
type Notification struct {
	Method string      `json:"method"`
	Params interface{} `json:"params,omitempty"`
}

// Progress represents progress information
type Progress struct {
	ProgressToken interface{} `json:"progressToken,omitempty"`
	Progress      int         `json:"progress,omitempty"`
	Total         int         `json:"total,omitempty"`
}

// PaginatedRequest represents a paginated request
type PaginatedRequest struct {
	Cursor string `json:"cursor,omitempty"`
}

// PaginatedResult represents a paginated result
type PaginatedResult struct {
	NextCursor string `json:"nextCursor,omitempty"`
}

// MCPError represents standard MCP error codes
type MCPError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// Standard MCP error codes
const (
	// JSON-RPC standard errors
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603

	// MCP specific errors
	InvalidTool      = -32000
	InvalidResource  = -32001
	InvalidPrompt    = -32002
	ResourceNotFound = -32003
	ToolNotFound     = -32004
	PromptNotFound   = -32005
)

// Common MCP methods
const (
	MethodInitialize       = "initialize"
	MethodInitialized      = "notifications/initialized"
	MethodPing             = "ping"
	MethodListTools        = "tools/list"
	MethodCallTool         = "tools/call"
	MethodListResources    = "resources/list"
	MethodReadResource     = "resources/read"
	MethodListPrompts      = "prompts/list"
	MethodGetPrompt        = "prompts/get"
	MethodLogging          = "notifications/message"
	MethodToolsChanged     = "notifications/tools/list_changed"
	MethodResourcesChanged = "notifications/resources/list_changed"
	MethodPromptsChanged   = "notifications/prompts/list_changed"
)

// CreateJSONRPCRequest creates a new JSON-RPC request
func CreateJSONRPCRequest(id interface{}, method string, params interface{}) *JSONRPCRequest {
	return &JSONRPCRequest{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Method:  method,
		Params:  params,
	}
}

// CreateJSONRPCResponse creates a new JSON-RPC response
func CreateJSONRPCResponse(id interface{}, result interface{}) *JSONRPCResponse {
	return &JSONRPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Result:  result,
	}
}

// CreateJSONRPCError creates a new JSON-RPC error response
func CreateJSONRPCError(id interface{}, code int, message string, data interface{}) *JSONRPCResponse {
	return &JSONRPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Error: &JSONRPCError{
			Code:    code,
			Message: message,
			Data:    data,
		},
	}
}

// CreateNotification creates a new notification
func CreateNotification(method string, params interface{}) *JSONRPCRequest {
	return &JSONRPCRequest{
		JSONRPC: JSONRPCVersion,
		Method:  method,
		Params:  params,
	}
}

// TextContentBlock creates a text content block
func TextContentBlock(text string) ContentBlock {
	return ContentBlock{
		Type: "text",
		Text: text,
	}
}

// ImageContentBlock creates an image content block
func ImageContentBlock(base64Data, mimeType string) ContentBlock {
	return ContentBlock{
		Type: "image",
		Text: base64Data, // base64-encoded image data
	}
}

// CreateToolResult creates a tool result
func CreateToolResult(content []ContentBlock, isError bool) *ToolResult {
	return &ToolResult{
		Content: content,
		IsError: isError,
	}
}

// Helper functions for OpenAI compatibility

// These types are defined in client.go to avoid duplication

// OpenAITool represents an OpenAI-compatible tool definition
type OpenAITool struct {
	Type     string             `json:"type"`
	Function OpenAIToolFunction `json:"function"`
}

// OpenAIToolFunction represents the function part of an OpenAI tool
type OpenAIToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// OpenAIToolCall represents an OpenAI tool call
type OpenAIToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function OpenAIToolCallFunction `json:"function"`
}

// OpenAIToolCallFunction represents the function call part
type OpenAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ConvertToolToOpenAI converts an MCP tool to OpenAI format
func ConvertToolToOpenAI(tool mcp.Tool) OpenAITool {
	openAITool := OpenAITool{
		Type: "function",
		Function: OpenAIToolFunction{
			Name:        tool.Name,
			Description: tool.Description,
		},
	}

	// Convert input schema to OpenAI parameters format
	if tool.InputSchema.Type != "" || len(tool.InputSchema.Properties) > 0 {
		openAITool.Function.Parameters = map[string]interface{}{
			"type":       "object",
			"properties": tool.InputSchema.Properties,
			"required":   tool.InputSchema.Required,
		}
	}

	return openAITool
}

// ConvertOpenAIToMCPCall converts an OpenAI tool call to MCP format
func ConvertOpenAIToMCPCall(openAICall OpenAIToolCall) (mcp.CallToolRequest, error) {
	var arguments map[string]interface{}
	if openAICall.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(openAICall.Function.Arguments), &arguments); err != nil {
			argStr := openAICall.Function.Arguments
			const maxLen = 200
			if len(argStr) > maxLen {
				argStr = argStr[:maxLen] + "...(truncated)"
			}
			return mcp.CallToolRequest{}, fmt.Errorf("failed to parse arguments (%q): %w", argStr, err)
		}
	}

	// Convert map[string]interface{} to map[string]string for MCP library compatibility
	stringArgs := make(map[string]string)
	for k, v := range arguments {
		if str, ok := v.(string); ok {
			stringArgs[k] = str
		} else {
			stringArgs[k] = fmt.Sprintf("%v", v)
		}
	}

	// Convert string args to map[string]interface{} for MCP API
	interfaceArgs := make(map[string]interface{})
	for k, v := range stringArgs {
		interfaceArgs[k] = v
	}

	req := mcp.CallToolRequest{}
	req.Params.Name = openAICall.Function.Name
	req.Params.Arguments = interfaceArgs
	return req, nil
}

// ConvertMCPResultToOpenAI converts an MCP tool result to OpenAI format
func ConvertMCPResultToOpenAI(result *mcp.CallToolResult) map[string]interface{} {
	if result == nil {
		return map[string]interface{}{
			"content": "No result",
		}
	}

	content := ""
	if len(result.Content) > 0 {
		firstContent := result.Content[0]

		// Use a type switch to match the actual types from mcp-go
		switch c := firstContent.(type) {
		case *mcp.TextContent:
			content = c.Text
		case mcp.TextContent:
			content = c.Text
		case *mcp.ImageContent:
			content = fmt.Sprintf("Image: %s", c.Data)
		case mcp.ImageContent:
			content = fmt.Sprintf("Image: %s", c.Data)
		case *mcp.EmbeddedResource:
			content = fmt.Sprintf("Resource: %v", c.Resource)
		case mcp.EmbeddedResource:
			content = fmt.Sprintf("Resource: %v", c.Resource)
		default:
			// Fallback: try to get string representation
			content = fmt.Sprintf("%v", firstContent)
		}
	}

	return map[string]interface{}{
		"content": content,
		"isError": result.IsError,
	}
}

// FilterTools applies tool filtering based on the configuration
func FilterTools(tools []mcp.Tool, filter ToolFilter) []mcp.Tool {
	if filter.Mode == "" || len(filter.List) == 0 {
		return tools
	}

	filtered := make([]mcp.Tool, 0)

	for _, tool := range tools {
		shouldInclude := false

		if filter.Mode == "allow" {
			// In allow mode, only include tools in the list
			for _, allowedTool := range filter.List {
				if tool.Name == allowedTool {
					shouldInclude = true
					break
				}
			}
		} else if filter.Mode == "block" {
			// In block mode, include all tools except those in the list
			shouldInclude = true
			for _, blockedTool := range filter.List {
				if tool.Name == blockedTool {
					shouldInclude = false
					break
				}
			}
		}

		if shouldInclude {
			filtered = append(filtered, tool)
		}
	}

	return filtered
}

// ValidateAuthToken validates an authentication token
func ValidateAuthToken(token string, validTokens []string) bool {
	if len(validTokens) == 0 {
		return true // No authentication required
	}

	for _, validToken := range validTokens {
		if token == validToken {
			return true
		}
	}

	return false
}

// ExtractAuthToken extracts the authentication token from an Authorization header
func ExtractAuthToken(authHeader string) string {
	if authHeader == "" {
		return ""
	}

	// Support both "Bearer <token>" and "<token>" formats
	if len(authHeader) > 7 && authHeader[:7] == "Bearer " {
		return authHeader[7:]
	}

	return authHeader
}
