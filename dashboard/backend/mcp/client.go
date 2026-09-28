package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

// Client is the MCP client (based on official SDK)
type Client struct {
	config *ServerConfig

	mu     sync.RWMutex
	status ServerStatus
	err    error
	tools  []ToolDefinition

	connectedAt *time.Time

	// SDK client
	mcpClient client.MCPClient

	// Cache original InputSchema for filling default values during CallTool
	// key: tool name, value: original InputSchema (map)
	originalSchemas map[string]map[string]interface{}
}

// NewClient creates an MCP client
func NewClient(config *ServerConfig) (*Client, error) {
	return &Client{
		config: config,
		status: StatusDisconnected,
	}, nil
}

// Connect establishes connection
func (c *Client) Connect(ctx context.Context) error {
	log.Printf("[MCP-Client] Connect() called: transport=%s", c.config.Transport)

	c.mu.Lock()
	c.status = StatusConnecting
	c.originalSchemas = make(map[string]map[string]interface{}) // Initialize schema cache
	c.mu.Unlock()

	var mcpClient client.MCPClient
	var err error

	switch c.config.Transport {
	case TransportStdio:
		mcpClient, err = c.createStdioClient(ctx)
	case TransportStreamableHTTP:
		mcpClient, err = c.createStreamableHTTPClient(ctx)
	default:
		return fmt.Errorf("unsupported transport type: %s", c.config.Transport)
	}

	if err != nil {
		log.Printf("[MCP-Client] Failed to create client: error_class=%T", err)
		c.mu.Lock()
		c.status = StatusError
		c.err = err
		c.mu.Unlock()
		return err
	}

	c.mu.Lock()
	c.mcpClient = mcpClient
	c.mu.Unlock()

	// Initialize connection
	log.Printf("[MCP-Client] Initializing connection...")
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{
		Name:    "semantic-router-mcp-client",
		Version: "1.0.0",
	}

	_, err = mcpClient.Initialize(ctx, initReq)
	if err != nil {
		log.Printf("[MCP-Client] Initialize failed: error_class=%T", err)
		_ = mcpClient.Close()
		c.mu.Lock()
		c.status = StatusError
		c.err = err
		c.mcpClient = nil
		c.mu.Unlock()
		return fmt.Errorf("initialize failed: %w", err)
	}
	log.Printf("[MCP-Client] Initialization complete")

	// Get tool list
	log.Printf("[MCP-Client] Listing tools...")
	tools, err := c.ListTools(ctx)
	if err != nil {
		log.Printf("[MCP-Client] Warning: failed to list tools: error_class=%T", err)
	} else {
		log.Printf("[MCP-Client] Got %d tools", len(tools))
	}

	now := time.Now()
	c.mu.Lock()
	c.status = StatusConnected
	c.err = nil
	c.tools = tools
	c.connectedAt = &now
	c.mu.Unlock()

	log.Printf("[MCP-Client] Connect() completed, status: connected, tools: %d", len(tools))
	return nil
}

// createStdioClient creates a Stdio client
func (c *Client) createStdioClient(ctx context.Context) (client.MCPClient, error) {
	log.Printf("[MCP-Client] Creating Stdio client: argument_count=%d", len(c.config.Connection.Args))

	// Build environment variables
	env := os.Environ()
	for k, v := range c.config.Connection.Env {
		env = append(env, fmt.Sprintf("%s=%s", k, v))
	}

	// Prepare options
	opts := []transport.StdioOption{}

	// If working directory needs to be set, use custom command function
	if c.config.Connection.Cwd != "" {
		opts = append(opts, transport.WithCommandFunc(func(ctx context.Context, command string, env []string, args []string) (*exec.Cmd, error) {
			cmd := exec.CommandContext(ctx, command, args...)
			cmd.Env = env
			cmd.Dir = c.config.Connection.Cwd
			return cmd, nil
		}))
	}

	// Use SDK to create Stdio client
	// NewStdioMCPClient automatically starts subprocess
	mcpClient, err := client.NewStdioMCPClientWithOptions(
		c.config.Connection.Command,
		env,
		c.config.Connection.Args,
		opts...,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create stdio client: %w", err)
	}

	return mcpClient, nil
}

// createStreamableHTTPClient creates a Streamable HTTP client
func (c *Client) createStreamableHTTPClient(ctx context.Context) (client.MCPClient, error) {
	log.Printf("[MCP-Client] Creating Streamable HTTP client")

	opts := []transport.StreamableHTTPCOption{}

	// Set timeout
	timeout := 30 * time.Second
	if c.config.Options != nil && c.config.Options.Timeout > 0 {
		timeout = time.Duration(c.config.Options.Timeout) * time.Millisecond
	}
	opts = append(opts, transport.WithHTTPTimeout(timeout))

	// Set custom headers
	if len(c.config.Connection.Headers) > 0 {
		opts = append(opts, transport.WithHTTPHeaders(c.config.Connection.Headers))
	}

	// If custom HTTP Client is needed (e.g., adding OAuth Token)
	if c.config.Security != nil && c.config.Security.OAuth != nil {
		customClient := &http.Client{
			Transport: &oauthTransport{
				base:  http.DefaultTransport,
				oauth: c.config.Security.OAuth,
			},
			Timeout: timeout,
		}
		opts = append(opts, transport.WithHTTPBasicClient(customClient))
	}

	mcpClient, err := client.NewStreamableHttpClient(c.config.Connection.URL, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create streamable http client: %w", err)
	}

	return mcpClient, nil
}

// oauthTransport is a custom HTTP Transport for adding OAuth Token
type oauthTransport struct {
	base  http.RoundTripper
	oauth *OAuthConfig

	// TODO: Implement token cache and refresh
	mu          sync.RWMutex
	accessToken string
}

func (t *oauthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// TODO: Implement OAuth 2.1 token acquisition and refresh logic
	// This is just a placeholder, actual implementation needs client_credentials flow
	t.mu.RLock()
	token := t.accessToken
	t.mu.RUnlock()

	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	return t.base.RoundTrip(req)
}

// Disconnect closes the connection
func (c *Client) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.mcpClient != nil {
		if err := c.mcpClient.Close(); err != nil {
			log.Printf("[MCP-Client] Error closing client: error_class=%T", err)
		}
		c.mcpClient = nil
	}

	c.status = StatusDisconnected
	c.tools = nil
	c.connectedAt = nil

	return nil
}

// transformInputSchema transforms InputSchema, intelligently filtering required fields
// Only keeps truly required parameters (those without default values)
func transformInputSchema(schema map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})

	// Copy basic fields
	for k, v := range schema {
		if k != "required" {
			result[k] = v
		}
	}

	// Get properties
	properties, hasProperties := schema["properties"].(map[string]interface{})
	if !hasProperties {
		return result
	}

	// Get original required list
	originalRequired, _ := schema["required"].([]interface{})
	if len(originalRequired) == 0 {
		return result
	}

	// Filter required: only keep parameters without default values
	newRequired := make([]interface{}, 0)
	for _, req := range originalRequired {
		paramName, ok := req.(string)
		if !ok {
			continue
		}

		// Check if this parameter has a default value
		if propSchema, ok := properties[paramName].(map[string]interface{}); ok {
			if _, hasDefault := propSchema["default"]; hasDefault {
				// Has default value, don't mark as required
				continue
			}
		}

		// No default value, keep as required
		newRequired = append(newRequired, req)
	}

	if len(newRequired) > 0 {
		result["required"] = newRequired
	}

	log.Printf("[MCP-Client] transformInputSchema: original required=%d, filtered required=%d",
		len(originalRequired), len(newRequired))

	return result
}

// coerceArgumentTypes converts arguments to correct types based on schema
// Used to handle incorrectly typed arguments generated by LLM (e.g., string should be array)
func coerceArgumentTypes(args map[string]interface{}, schema map[string]interface{}) map[string]interface{} {
	if args == nil {
		return args
	}

	properties, hasProperties := schema["properties"].(map[string]interface{})
	if !hasProperties {
		return args
	}

	for paramName, value := range args {
		propSchema, ok := properties[paramName].(map[string]interface{})
		if !ok {
			continue
		}

		args[paramName] = coerceArgumentValue(value, propSchema)
	}

	return args
}

func coerceArgumentValue(value interface{}, schema map[string]interface{}) interface{} {
	expectedType, _ := schema["type"].(string)

	switch expectedType {
	case "array":
		return coerceArrayArgument(value)
	case "object":
		if nestedMap, ok := value.(map[string]interface{}); ok {
			return coerceArgumentTypes(nestedMap, schema)
		}
	case "string":
		return coerceStringArgument(value)
	case "number", "integer":
		return coerceNumberArgument(value)
	case "boolean":
		return coerceBooleanArgument(value)
	}

	return value
}

func coerceArrayArgument(value interface{}) interface{} {
	switch v := value.(type) {
	case string:
		if v == "" {
			return []interface{}{}
		}
		return []interface{}{v}
	case nil:
		return []interface{}{}
	default:
		return value
	}
}

func coerceStringArgument(value interface{}) interface{} {
	switch v := value.(type) {
	case []interface{}:
		return coerceStringArrayArgument(v)
	case float64:
		return fmt.Sprintf("%v", v)
	case int:
		return fmt.Sprintf("%d", v)
	case bool:
		return fmt.Sprintf("%v", v)
	default:
		return value
	}
}

func coerceStringArrayArgument(value []interface{}) interface{} {
	if len(value) == 0 {
		return ""
	}
	if len(value) == 1 {
		if str, ok := value[0].(string); ok {
			return str
		}
	}
	return value
}

func coerceNumberArgument(value interface{}) interface{} {
	str, ok := value.(string)
	if !ok {
		return value
	}

	number, err := strconv.ParseFloat(str, 64)
	if err != nil {
		return value
	}
	return number
}

func coerceBooleanArgument(value interface{}) interface{} {
	str, ok := value.(string)
	if !ok {
		return value
	}

	switch strings.ToLower(str) {
	case "true", "1", "yes":
		return true
	case "false", "0", "no", "":
		return false
	default:
		return value
	}
}

// fillDefaultValues fills missing parameters with default values based on original Schema
// Important: Only fills parameters with explicitly defined default values in schema
// No longer generates empty values for parameters without default to avoid API rejecting unknown/deprecated parameters
func fillDefaultValues(args map[string]interface{}, schema map[string]interface{}) map[string]interface{} {
	if args == nil {
		args = make(map[string]interface{})
	}

	properties, hasProperties := schema["properties"].(map[string]interface{})
	if !hasProperties {
		return args
	}

	// Iterate all properties (not just required), fill parameters with default values
	for paramName, propSchemaRaw := range properties {
		propSchema, ok := propSchemaRaw.(map[string]interface{})
		if !ok {
			continue
		}

		// If parameter already exists
		existingValue, exists := args[paramName]
		if exists {
			// If object type, need to recursively check nested fields
			paramType, _ := propSchema["type"].(string)
			if paramType == "object" {
				if existingMap, ok := existingValue.(map[string]interface{}); ok {
					// Recursively fill fields with default values in nested objects
					args[paramName] = fillDefaultValues(existingMap, propSchema)
				}
			}
			continue
		}

		// Only fill parameters with explicit default values
		if defaultValue, hasDefault := propSchema["default"]; hasDefault {
			args[paramName] = defaultValue
		}
		// Note: No longer generate empty values for parameters without default
		// This avoids sending parameters that API doesn't recognize
	}

	return args
}

// ListTools retrieves the tool list
func (c *Client) ListTools(ctx context.Context) ([]ToolDefinition, error) {
	c.mu.RLock()
	mcpClient := c.mcpClient
	c.mu.RUnlock()

	if mcpClient == nil {
		return nil, fmt.Errorf("not connected")
	}

	log.Printf("[MCP-Client] Calling tools/list...")
	result, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		log.Printf("[MCP-Client] tools/list failed: error_class=%T", err)
		return nil, err
	}

	// Convert to our types
	tools := make([]ToolDefinition, 0, len(result.Tools))
	for _, t := range result.Tools {
		inputSchema, _ := json.Marshal(t.InputSchema)

		// Parse original schema
		var schemaMap map[string]interface{}
		if err := json.Unmarshal(inputSchema, &schemaMap); err == nil {
			if properties, ok := schemaMap["properties"]; ok {
				if propsMap, ok := properties.(map[string]interface{}); ok {
					log.Printf("[MCP-Client]   properties count: %d", len(propsMap))
				}
			}

			// Cache original schema for filling default values during CallTool
			c.mu.Lock()
			if c.originalSchemas == nil {
				c.originalSchemas = make(map[string]map[string]interface{})
			}
			c.originalSchemas[t.Name] = schemaMap
			c.mu.Unlock()

			// Transform schema: intelligently filter required
			transformedSchema := transformInputSchema(schemaMap)
			transformedJSON, _ := json.Marshal(transformedSchema)

			// Use transformed schema
			inputSchema = transformedJSON

		}

		tools = append(tools, ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: inputSchema,
		})
	}

	log.Printf("[MCP-Client] Got %d tools", len(tools))
	return tools, nil
}

// CallTool calls a tool
func (c *Client) CallTool(ctx context.Context, name string, arguments json.RawMessage) (*CallToolResult, error) {
	log.Printf("[MCP-Client] CallTool() called: argument_bytes=%d", len(arguments))

	c.mu.RLock()
	mcpClient := c.mcpClient
	originalSchema := c.originalSchemas[name]
	c.mu.RUnlock()

	if mcpClient == nil {
		log.Printf("[MCP-Client] CallTool() error: not connected")
		return nil, fmt.Errorf("not connected")
	}

	// Parse arguments
	var args map[string]interface{}
	if len(arguments) > 0 {
		if err := json.Unmarshal(arguments, &args); err != nil {
			log.Printf("[MCP-Client] CallTool() error: failed to parse arguments: error_class=%T", err)
			return nil, fmt.Errorf("failed to parse arguments: %w", err)
		}
	}
	// Perform type conversion and fill default values based on original schema
	if originalSchema != nil {
		// First convert argument types to schema-required types
		args = coerceArgumentTypes(args, originalSchema)

		// Then fill missing parameters with default values
		args = fillDefaultValues(args, originalSchema)
	}

	// Build request
	req := mcp.CallToolRequest{}
	req.Params.Name = name
	req.Params.Arguments = args

	log.Printf("[MCP-Client] CallTool() sending request to MCP server...")
	result, err := mcpClient.CallTool(ctx, req)
	if err != nil {
		log.Print(mcpCallFailureLogMessage(err))
		return nil, err
	}
	log.Printf("[MCP-Client] CallTool() success, content items: %d, isError: %v", len(result.Content), result.IsError)

	// Convert result
	content := make([]ContentItem, 0, len(result.Content))
	for _, item := range result.Content {
		contentItem := ContentItem{Type: "text"}
		switch v := item.(type) {
		case mcp.TextContent:
			contentItem.Text = v.Text
		case *mcp.TextContent:
			contentItem.Text = v.Text
		default:
			// Convert other types to JSON
			data, _ := json.Marshal(item)
			contentItem.Text = string(data)
		}
		content = append(content, contentItem)
	}

	return &CallToolResult{
		Content: content,
		IsError: result.IsError,
	}, nil
}

func mcpCallFailureLogMessage(err error) string {
	return fmt.Sprintf(
		"[MCP-Client] CallTool() failed: error_class=%T",
		err,
	)
}

// CallToolStreaming calls a tool with streaming
// Note: SDK may not fully support streaming yet, providing compatible implementation here
func (c *Client) CallToolStreaming(ctx context.Context, name string, arguments json.RawMessage, onChunk func(StreamChunk) error) error {
	// Current SDK version may not support true streaming
	// Using synchronous call simulation
	result, err := c.CallTool(ctx, name, arguments)
	if err != nil {
		return onChunk(StreamChunk{Type: "error", Data: "Tool execution failed"})
	}

	// Send completion event
	var data interface{}
	if len(result.Content) > 0 && result.Content[0].Type == "text" {
		data = result.Content[0].Text
	} else {
		data = result.Content
	}

	return onChunk(StreamChunk{Type: "complete", Data: data, Progress: 100})
}

// GetStatus returns the status
func (c *Client) GetStatus() ServerStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.status
}

// GetState returns the complete state
func (c *Client) GetState() *ServerState {
	c.mu.RLock()
	defer c.mu.RUnlock()

	errMsg := ""
	if c.err != nil {
		errMsg = "Connection unavailable"
	}

	return &ServerState{
		Config:      c.config,
		Status:      c.status,
		Error:       errMsg,
		Tools:       c.tools,
		ConnectedAt: c.connectedAt,
	}
}

// GetTools returns the cached tool list
func (c *Client) GetTools() []ToolDefinition {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tools
}

// GetConfig returns the configuration
func (c *Client) GetConfig() *ServerConfig {
	return c.config
}

// ========== Compatible Types ==========

// CallToolResult is the tools/call response result
type CallToolResult struct {
	Content []ContentItem `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// ContentItem represents a content item
type ContentItem struct {
	Type string `json:"type"` // "text" | "image" | "resource"
	Text string `json:"text,omitempty"`
}
