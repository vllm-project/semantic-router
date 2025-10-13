package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/mark3labs/mcp-go/mcp"
)

const (
	// MCPProtocolVersion is the MCP protocol version supported by this implementation.
	// This version is defined by the Model Context Protocol specification and must match
	// the version expected by MCP servers. The version "2024-11-05" represents the current
	// stable MCP specification release. This is not derived from mcp-go as the library
	// does not export a version constant; instead, it's specified by the protocol itself.
	MCPProtocolVersion = "2024-11-05"
	// MCPClientVersion is the version of this MCP client implementation
	MCPClientVersion = "1.0.0"
)

// HTTPClient implements MCPClient for streamable HTTP transport
type HTTPClient struct {
	*BaseClient
	httpClient *http.Client
	baseURL    string
}

// NewHTTPClient creates a new HTTP MCP client
func NewHTTPClient(name string, config ClientConfig) *HTTPClient {
	baseClient := NewBaseClient(name, config)
	return &HTTPClient{
		BaseClient: baseClient,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		baseURL: config.URL,
	}
}

// Connect establishes connection to the MCP server via HTTP
func (c *HTTPClient) Connect() error {
	if c.connected {
		return nil
	}

	if c.config.URL == "" {
		return fmt.Errorf("URL is required for HTTP transport")
	}

	c.log(LoggingLevelInfo, fmt.Sprintf("Connecting to HTTP endpoint: %s", c.config.URL))

	// Parse the URL to validate it
	parsedURL, err := url.Parse(c.config.URL)
	if err != nil {
		return fmt.Errorf("invalid HTTP URL: %w", err)
	}

	// Validate that it's an HTTP/HTTPS URL
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return fmt.Errorf("HTTP URL must use http or https scheme")
	}

	c.baseURL = c.config.URL

	// Test connectivity with a simple health check or initialize call
	if err := c.testConnection(); err != nil {
		return fmt.Errorf("failed to connect to HTTP endpoint: %w", err)
	}

	c.connected = true
	c.log(LoggingLevelInfo, "Successfully connected to HTTP endpoint")

	// Initialize capabilities
	if err := c.initializeCapabilities(); err != nil {
		c.log(LoggingLevelWarning, fmt.Sprintf("Failed to initialize capabilities: %v", err))
	}

	return nil
}

// testConnection tests the HTTP connection
func (c *HTTPClient) testConnection() error {
	ctx := context.Background()
	if c.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.Timeout)
		defer cancel()
	}

	// Try to initialize the connection
	initRequest := mcp.InitializeRequest{}
	initRequest.Params.ProtocolVersion = MCPProtocolVersion
	initRequest.Params.Capabilities = mcp.ClientCapabilities{
		Roots: &struct {
			ListChanged bool `json:"listChanged,omitempty"`
		}{},
	}
	initRequest.Params.ClientInfo = mcp.Implementation{
		Name:    "http-mcp-client",
		Version: MCPClientVersion,
	}

	_, err := c.sendRequest(ctx, "initialize", initRequest)
	return err
}

// initializeCapabilities initializes capabilities for HTTP client
func (c *HTTPClient) initializeCapabilities() error {
	ctx := context.Background()
	if c.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.Timeout)
		defer cancel()
	}

	// Load tools
	if err := c.loadTools(ctx); err != nil {
		c.log(LoggingLevelWarning, fmt.Sprintf("Failed to load tools: %v", err))
	}

	// Load resources
	if err := c.loadResources(ctx); err != nil {
		c.log(LoggingLevelWarning, fmt.Sprintf("Failed to load resources: %v", err))
	}

	// Load prompts
	if err := c.loadPrompts(ctx); err != nil {
		c.log(LoggingLevelWarning, fmt.Sprintf("Failed to load prompts: %v", err))
	}

	return nil
}

// loadTools loads available tools from the HTTP MCP server
func (c *HTTPClient) loadTools(ctx context.Context) error {
	request := mcp.ListToolsRequest{}

	response, err := c.sendRequest(ctx, "tools/list", request)
	if err != nil {
		return fmt.Errorf("failed to load tools: %w", err)
	}

	var toolsResult mcp.ListToolsResult
	if err := json.Unmarshal(response, &toolsResult); err != nil {
		return fmt.Errorf("failed to parse tools response: %w", err)
	}

	// Apply tool filtering
	filteredTools := FilterTools(toolsResult.Tools, c.config.Options.ToolFilter)

	c.tools = filteredTools
	c.log(LoggingLevelInfo, fmt.Sprintf("Loaded %d tools (filtered from %d)", len(c.tools), len(toolsResult.Tools)))

	return nil
}

// loadResources loads available resources from the HTTP MCP server
func (c *HTTPClient) loadResources(ctx context.Context) error {
	request := mcp.ListResourcesRequest{}

	response, err := c.sendRequest(ctx, "resources/list", request)
	if err != nil {
		return fmt.Errorf("failed to load resources: %w", err)
	}

	var resourcesResult mcp.ListResourcesResult
	if err := json.Unmarshal(response, &resourcesResult); err != nil {
		return fmt.Errorf("failed to parse resources response: %w", err)
	}

	c.resources = resourcesResult.Resources
	c.log(LoggingLevelInfo, fmt.Sprintf("Loaded %d resources", len(c.resources)))

	return nil
}

// loadPrompts loads available prompts from the HTTP MCP server
func (c *HTTPClient) loadPrompts(ctx context.Context) error {
	request := mcp.ListPromptsRequest{}

	response, err := c.sendRequest(ctx, "prompts/list", request)
	if err != nil {
		return fmt.Errorf("failed to load prompts: %w", err)
	}

	var promptsResult mcp.ListPromptsResult
	if err := json.Unmarshal(response, &promptsResult); err != nil {
		return fmt.Errorf("failed to parse prompts response: %w", err)
	}

	c.prompts = promptsResult.Prompts
	c.log(LoggingLevelInfo, fmt.Sprintf("Loaded %d prompts", len(c.prompts)))

	return nil
}

// CallTool calls a tool on the MCP server via HTTP
func (c *HTTPClient) CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.CallToolResult, error) {
	if !c.connected {
		return nil, fmt.Errorf("client not connected")
	}

	// Check if tool exists and is allowed
	var toolFound bool
	for _, tool := range c.tools {
		if tool.Name == name {
			toolFound = true
			break
		}
	}

	if !toolFound {
		return nil, fmt.Errorf("tool '%s' not found or not allowed", name)
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Calling tool via HTTP: %s", name))

	request := mcp.CallToolRequest{}
	request.Params.Name = name
	request.Params.Arguments = arguments

	response, err := c.sendRequest(ctx, "tools/call", request)
	if err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Tool call failed: %v", err))
		return nil, fmt.Errorf("tool call failed: %w", err)
	}

	var result mcp.CallToolResult
	if err := json.Unmarshal(response, &result); err != nil {
		return nil, fmt.Errorf("failed to parse tool call response: %w", err)
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Tool call successful via HTTP: %s", name))
	return &result, nil
}

// ReadResource reads a resource from the MCP server via HTTP
func (c *HTTPClient) ReadResource(ctx context.Context, uri string) (*mcp.ReadResourceResult, error) {
	if !c.connected {
		return nil, fmt.Errorf("client not connected")
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Reading resource via HTTP: %s", uri))

	request := mcp.ReadResourceRequest{}
	request.Params.URI = uri

	response, err := c.sendRequest(ctx, "resources/read", request)
	if err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Resource read failed: %v", err))
		return nil, fmt.Errorf("resource read failed: %w", err)
	}

	var result mcp.ReadResourceResult
	if err := json.Unmarshal(response, &result); err != nil {
		return nil, fmt.Errorf("failed to parse resource response: %w", err)
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Resource read successful via HTTP: %s", uri))
	return &result, nil
}

// GetPrompt gets a prompt from the MCP server via HTTP
func (c *HTTPClient) GetPrompt(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.GetPromptResult, error) {
	if !c.connected {
		return nil, fmt.Errorf("client not connected")
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Getting prompt via HTTP: %s", name))

	request := mcp.GetPromptRequest{}
	request.Params.Name = name
	request.Params.Arguments = convertArgsToStringMap(arguments)

	response, err := c.sendRequest(ctx, "prompts/get", request)
	if err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Prompt get failed: %v", err))
		return nil, fmt.Errorf("prompt get failed: %w", err)
	}

	var result mcp.GetPromptResult
	if err := json.Unmarshal(response, &result); err != nil {
		return nil, fmt.Errorf("failed to parse prompt response: %w", err)
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Prompt get successful via HTTP: %s", name))
	return &result, nil
}

// Ping sends a ping to the MCP server via HTTP
func (c *HTTPClient) Ping(ctx context.Context) error {
	if !c.connected {
		return fmt.Errorf("client not connected")
	}

	_, err := c.sendRequest(ctx, "ping", nil)
	if err != nil {
		return fmt.Errorf("ping failed: %w", err)
	}

	c.log(LoggingLevelDebug, "Ping successful via HTTP")
	return nil
}

// RefreshCapabilities reloads tools, resources, and prompts
func (c *HTTPClient) RefreshCapabilities(_ context.Context) error {
	if !c.connected {
		return fmt.Errorf("client not connected")
	}

	return c.initializeCapabilities()
}

// sendRequest sends an HTTP request to the MCP server
func (c *HTTPClient) sendRequest(ctx context.Context, endpoint string, payload interface{}) ([]byte, error) {
	// Construct URL
	requestURL := fmt.Sprintf("%s/%s", c.baseURL, endpoint)

	// Prepare request body
	var body io.Reader
	if payload != nil {
		jsonData, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}
		body = bytes.NewReader(jsonData)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", requestURL, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	// Add custom headers if provided
	for key, value := range c.config.Headers {
		req.Header.Set(key, value)
	}

	// Make the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check status code
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP request failed with status %d: %s", resp.StatusCode, string(responseBody))
	}

	return responseBody, nil
}

// convertArgsToStringMap converts arguments to string map for compatibility
func convertArgsToStringMap(arguments map[string]interface{}) map[string]string {
	stringArgs := make(map[string]string)
	for k, v := range arguments {
		if str, ok := v.(string); ok {
			stringArgs[k] = str
		} else {
			stringArgs[k] = fmt.Sprintf("%v", v)
		}
	}
	return stringArgs
}

// Close closes the connection to the MCP server
func (c *HTTPClient) Close() error {
	c.connected = false
	c.log(LoggingLevelInfo, "Disconnected from HTTP MCP server")
	return nil
}
