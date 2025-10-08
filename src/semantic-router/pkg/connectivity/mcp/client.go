package mcp

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

// Client wraps the mark3labs/mcp-go client with our configuration
type Client struct {
	name       string
	config     ClientConfig
	mcpClient  client.MCPClient
	tools      []mcp.Tool
	resources  []mcp.Resource
	prompts    []mcp.Prompt
	logHandler func(LoggingLevel, string)
	connected  bool
	// SSE-specific fields
	sseConn   *http.Response
	sseClient *http.Client
}

// Legacy types moved to interface.go - keeping for backward compatibility

// NewLegacyClient creates a legacy MCP client wrapper (deprecated - use factory instead)
func NewLegacyClient(name string, config ClientConfig) *Client {
	return &Client{
		name:      name,
		config:    config,
		connected: false,
		logHandler: func(level LoggingLevel, message string) {
			log.Printf("[%s] %s: %s", level, name, message)
		},
	}
}

// Connect establishes connection to the MCP server
func (c *Client) Connect() error {
	if c.connected {
		return nil
	}

	c.log(LoggingLevelInfo, fmt.Sprintf("Connecting to MCP server with transport: %s", c.determineTransportType()))

	switch c.determineTransportType() {
	case "stdio":
		return c.connectStdio()
	case "sse":
		return c.connectSSE()
	case "http":
		return c.connectHTTP()
	default:
		return fmt.Errorf("unsupported transport type: %s", c.determineTransportType())
	}
}

// connectStdio connects using stdio transport
func (c *Client) connectStdio() error {
	if c.config.Command == "" {
		return fmt.Errorf("command is required for stdio transport")
	}

	// Prepare command
	cmd := exec.Command(c.config.Command, c.config.Args...)

	// Set environment variables
	cmd.Env = os.Environ()
	for key, value := range c.config.Env {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
	}

	// Prepare environment variables
	env := os.Environ()
	for key, value := range c.config.Env {
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}

	// Create MCP client using the correct API
	mcpClient, err := client.NewStdioMCPClient(c.config.Command, env, c.config.Args...)
	if err != nil {
		return fmt.Errorf("failed to create stdio MCP client: %w", err)
	}

	c.mcpClient = mcpClient

	// Initialize the connection with proper context handling
	ctx := context.Background()
	timeout := 30 * time.Second // default timeout
	if c.config.Timeout > 0 {
		timeout = c.config.Timeout
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Use a channel to handle the initialization asynchronously
	initDone := make(chan error, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				initDone <- fmt.Errorf("panic during initialization: %v", r)
			}
		}()

		// Start the client (required before Initialize)
		// NOTE: Start() method removed in mcp-go v0.9.0 - connection starts automatically
		// if err := c.mcpClient.Start(ctx); err != nil {
		// 	initDone <- fmt.Errorf("failed to start MCP client: %w", err)
		// 	return
		// }

		// Initialize the client with a simple request (no params needed for basic initialization)
		initResult, err := c.mcpClient.Initialize(ctx, mcp.InitializeRequest{})
		if err != nil {
			initDone <- fmt.Errorf("failed to initialize MCP client: %w", err)
			return
		}

		c.log(LoggingLevelInfo, fmt.Sprintf("Initialized MCP client: %s v%s", initResult.ServerInfo.Name, initResult.ServerInfo.Version))
		initDone <- nil
	}()

	// Wait for initialization to complete or timeout
	select {
	case err := <-initDone:
		if err != nil {
			c.connected = false
			return err
		}
		c.connected = true
	case <-ctx.Done():
		c.connected = false
		return fmt.Errorf("timeout connecting to MCP server after %v", timeout)
	}

	// Load available capabilities in background
	go func() {
		capCtx, capCancel := context.WithTimeout(context.Background(), timeout)
		defer capCancel()

		if err := c.loadCapabilities(capCtx); err != nil {
			c.log(LoggingLevelWarning, fmt.Sprintf("Failed to load capabilities: %v", err))
		}
	}()

	return nil
}

// connectSSE connects using SSE transport
func (c *Client) connectSSE() error {
	if c.config.URL == "" {
		return fmt.Errorf("URL is required for SSE transport")
	}

	c.log(LoggingLevelInfo, fmt.Sprintf("Connecting to SSE endpoint: %s", c.config.URL))

	// For SSE transport, we need to use HTTP client to connect to the SSE endpoint
	// The mark3labs/mcp-go library primarily focuses on stdio transport
	// For now, we'll create a basic HTTP connection to the SSE endpoint
	// This is a simplified implementation that can be enhanced

	// Parse the URL to validate it
	parsedURL, err := url.Parse(c.config.URL)
	if err != nil {
		return fmt.Errorf("invalid SSE URL: %w", err)
	}

	// Validate that it's an HTTP/HTTPS URL
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return fmt.Errorf("SSE URL must use http or https scheme")
	}

	// Create HTTP client with timeout
	httpClient := &http.Client{}
	if c.config.Timeout > 0 {
		httpClient.Timeout = c.config.Timeout
	}

	// Create request for SSE connection
	req, err := http.NewRequest("GET", c.config.URL, nil)
	if err != nil {
		return fmt.Errorf("failed to create SSE request: %w", err)
	}

	// Set SSE headers
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Connection", "keep-alive")

	// Add custom headers if provided
	for key, value := range c.config.Headers {
		req.Header.Set(key, value)
	}

	// Make the request
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to SSE endpoint: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return fmt.Errorf("SSE connection failed with status: %d", resp.StatusCode)
	}

	// Verify content type
	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/event-stream") {
		resp.Body.Close()
		return fmt.Errorf("invalid content type for SSE: %s", contentType)
	}

	c.log(LoggingLevelInfo, "Successfully connected to SSE endpoint")
	c.connected = true

	// For a complete SSE implementation, we would need to:
	// 1. Parse SSE events from the response body
	// 2. Handle reconnection logic
	// 3. Implement bidirectional communication
	// 4. Process MCP protocol messages over SSE

	// Store the connection for later use
	c.sseConn = resp
	c.sseClient = httpClient

	// Start a goroutine to handle incoming SSE events
	go c.handleSSEEvents()

	c.log(LoggingLevelInfo, "SSE transport connected and event handling started")

	return nil
}

// handleSSEEvents processes incoming SSE events
func (c *Client) handleSSEEvents() {
	if c.sseConn == nil {
		c.log(LoggingLevelError, "No SSE connection to handle events")
		return
	}

	defer c.sseConn.Body.Close()

	scanner := bufio.NewScanner(c.sseConn.Body)
	var eventType, eventData string

	for scanner.Scan() {
		line := scanner.Text()

		// Parse SSE format
		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			eventData = strings.TrimPrefix(line, "data: ")
		} else if line == "" {
			// Empty line indicates end of event
			if eventData != "" {
				c.processSSEEvent(eventType, eventData)
				// Reset for next event
				eventType = ""
				eventData = ""
			}
		}
	}

	if err := scanner.Err(); err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Error reading SSE stream: %v", err))
		c.connected = false
	}
}

// processSSEEvent handles a complete SSE event
func (c *Client) processSSEEvent(eventType, data string) {
	c.log(LoggingLevelDebug, fmt.Sprintf("Received SSE event - Type: %s, Data: %s", eventType, data))

	// In a full MCP implementation, you would:
	// 1. Parse the JSON-RPC message from the data
	// 2. Handle different MCP message types (responses, notifications, etc.)
	// 3. Update local state (tools, resources, prompts)
	// 4. Trigger appropriate callbacks

	// For now, we just log the event
	if data != "" {
		c.log(LoggingLevelInfo, fmt.Sprintf("SSE Event: %s", data))
	}
}

// connectHTTP connects using HTTP transport (placeholder)
func (c *Client) connectHTTP() error {
	// For now, HTTP connections would need additional implementation
	c.log(LoggingLevelWarning, "HTTP transport not fully implemented yet")
	return fmt.Errorf("HTTP transport not implemented")
}

// loadCapabilities loads tools, resources, and prompts from the MCP server
func (c *Client) loadCapabilities(ctx context.Context) error {
	// Load tools
	if err := c.loadTools(ctx); err != nil {
		return fmt.Errorf("failed to load tools: %w", err)
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

// loadTools loads available tools from the MCP server
func (c *Client) loadTools(ctx context.Context) error {
	if c.mcpClient == nil {
		return fmt.Errorf("client not connected")
	}

	toolsResult, err := c.mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list tools: %w", err)
	}

	// Apply tool filtering
	c.tools = FilterTools(toolsResult.Tools, c.config.Options.ToolFilter)

	c.log(LoggingLevelInfo, fmt.Sprintf("Loaded %d tools (filtered from %d)", len(c.tools), len(toolsResult.Tools)))

	return nil
}

// loadResources loads available resources from the MCP server
func (c *Client) loadResources(ctx context.Context) error {
	if c.mcpClient == nil {
		return fmt.Errorf("client not connected")
	}

	resourcesResult, err := c.mcpClient.ListResources(ctx, mcp.ListResourcesRequest{})
	if err != nil {
		return fmt.Errorf("failed to list resources: %w", err)
	}

	c.resources = resourcesResult.Resources
	c.log(LoggingLevelInfo, fmt.Sprintf("Loaded %d resources", len(c.resources)))

	return nil
}

// loadPrompts loads available prompts from the MCP server
func (c *Client) loadPrompts(ctx context.Context) error {
	if c.mcpClient == nil {
		return fmt.Errorf("client not connected")
	}

	promptsResult, err := c.mcpClient.ListPrompts(ctx, mcp.ListPromptsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list prompts: %w", err)
	}

	c.prompts = promptsResult.Prompts
	c.log(LoggingLevelInfo, fmt.Sprintf("Loaded %d prompts", len(c.prompts)))

	return nil
}

// GetTools returns the available tools
func (c *Client) GetTools() []mcp.Tool {
	return c.tools
}

// GetResources returns the available resources
func (c *Client) GetResources() []mcp.Resource {
	return c.resources
}

// GetPrompts returns the available prompts
func (c *Client) GetPrompts() []mcp.Prompt {
	return c.prompts
}

// CallTool calls a tool on the MCP server
func (c *Client) CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.CallToolResult, error) {
	if c.mcpClient == nil {
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

	c.log(LoggingLevelDebug, fmt.Sprintf("Calling tool: %s", name))

	// Convert map[string]interface{} to map[string]string for MCP library compatibility
	stringArgs := make(map[string]string)
	for k, v := range arguments {
		if str, ok := v.(string); ok {
			stringArgs[k] = str
		} else {
			stringArgs[k] = fmt.Sprintf("%v", v)
		}
	}

	callReq := mcp.CallToolRequest{}
	callReq.Params.Name = name
	callReq.Params.Arguments = arguments

	result, err := c.mcpClient.CallTool(ctx, callReq)

	if err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Tool call failed: %v", err))
		return nil, fmt.Errorf("tool call failed: %w", err)
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Tool call successful: %s", name))
	return result, nil
}

// ReadResource reads a resource from the MCP server
func (c *Client) ReadResource(ctx context.Context, uri string) (*mcp.ReadResourceResult, error) {
	if c.mcpClient == nil {
		return nil, fmt.Errorf("client not connected")
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Reading resource: %s", uri))

	readReq := mcp.ReadResourceRequest{}
	readReq.Params.URI = uri

	result, err := c.mcpClient.ReadResource(ctx, readReq)

	if err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Resource read failed: %v", err))
		return nil, fmt.Errorf("resource read failed: %w", err)
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Resource read successful: %s", uri))
	return result, nil
}

// GetPrompt gets a prompt from the MCP server
func (c *Client) GetPrompt(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.GetPromptResult, error) {
	if c.mcpClient == nil {
		return nil, fmt.Errorf("client not connected")
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Getting prompt: %s", name))

	// Convert map[string]interface{} to map[string]string for MCP library compatibility
	stringArgs := make(map[string]string)
	for k, v := range arguments {
		if str, ok := v.(string); ok {
			stringArgs[k] = str
		} else {
			stringArgs[k] = fmt.Sprintf("%v", v)
		}
	}

	getPromptReq := mcp.GetPromptRequest{}
	getPromptReq.Params.Name = name
	getPromptReq.Params.Arguments = stringArgs

	result, err := c.mcpClient.GetPrompt(ctx, getPromptReq)

	if err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Prompt get failed: %v", err))
		return nil, fmt.Errorf("prompt get failed: %w", err)
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Prompt get successful: %s", name))
	return result, nil
}

// Ping sends a ping to the MCP server
func (c *Client) Ping(ctx context.Context) error {
	if c.mcpClient == nil {
		return fmt.Errorf("client not connected")
	}

	err := c.mcpClient.Ping(ctx)
	return err
}

// Close closes the connection to the MCP server
func (c *Client) Close() error {
	var err error

	// Close MCP client if it exists
	if c.mcpClient != nil {
		err = c.mcpClient.Close()
		c.mcpClient = nil
	}

	// Close SSE connection if it exists
	if c.sseConn != nil {
		c.sseConn.Body.Close()
		c.sseConn = nil
	}

	c.connected = false
	c.log(LoggingLevelInfo, "Disconnected from MCP server")

	return err
}

// IsConnected returns whether the client is connected
func (c *Client) IsConnected() bool {
	return c.connected && c.mcpClient != nil
}

// SetLogHandler sets the log handler function
func (c *Client) SetLogHandler(handler func(LoggingLevel, string)) {
	c.logHandler = handler
}

// log writes a log message using the configured handler
func (c *Client) log(level LoggingLevel, message string) {
	if c.logHandler != nil {
		c.logHandler(level, message)
	}
}

// determineTransportType determines the transport type from configuration
func (c *Client) determineTransportType() string {
	if c.config.TransportType != "" {
		return c.config.TransportType
	}

	if c.config.Command != "" {
		return "stdio"
	}

	if c.config.URL != "" {
		if strings.Contains(c.config.URL, "/sse") {
			return "sse"
		}
		return "http"
	}

	return "stdio"
}

// RefreshCapabilities reloads tools, resources, and prompts
func (c *Client) RefreshCapabilities(ctx context.Context) error {
	if !c.connected {
		return fmt.Errorf("client not connected")
	}

	return c.loadCapabilities(ctx)
}

// sendSSEMessage sends a message over SSE transport (typically via HTTP POST)
func (c *Client) sendSSEMessage(ctx context.Context, message []byte) error {
	if c.sseClient == nil {
		return fmt.Errorf("SSE client not connected")
	}

	// For SSE transport, messages are typically sent via HTTP POST to a companion endpoint
	// The message endpoint is usually the same URL with /message appended or similar
	messageURL := c.config.URL
	if strings.HasSuffix(messageURL, "/sse") {
		messageURL = strings.TrimSuffix(messageURL, "/sse") + "/message"
	} else {
		messageURL += "/message"
	}

	req, err := http.NewRequestWithContext(ctx, "POST", messageURL, bytes.NewReader(message))
	if err != nil {
		return fmt.Errorf("failed to create message request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")

	// Add custom headers if provided
	for key, value := range c.config.Headers {
		req.Header.Set(key, value)
	}

	resp, err := c.sseClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send message: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("message request failed with status: %d", resp.StatusCode)
	}

	c.log(LoggingLevelDebug, "Successfully sent message over SSE transport")
	return nil
}
