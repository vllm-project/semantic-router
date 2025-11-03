package mcp

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

// StdioClient implements MCPClient for stdio transport
type StdioClient struct {
	*BaseClient
	mcpClient client.MCPClient
}

// NewStdioClient creates a new stdio MCP client
func NewStdioClient(name string, config ClientConfig) *StdioClient {
	baseClient := NewBaseClient(name, config)
	return &StdioClient{
		BaseClient: baseClient,
	}
}

// Connect establishes connection to the MCP server via stdio
func (c *StdioClient) Connect() error {
	if c.connected {
		return nil
	}

	c.log(LoggingLevelInfo, "Connecting to MCP server with stdio transport")

	if c.config.Command == "" {
		return fmt.Errorf("command is required for stdio transport")
	}

	// Prepare environment variables
	env := os.Environ()
	for key, value := range c.config.Env {
		env = append(env, fmt.Sprintf("%s=%s", key, value))
	}

	c.log(LoggingLevelInfo, fmt.Sprintf("Starting MCP server: %s %v", c.config.Command, c.config.Args))
	c.log(LoggingLevelDebug, fmt.Sprintf("Environment variables: %v", c.config.Env))

	// Create MCP client using the correct API
	mcpClient, err := client.NewStdioMCPClient(c.config.Command, env, c.config.Args...)
	if err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Failed to create MCP client: %v", err))
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
				c.log(LoggingLevelError, fmt.Sprintf("Panic during MCP initialization: %v", r))
				initDone <- fmt.Errorf("panic during initialization: %v", r)
			}
		}()

		c.log(LoggingLevelDebug, "Starting MCP client...")
		// Note: Connection starts automatically; Start() method is not required

		c.log(LoggingLevelDebug, "Initializing MCP client...")
		// Initialize the client with a simple request (no params needed for basic initialization)
		initResult, err := c.mcpClient.Initialize(ctx, mcp.InitializeRequest{})
		if err != nil {
			c.log(LoggingLevelError, fmt.Sprintf("Failed to initialize MCP client: %v", err))
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

// loadCapabilities loads tools, resources, and prompts from the MCP server
func (c *StdioClient) loadCapabilities(ctx context.Context) error {
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
func (c *StdioClient) loadTools(ctx context.Context) error {
	if c.mcpClient == nil {
		return fmt.Errorf("client not connected")
	}

	toolsResult, err := c.mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return fmt.Errorf("failed to list tools: %w", err)
	}

	// Apply tool filtering
	filteredTools := FilterTools(toolsResult.Tools, c.config.Options.ToolFilter)

	c.tools = filteredTools
	c.log(LoggingLevelInfo, fmt.Sprintf("Loaded %d tools (filtered from %d)", len(c.tools), len(toolsResult.Tools)))

	return nil
}

// loadResources loads available resources from the MCP server
func (c *StdioClient) loadResources(ctx context.Context) error {
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
func (c *StdioClient) loadPrompts(ctx context.Context) error {
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

// CallTool calls a tool on the MCP server
func (c *StdioClient) CallTool(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.CallToolResult, error) {
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

	// Validate that the tool hasn't been flagged as malicious
	c.log(LoggingLevelDebug, fmt.Sprintf("Calling tool: %s with arguments: %+v", name, arguments))

	// Log the type of each argument for debugging
	for key, value := range arguments {
		c.log(LoggingLevelDebug, fmt.Sprintf("Argument %s: type=%T, value=%+v", key, value, value))
	}

	// For debugging, keep the original arguments as-is for the MCP call
	// The MCP library should handle the conversion properly
	callReq := mcp.CallToolRequest{}
	callReq.Params.Name = name
	callReq.Params.Arguments = arguments

	result, err := c.mcpClient.CallTool(ctx, callReq)
	if err != nil {
		c.log(LoggingLevelError, fmt.Sprintf("Tool call failed for %s: %v", name, err))
		c.log(LoggingLevelError, fmt.Sprintf("Arguments were: %+v", arguments))

		// Log detailed argument types for debugging
		for key, value := range arguments {
			c.log(LoggingLevelError, fmt.Sprintf("Failed arg %s: type=%T, value=%+v", key, value, value))
		}

		return nil, fmt.Errorf("tool call failed: %w", err)
	}

	c.log(LoggingLevelDebug, fmt.Sprintf("Tool call successful: %s", name))
	if result != nil && result.Content != nil {
		c.log(LoggingLevelDebug, fmt.Sprintf("Tool result content length: %d items", len(result.Content)))
	}
	return result, nil
}

// ReadResource reads a resource from the MCP server
func (c *StdioClient) ReadResource(ctx context.Context, uri string) (*mcp.ReadResourceResult, error) {
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
func (c *StdioClient) GetPrompt(ctx context.Context, name string, arguments map[string]interface{}) (*mcp.GetPromptResult, error) {
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
func (c *StdioClient) Ping(ctx context.Context) error {
	if c.mcpClient == nil {
		return fmt.Errorf("client not connected")
	}

	err := c.mcpClient.Ping(ctx)
	return err
}

// RefreshCapabilities reloads tools, resources, and prompts
func (c *StdioClient) RefreshCapabilities(ctx context.Context) error {
	if !c.connected {
		return fmt.Errorf("client not connected")
	}

	return c.loadCapabilities(ctx)
}

// Close closes the connection to the MCP server
func (c *StdioClient) Close() error {
	var err error

	// Close MCP client if it exists
	if c.mcpClient != nil {
		err = c.mcpClient.Close()
		c.mcpClient = nil
	}

	c.connected = false
	c.log(LoggingLevelInfo, "Disconnected from MCP server")

	return err
}
