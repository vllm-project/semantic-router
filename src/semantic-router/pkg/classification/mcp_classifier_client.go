package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	mcpclient "github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcp"
	api "github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcp/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Init initializes the MCP client connection.
func (m *MCPCategoryClassifier) Init(cfg *config.RouterConfig) error {
	if cfg == nil {
		return fmt.Errorf("config is nil")
	}

	if !cfg.MCPCategoryModel.Enabled {
		return fmt.Errorf("MCP category classifier is not enabled")
	}

	m.config = cfg
	mcpConfig := mcpclient.ClientConfig{
		TransportType: cfg.TransportType,
		Command:       cfg.Command,
		Args:          cfg.Args,
		Env:           cfg.Env,
		URL:           cfg.URL,
		Options: mcpclient.ClientOptions{
			LogEnabled: true,
		},
	}
	if cfg.TimeoutSeconds > 0 {
		mcpConfig.Timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	client, err := mcpclient.NewClient("category_classifier", mcpConfig)
	if err != nil {
		return fmt.Errorf("failed to create MCP client: %w", err)
	}
	if err := client.Connect(); err != nil {
		return fmt.Errorf("failed to connect to MCP server: %w", err)
	}

	m.client = client
	if err := m.discoverClassificationTool(); err != nil {
		client.Close()
		return fmt.Errorf("failed to discover classification tool: %w", err)
	}

	logging.ComponentEvent("classifier", "mcp_category_classifier_initialized", map[string]interface{}{
		"tool_name":      m.toolName,
		"transport_type": cfg.TransportType,
	})
	return nil
}

// discoverClassificationTool finds the appropriate classification tool from available MCP tools.
func (m *MCPCategoryClassifier) discoverClassificationTool() error {
	if m.config.ToolName != "" {
		m.toolName = m.config.ToolName
		logging.ComponentEvent("classifier", "mcp_category_tool_selected", map[string]interface{}{
			"tool_name":      m.toolName,
			"selection_mode": "configured",
		})
		return nil
	}

	tools := m.client.GetTools()
	if len(tools) == 0 {
		return fmt.Errorf("no tools available from MCP server")
	}

	classificationToolNames := []string{
		"classify_text",
		"classify",
		"categorize",
		"categorize_text",
	}

	for _, toolName := range classificationToolNames {
		for _, tool := range tools {
			if tool.Name == toolName {
				m.toolName = tool.Name
				logging.ComponentEvent("classifier", "mcp_category_tool_selected", map[string]interface{}{
					"tool_name":      m.toolName,
					"selection_mode": "auto_name_match",
				})
				return nil
			}
		}
	}

	for _, tool := range tools {
		lowerName := strings.ToLower(tool.Name)
		lowerDesc := strings.ToLower(tool.Description)
		if strings.Contains(lowerName, "classif") || strings.Contains(lowerDesc, "classif") {
			m.toolName = tool.Name
			logging.ComponentEvent("classifier", "mcp_category_tool_selected", map[string]interface{}{
				"tool_name":      m.toolName,
				"selection_mode": "auto_pattern_match",
			})
			return nil
		}
	}

	var toolNames []string
	for _, tool := range tools {
		toolNames = append(toolNames, tool.Name)
	}
	return fmt.Errorf("no classification tool found among available tools: %v", toolNames)
}

// Close closes the MCP client connection.
func (m *MCPCategoryClassifier) Close() error {
	if m.client != nil {
		return m.client.Close()
	}
	return nil
}

// Classify performs category classification via MCP.
func (m *MCPCategoryClassifier) Classify(ctx context.Context, text string) (candle_binding.ClassResult, error) {
	responseText, err := m.callMCPTextTool(ctx, m.toolName, map[string]interface{}{
		"text": text,
	}, "MCP tool call failed")
	if err != nil {
		return candle_binding.ClassResult{}, err
	}

	var response api.ClassifyResponse
	if err := json.Unmarshal([]byte(responseText), &response); err != nil {
		return candle_binding.ClassResult{}, fmt.Errorf("failed to parse MCP response: %w", err)
	}

	return candle_binding.ClassResult{
		Class:      response.Class,
		Confidence: response.Confidence,
	}, nil
}

// ClassifyWithProbabilities performs category classification with full probability distribution via MCP.
func (m *MCPCategoryClassifier) ClassifyWithProbabilities(ctx context.Context, text string) (candle_binding.ClassResultWithProbs, error) {
	responseText, err := m.callMCPTextTool(ctx, m.toolName, map[string]interface{}{
		"text":               text,
		"with_probabilities": true,
	}, "MCP tool call failed")
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, err
	}

	var response api.ClassifyWithProbabilitiesResponse
	if err := json.Unmarshal([]byte(responseText), &response); err != nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("failed to parse MCP response: %w", err)
	}

	return candle_binding.ClassResultWithProbs{
		Class:         response.Class,
		Confidence:    response.Confidence,
		Probabilities: response.Probabilities,
	}, nil
}

// ListCategories retrieves the category mapping from the MCP server.
func (m *MCPCategoryClassifier) ListCategories(ctx context.Context) (*CategoryMapping, error) {
	responseText, err := m.callMCPTextTool(ctx, "list_categories", map[string]interface{}{}, "MCP list_categories call failed")
	if err != nil {
		return nil, err
	}

	var response api.ListCategoriesResponse
	if err := json.Unmarshal([]byte(responseText), &response); err != nil {
		return nil, fmt.Errorf("failed to parse MCP categories response: %w", err)
	}

	mapping := &CategoryMapping{
		CategoryToIdx:         make(map[string]int),
		IdxToCategory:         make(map[string]string),
		CategorySystemPrompts: response.CategorySystemPrompts,
		CategoryDescriptions:  response.CategoryDescriptions,
	}
	for idx, category := range response.Categories {
		mapping.CategoryToIdx[category] = idx
		mapping.IdxToCategory[fmt.Sprintf("%d", idx)] = category
	}

	logging.ComponentEvent("classifier", "mcp_category_mapping_loaded", map[string]interface{}{
		"tool_name":               m.toolName,
		"categories":              len(response.Categories),
		"system_prompt_count":     len(response.CategorySystemPrompts),
		"categories_with_prompts": len(response.CategorySystemPrompts) > 0,
	})

	return mapping, nil
}

func (m *MCPCategoryClassifier) callMCPTextTool(
	ctx context.Context,
	toolName string,
	arguments map[string]interface{},
	callErrorPrefix string,
) (string, error) {
	if m.client == nil {
		return "", fmt.Errorf("MCP client not initialized")
	}

	result, err := m.client.CallTool(ctx, toolName, arguments)
	if err != nil {
		return "", fmt.Errorf("%s: %w", callErrorPrefix, err)
	}
	if result.IsError {
		return "", fmt.Errorf("MCP tool returned error: %v", result.Content)
	}
	if len(result.Content) == 0 {
		return "", fmt.Errorf("MCP tool returned empty content")
	}

	textContent, ok := mcp.AsTextContent(result.Content[0])
	if !ok {
		return "", fmt.Errorf("MCP tool returned non-text content")
	}
	return textContent.Text, nil
}
