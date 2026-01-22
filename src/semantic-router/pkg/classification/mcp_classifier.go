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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const (
	// DefaultMCPThreshold is the default confidence threshold for MCP classification.
	// For multi-class classification, a value of 0.5 means that a predicted class must have at least 50% confidence
	// to be selected. Adjust this threshold as needed for your use case.
	DefaultMCPThreshold = 0.5
)

// MCPClassificationResult holds the classification result with routing information from MCP server
type MCPClassificationResult struct {
	Class        int
	Confidence   float32
	CategoryName string
	Model        string // Model recommended by MCP server
	UseReasoning *bool  // Whether to use reasoning (nil means use default)
}

// MCPCategoryInitializer initializes MCP connection for category classification
type MCPCategoryInitializer interface {
	Init(cfg *config.RouterConfig) error
	Close() error
}

// MCPCategoryInference performs classification via MCP
type MCPCategoryInference interface {
	Classify(ctx context.Context, text string) (candle_binding.ClassResult, error)
	ClassifyWithProbabilities(ctx context.Context, text string) (candle_binding.ClassResultWithProbs, error)
	ListCategories(ctx context.Context) (*CategoryMapping, error)
}

// MCPCategoryClassifier implements both MCPCategoryInitializer and MCPCategoryInference.
//
// Protocol Contract:
// This client relies on the MCP server to respect the protocol defined in the
// github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcp/api package.
//
// The MCP server must implement these tools:
//  1. list_categories - Returns api.ListCategoriesResponse
//  2. classify_text - Returns api.ClassifyResponse or api.ClassifyWithProbabilitiesResponse
//
// The MCP server controls both classification AND routing decisions. When the server returns
// "model" and "use_reasoning" in the classification response, the router will use those values.
// If not provided, the router falls back to the default_model configuration.
//
// For detailed type definitions and examples, see the api package documentation.
type MCPCategoryClassifier struct {
	client   mcpclient.MCPClient
	toolName string
	config   *config.RouterConfig
}

// Init initializes the MCP client connection
func (m *MCPCategoryClassifier) Init(cfg *config.RouterConfig) error {
	if cfg == nil {
		return fmt.Errorf("config is nil")
	}

	// Validate MCP configuration
	if !cfg.MCPCategoryModel.Enabled {
		return fmt.Errorf("MCP category classifier is not enabled")
	}

	// Store config
	m.config = cfg

	// Create MCP client configuration
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

	// Set timeout if specified
	if cfg.TimeoutSeconds > 0 {
		mcpConfig.Timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	// Create MCP client
	client, err := mcpclient.NewClient("category_classifier", mcpConfig)
	if err != nil {
		return fmt.Errorf("failed to create MCP client: %w", err)
	}

	// Connect to MCP server
	if err := client.Connect(); err != nil {
		return fmt.Errorf("failed to connect to MCP server: %w", err)
	}

	m.client = client

	// Discover classification tool
	if err := m.discoverClassificationTool(); err != nil {
		client.Close()
		return fmt.Errorf("failed to discover classification tool: %w", err)
	}

	logging.Infof("Successfully initialized MCP category classifier with tool '%s'", m.toolName)
	return nil
}

// discoverClassificationTool finds the appropriate classification tool from available MCP tools
func (m *MCPCategoryClassifier) discoverClassificationTool() error {
	// If tool name is explicitly specified, use it
	if m.config.ToolName != "" {
		m.toolName = m.config.ToolName
		logging.Infof("Using explicitly configured tool: %s", m.toolName)
		return nil
	}

	// Otherwise, auto-discover by listing available tools
	tools := m.client.GetTools()
	if len(tools) == 0 {
		return fmt.Errorf("no tools available from MCP server")
	}

	// Look for classification-related tools by common names
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
				logging.Infof("Auto-discovered classification tool: %s - %s", m.toolName, tool.Description)
				return nil
			}
		}
	}

	// If no common name found, look for tools that mention "classif" in name or description
	for _, tool := range tools {
		lowerName := strings.ToLower(tool.Name)
		lowerDesc := strings.ToLower(tool.Description)
		if strings.Contains(lowerName, "classif") || strings.Contains(lowerDesc, "classif") {
			m.toolName = tool.Name
			logging.Infof("Auto-discovered classification tool by pattern match: %s - %s", m.toolName, tool.Description)
			return nil
		}
	}

	// Log available tools for debugging
	var toolNames []string
	for _, tool := range tools {
		toolNames = append(toolNames, tool.Name)
	}
	return fmt.Errorf("no classification tool found among available tools: %v", toolNames)
}

// Close closes the MCP client connection
func (m *MCPCategoryClassifier) Close() error {
	if m.client != nil {
		return m.client.Close()
	}
	return nil
}

// Classify performs category classification via MCP
func (m *MCPCategoryClassifier) Classify(ctx context.Context, text string) (candle_binding.ClassResult, error) {
	if m.client == nil {
		return candle_binding.ClassResult{}, fmt.Errorf("MCP client not initialized")
	}

	// Prepare arguments for MCP tool call
	arguments := map[string]interface{}{
		"text": text,
	}

	// Call MCP tool
	result, err := m.client.CallTool(ctx, m.toolName, arguments)
	if err != nil {
		return candle_binding.ClassResult{}, fmt.Errorf("MCP tool call failed: %w", err)
	}

	// Check for errors in result
	if result.IsError {
		return candle_binding.ClassResult{}, fmt.Errorf("MCP tool returned error: %v", result.Content)
	}

	// Parse response from first content block
	if len(result.Content) == 0 {
		return candle_binding.ClassResult{}, fmt.Errorf("MCP tool returned empty content")
	}

	// Extract text content
	var responseText string
	firstContent := result.Content[0]
	if textContent, ok := mcp.AsTextContent(firstContent); ok {
		responseText = textContent.Text
	} else {
		return candle_binding.ClassResult{}, fmt.Errorf("MCP tool returned non-text content")
	}

	// Parse JSON response using the API type
	var response api.ClassifyResponse
	if err := json.Unmarshal([]byte(responseText), &response); err != nil {
		return candle_binding.ClassResult{}, fmt.Errorf("failed to parse MCP response: %w", err)
	}

	return candle_binding.ClassResult{
		Class:      response.Class,
		Confidence: response.Confidence,
	}, nil
}

// ClassifyWithProbabilities performs category classification with full probability distribution via MCP
func (m *MCPCategoryClassifier) ClassifyWithProbabilities(ctx context.Context, text string) (candle_binding.ClassResultWithProbs, error) {
	if m.client == nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("MCP client not initialized")
	}

	// Prepare arguments for MCP tool call with probabilities request
	arguments := map[string]interface{}{
		"text":               text,
		"with_probabilities": true,
	}

	// Call MCP tool
	result, err := m.client.CallTool(ctx, m.toolName, arguments)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("MCP tool call failed: %w", err)
	}

	// Check for errors in result
	if result.IsError {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("MCP tool returned error: %v", result.Content)
	}

	// Parse response from first content block
	if len(result.Content) == 0 {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("MCP tool returned empty content")
	}

	// Extract text content
	var responseText string
	firstContent := result.Content[0]
	if textContent, ok := mcp.AsTextContent(firstContent); ok {
		responseText = textContent.Text
	} else {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("MCP tool returned non-text content")
	}

	// Parse JSON response using the API type
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

// ListCategories retrieves the category mapping from the MCP server
func (m *MCPCategoryClassifier) ListCategories(ctx context.Context) (*CategoryMapping, error) {
	if m.client == nil {
		return nil, fmt.Errorf("MCP client not initialized")
	}

	// Call the list_categories tool
	result, err := m.client.CallTool(ctx, "list_categories", map[string]interface{}{})
	if err != nil {
		return nil, fmt.Errorf("MCP list_categories call failed: %w", err)
	}

	// Check for errors in result
	if result.IsError {
		return nil, fmt.Errorf("MCP tool returned error: %v", result.Content)
	}

	// Parse response from first content block
	if len(result.Content) == 0 {
		return nil, fmt.Errorf("MCP tool returned empty content")
	}

	// Extract text content
	var responseText string
	firstContent := result.Content[0]
	if textContent, ok := mcp.AsTextContent(firstContent); ok {
		responseText = textContent.Text
	} else {
		return nil, fmt.Errorf("MCP tool returned non-text content")
	}

	// Parse JSON response using the API type
	var response api.ListCategoriesResponse
	if err := json.Unmarshal([]byte(responseText), &response); err != nil {
		return nil, fmt.Errorf("failed to parse MCP categories response: %w", err)
	}

	// Build CategoryMapping from the list
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

	if len(response.CategorySystemPrompts) > 0 {
		logging.Infof("Loaded %d categories with %d system prompts from MCP server: %v",
			len(response.Categories), len(response.CategorySystemPrompts), response.Categories)
	} else {
		logging.Infof("Loaded %d categories from MCP server: %v", len(response.Categories), response.Categories)
	}

	return mapping, nil
}

// createMCPCategoryInitializer creates an MCP category initializer
func createMCPCategoryInitializer() MCPCategoryInitializer {
	return &MCPCategoryClassifier{}
}

// createMCPCategoryInference creates an MCP category inference from the initializer
func createMCPCategoryInference(initializer MCPCategoryInitializer) MCPCategoryInference {
	if classifier, ok := initializer.(*MCPCategoryClassifier); ok {
		return classifier
	}
	return nil
}

// IsMCPCategoryEnabled checks if MCP-based category classification is properly configured.
// Note: tool_name is optional and will be auto-discovered during initialization if not specified.
func (c *Classifier) IsMCPCategoryEnabled() bool {
	return c.Config.MCPCategoryModel.Enabled
}

// initializeMCPCategoryClassifier initializes the MCP category classification model
func (c *Classifier) initializeMCPCategoryClassifier() error {
	if !c.IsMCPCategoryEnabled() {
		return fmt.Errorf("MCP category classification is not properly configured")
	}

	if c.mcpCategoryInitializer == nil {
		return fmt.Errorf("MCP category initializer is not set")
	}

	if err := c.mcpCategoryInitializer.Init(c.Config); err != nil {
		return fmt.Errorf("failed to initialize MCP category classifier: %w", err)
	}

	// If no in-tree category model is configured and no category mapping exists,
	// load categories from the MCP server
	if c.Config.CategoryModel.ModelID == "" && c.CategoryMapping == nil {
		logging.Infof("Loading category mapping from MCP server...")

		// Create a context with timeout for the list_categories call
		ctx := context.Background()
		if c.Config.TimeoutSeconds > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, time.Duration(c.Config.TimeoutSeconds)*time.Second)
			defer cancel()
		}

		// Get categories from MCP server
		categoryMapping, err := c.mcpCategoryInference.ListCategories(ctx)
		if err != nil {
			return fmt.Errorf("failed to load categories from MCP server: %w", err)
		}

		// Store the category mapping
		c.CategoryMapping = categoryMapping
		logging.Infof("Successfully loaded %d categories from MCP server", c.CategoryMapping.GetCategoryCount())
	}

	logging.Infof("Successfully initialized MCP category classifier")
	return nil
}

// classifyCategoryWithEntropyMCP performs category classification with entropy using MCP
func (c *Classifier) classifyCategoryWithEntropyMCP(text string) (string, float64, entropy.ReasoningDecision, error) {
	if !c.IsMCPCategoryEnabled() {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("MCP category classification is not properly configured")
	}

	if c.mcpCategoryInference == nil {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("MCP category inference is not initialized")
	}

	// Create context with timeout
	ctx := context.Background()
	if c.Config.TimeoutSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(c.Config.TimeoutSeconds)*time.Second)
		defer cancel()
	}

	// Get full probability distribution via MCP
	result, err := c.mcpCategoryInference.ClassifyWithProbabilities(ctx, text)
	if err != nil {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("MCP classification error: %w", err)
	}

	logging.Infof("MCP classification result: class=%d, confidence=%.4f, entropy_available=%t",
		result.Class, result.Confidence, len(result.Probabilities) > 0)

	// Get category names for all classes and translate to generic names when configured
	categoryNames := make([]string, len(result.Probabilities))
	for i := range result.Probabilities {
		if c.CategoryMapping != nil {
			if name, ok := c.CategoryMapping.GetCategoryFromIndex(i); ok {
				categoryNames[i] = c.translateMMLUToGeneric(name)
			} else {
				categoryNames[i] = fmt.Sprintf("unknown_%d", i)
			}
		} else {
			categoryNames[i] = fmt.Sprintf("category_%d", i)
		}
	}

	// Build category reasoning map from decisions configuration
	categoryReasoningMap := make(map[string]bool)
	for _, decision := range c.Config.Decisions {
		useReasoning := false
		if len(decision.ModelRefs) > 0 && decision.ModelRefs[0].UseReasoning != nil {
			useReasoning = *decision.ModelRefs[0].UseReasoning
		}
		categoryReasoningMap[strings.ToLower(decision.Name)] = useReasoning
	}

	// Determine threshold
	threshold := c.Config.MCPCategoryModel.Threshold
	if threshold == 0 {
		threshold = DefaultMCPThreshold
	}

	// Make entropy-based reasoning decision
	entropyStart := time.Now()
	reasoningDecision := entropy.MakeEntropyBasedReasoningDecision(
		result.Probabilities,
		categoryNames,
		categoryReasoningMap,
		float64(threshold),
	)
	entropyLatency := time.Since(entropyStart).Seconds()

	// Calculate entropy value for metrics
	entropyValue := entropy.CalculateEntropy(result.Probabilities)

	// Determine top category for metrics
	topCategory := "none"
	if len(reasoningDecision.TopCategories) > 0 {
		topCategory = reasoningDecision.TopCategories[0].Category
	}

	// Validate probability distribution quality
	probSum := float32(0.0)
	for _, prob := range result.Probabilities {
		probSum += prob
	}

	// Record probability distribution quality checks
	if probSum >= 0.99 && probSum <= 1.01 {
		metrics.RecordProbabilityDistributionQuality("sum_check", "valid")
	} else {
		metrics.RecordProbabilityDistributionQuality("sum_check", "invalid")
		logging.Warnf("MCP probability distribution sum is %.3f (should be ~1.0)", probSum)
	}

	// Check for negative probabilities
	hasNegative := false
	for _, prob := range result.Probabilities {
		if prob < 0 {
			hasNegative = true
			break
		}
	}

	if hasNegative {
		metrics.RecordProbabilityDistributionQuality("negative_check", "invalid")
	} else {
		metrics.RecordProbabilityDistributionQuality("negative_check", "valid")
	}

	// Calculate uncertainty level from entropy value
	entropyResult := entropy.AnalyzeEntropy(result.Probabilities)
	uncertaintyLevel := entropyResult.UncertaintyLevel

	// Record comprehensive entropy classification metrics
	metrics.RecordEntropyClassificationMetrics(
		topCategory,
		uncertaintyLevel,
		entropyValue,
		reasoningDecision.Confidence,
		reasoningDecision.UseReasoning,
		reasoningDecision.DecisionReason,
		topCategory,
		entropyLatency,
	)

	// Check confidence threshold for category determination
	if result.Confidence < threshold {
		// Determine fallback category (default to "other" if not configured)
		fallbackCategory := c.Config.FallbackCategory
		if fallbackCategory == "" {
			fallbackCategory = "other"
		}

		logging.Infof("MCP classification confidence (%.4f) below threshold (%.4f), falling back to category: %s",
			result.Confidence, threshold, fallbackCategory)

		// Record the fallback category as a signal match
		metrics.RecordSignalMatch(config.SignalTypeKeyword, fallbackCategory)

		// Return fallback category instead of empty string to enable proper decision routing
		return fallbackCategory, float64(result.Confidence), reasoningDecision, nil
	}

	// Map class index to category name
	var categoryName string
	var genericCategory string
	if c.CategoryMapping != nil {
		name, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class)
		if ok {
			categoryName = name
			genericCategory = c.translateMMLUToGeneric(name)
		} else {
			categoryName = fmt.Sprintf("category_%d", result.Class)
			genericCategory = categoryName
		}
	} else {
		categoryName = fmt.Sprintf("category_%d", result.Class)
		genericCategory = categoryName
	}

	// Record the category as a signal match
	metrics.RecordSignalMatch(config.SignalTypeKeyword, genericCategory)

	logging.Infof("MCP classified as category: %s (mmlu=%s), reasoning_decision: use=%t, confidence=%.3f, reason=%s",
		genericCategory, categoryName, reasoningDecision.UseReasoning, reasoningDecision.Confidence, reasoningDecision.DecisionReason)

	return genericCategory, float64(result.Confidence), reasoningDecision, nil
}

// withMCPCategory creates an option function for MCP category classifier
func withMCPCategory(mcpInitializer MCPCategoryInitializer, mcpInference MCPCategoryInference) option {
	return func(c *Classifier) {
		c.mcpCategoryInitializer = mcpInitializer
		c.mcpCategoryInference = mcpInference
	}
}
