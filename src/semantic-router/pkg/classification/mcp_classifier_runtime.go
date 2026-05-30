package classification

import (
	"context"
	"fmt"
	"strings"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// IsMCPCategoryEnabled checks if MCP-based category classification is properly configured.
// Note: tool_name is optional and will be auto-discovered during initialization if not specified.
func (c *Classifier) IsMCPCategoryEnabled() bool {
	return c.Config.MCPCategoryModel.Enabled
}

// initializeMCPCategoryClassifier initializes the MCP category classification model.
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

	if c.Config.CategoryModel.ModelID == "" && c.CategoryMapping == nil {
		return c.loadMCPCategoryMapping()
	}
	return nil
}

func (c *Classifier) loadMCPCategoryMapping() error {
	toolName := ""
	if classifier, ok := c.mcpCategoryInitializer.(*MCPCategoryClassifier); ok {
		toolName = classifier.toolName
	}
	logging.ComponentDebugEvent("classifier", "mcp_category_mapping_load_started", map[string]interface{}{
		"tool_name": toolName,
	})

	ctx := context.Background()
	if c.Config.TimeoutSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(c.Config.TimeoutSeconds)*time.Second)
		defer cancel()
	}

	categoryMapping, err := c.mcpCategoryInference.ListCategories(ctx)
	if err != nil {
		return fmt.Errorf("failed to load categories from MCP server: %w", err)
	}

	c.CategoryMapping = categoryMapping
	logging.ComponentEvent("classifier", "mcp_category_mapping_attached", map[string]interface{}{
		"categories": c.CategoryMapping.GetCategoryCount(),
	})
	return nil
}

// classifyCategoryWithEntropyMCP performs category classification with entropy using MCP.
func (c *Classifier) classifyCategoryWithEntropyMCP(text string) (string, float64, entropy.ReasoningDecision, error) {
	if !c.IsMCPCategoryEnabled() {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("MCP category classification is not properly configured")
	}
	if c.mcpCategoryInference == nil {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("MCP category inference is not initialized")
	}

	result, err := c.classifyMCPWithProbabilities(text)
	if err != nil {
		return "", 0.0, entropy.ReasoningDecision{}, err
	}

	logging.Infof("MCP classification result: class=%d, confidence=%.4f, entropy_available=%t",
		result.Class, result.Confidence, len(result.Probabilities) > 0)

	categoryNames := c.mcpCategoryNames(result.Probabilities)
	reasoningDecision, entropyLatency := c.makeMCPReasoningDecision(result.Probabilities, categoryNames)
	recordMCPProbabilityMetrics(result.Probabilities, reasoningDecision, entropyLatency)

	threshold := c.mcpCategoryThreshold()
	if result.Confidence < threshold {
		fallbackCategory := c.Config.FallbackCategory
		if fallbackCategory == "" {
			fallbackCategory = "other"
		}

		logging.Infof("MCP classification confidence (%.4f) below threshold (%.4f), falling back to category: %s",
			result.Confidence, threshold, fallbackCategory)
		metrics.RecordSignalMatch(config.SignalTypeKeyword, fallbackCategory)
		return fallbackCategory, float64(result.Confidence), reasoningDecision, nil
	}

	categoryName, genericCategory := c.mcpCategoryNameForClass(result.Class)
	metrics.RecordSignalMatch(config.SignalTypeKeyword, genericCategory)
	logging.Infof("MCP classified as category: %s (mmlu=%s), reasoning_decision: use=%t, confidence=%.3f, reason=%s",
		genericCategory, categoryName, reasoningDecision.UseReasoning, reasoningDecision.Confidence, reasoningDecision.DecisionReason)

	return genericCategory, float64(result.Confidence), reasoningDecision, nil
}

func (c *Classifier) classifyMCPWithProbabilities(text string) (candle_binding.ClassResultWithProbs, error) {
	ctx := context.Background()
	if c.Config.TimeoutSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(c.Config.TimeoutSeconds)*time.Second)
		defer cancel()
	}

	result, err := c.mcpCategoryInference.ClassifyWithProbabilities(ctx, text)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("MCP classification error: %w", err)
	}
	return result, nil
}

func (c *Classifier) mcpCategoryNames(probabilities []float32) []string {
	categoryNames := make([]string, len(probabilities))
	for i := range probabilities {
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
	return categoryNames
}

func (c *Classifier) makeMCPReasoningDecision(probabilities []float32, categoryNames []string) (entropy.ReasoningDecision, float64) {
	categoryReasoningMap := make(map[string]bool)
	for _, decision := range c.Config.Decisions {
		useReasoning := false
		if len(decision.ModelRefs) > 0 && decision.ModelRefs[0].UseReasoning != nil {
			useReasoning = *decision.ModelRefs[0].UseReasoning
		}
		categoryReasoningMap[strings.ToLower(decision.Name)] = useReasoning
	}

	entropyStart := time.Now()
	reasoningDecision := entropy.MakeEntropyBasedReasoningDecision(
		probabilities,
		categoryNames,
		categoryReasoningMap,
		float64(c.mcpCategoryThreshold()),
	)
	return reasoningDecision, time.Since(entropyStart).Seconds()
}

func (c *Classifier) mcpCategoryThreshold() float32 {
	threshold := c.Config.MCPCategoryModel.Threshold
	if threshold == 0 {
		threshold = DefaultMCPThreshold
	}
	return threshold
}

func (c *Classifier) mcpCategoryNameForClass(class int) (string, string) {
	if c.CategoryMapping == nil {
		categoryName := fmt.Sprintf("category_%d", class)
		return categoryName, categoryName
	}

	name, ok := c.CategoryMapping.GetCategoryFromIndex(class)
	if !ok {
		categoryName := fmt.Sprintf("category_%d", class)
		return categoryName, categoryName
	}
	return name, c.translateMMLUToGeneric(name)
}

func recordMCPProbabilityMetrics(probabilities []float32, reasoningDecision entropy.ReasoningDecision, entropyLatency float64) {
	entropyValue := entropy.CalculateEntropy(probabilities)
	topCategory := "none"
	if len(reasoningDecision.TopCategories) > 0 {
		topCategory = reasoningDecision.TopCategories[0].Category
	}

	probSum := float32(0.0)
	hasNegative := false
	for _, prob := range probabilities {
		probSum += prob
		if prob < 0 {
			hasNegative = true
		}
	}

	if probSum >= 0.99 && probSum <= 1.01 {
		metrics.RecordProbabilityDistributionQuality("sum_check", "valid")
	} else {
		metrics.RecordProbabilityDistributionQuality("sum_check", "invalid")
		logging.Warnf("MCP probability distribution sum is %.3f (should be ~1.0)", probSum)
	}

	if hasNegative {
		metrics.RecordProbabilityDistributionQuality("negative_check", "invalid")
	} else {
		metrics.RecordProbabilityDistributionQuality("negative_check", "valid")
	}

	entropyResult := entropy.AnalyzeEntropy(probabilities)
	metrics.RecordEntropyClassificationMetrics(
		topCategory,
		entropyResult.UncertaintyLevel,
		entropyValue,
		reasoningDecision.Confidence,
		reasoningDecision.UseReasoning,
		reasoningDecision.DecisionReason,
		topCategory,
		entropyLatency,
	)
}
