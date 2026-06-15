package classification

import (
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// ClassifyCategoryWithEntropy performs category classification with entropy-based reasoning decision
func (c *Classifier) ClassifyCategoryWithEntropy(text string) (string, float64, entropy.ReasoningDecision, error) {
	// Try keyword and embedding classifiers first
	category, confidence, decision, matched, err := c.tryKeywordBasedClassification(text)
	if err != nil {
		return "", 0.0, entropy.ReasoningDecision{}, err
	}
	if matched {
		return category, confidence, decision, nil
	}

	// Try in-tree first if properly configured
	if c.IsCategoryEnabled() && c.categoryInference != nil {
		return c.classifyCategoryWithEntropyInTree(text)
	}

	// If in-tree classifier was initialized but config is now invalid, return specific error
	if c.categoryInference != nil && !c.IsCategoryEnabled() {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("category classification is not properly configured")
	}

	// Fall back to MCP
	if c.IsMCPCategoryEnabled() && c.mcpCategoryInference != nil {
		return c.classifyCategoryWithEntropyMCP(text)
	}

	return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("no category classification method available")
}

// tryKeywordBasedClassification attempts classification via keyword and embedding classifiers.
// Returns matched=true if a classifier produced a result.
func (c *Classifier) tryKeywordBasedClassification(text string) (string, float64, entropy.ReasoningDecision, bool, error) {
	for _, clf := range []interface {
		Classify(string) (string, float64, error)
	}{c.keywordClassifier, c.keywordEmbeddingClassifier} {
		if clf == nil {
			continue
		}
		category, confidence, err := clf.Classify(text)
		if err != nil {
			return "", 0.0, entropy.ReasoningDecision{}, false, err
		}
		if category != "" {
			reasoningDecision := c.makeReasoningDecisionForKeywordCategory(category)
			return category, confidence, reasoningDecision, true, nil
		}
	}
	return "", 0.0, entropy.ReasoningDecision{}, false, nil
}

// makeReasoningDecisionForKeywordCategory creates a reasoning decision for keyword-matched categories
func (c *Classifier) makeReasoningDecisionForKeywordCategory(category string) entropy.ReasoningDecision {
	// Find the decision configuration
	normalizedCategory := strings.ToLower(strings.TrimSpace(category))
	useReasoning := false

	for _, decision := range c.Config.Decisions {
		if strings.ToLower(decision.Name) == normalizedCategory {
			// Check if the decision has reasoning enabled in its best model
			if len(decision.ModelRefs) > 0 && decision.ModelRefs[0].UseReasoning != nil {
				useReasoning = *decision.ModelRefs[0].UseReasoning
			}
			break
		}
	}

	return entropy.ReasoningDecision{
		UseReasoning:     useReasoning,
		Confidence:       1.0, // Keyword matches have 100% confidence
		DecisionReason:   "keyword_match_category_config",
		FallbackStrategy: "keyword_based_classification",
		TopCategories: []entropy.CategoryProbability{
			{
				Category:    category,
				Probability: 1.0,
			},
		},
	}
}

// classifyCategoryWithEntropyInTree performs category classification with entropy using in-tree model
func (c *Classifier) classifyCategoryWithEntropyInTree(text string) (string, float64, entropy.ReasoningDecision, error) {
	if !c.IsCategoryEnabled() {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("category classification is not properly configured")
	}

	// Get full probability distribution
	result, err := c.categoryInference.ClassifyWithProbabilities(text)
	if err != nil {
		return "", 0.0, entropy.ReasoningDecision{}, fmt.Errorf("classification error: %w", err)
	}

	logging.Debugf("Classification result: class=%d, confidence=%.4f, entropy_available=%t",
		result.Class, result.Confidence, len(result.Probabilities) > 0)

	// Get category names for all classes and translate to generic names when configured
	categoryNames := make([]string, len(result.Probabilities))
	for i := range result.Probabilities {
		if name, ok := c.CategoryMapping.GetCategoryFromIndex(i); ok {
			categoryNames[i] = c.translateMMLUToGeneric(name)
		} else {
			categoryNames[i] = fmt.Sprintf("unknown_%d", i)
		}
	}

	// Build decision reasoning map from configuration
	// Use the best model's reasoning capability for each decision
	categoryReasoningMap := make(map[string]bool)
	for _, decision := range c.Config.Decisions {
		useReasoning := false
		if len(decision.ModelRefs) > 0 && decision.ModelRefs[0].UseReasoning != nil {
			// Use the first (best) model's reasoning capability
			useReasoning = *decision.ModelRefs[0].UseReasoning
		}
		categoryReasoningMap[strings.ToLower(decision.Name)] = useReasoning
	}

	// Make entropy-based reasoning decision
	entropyStart := time.Now()
	reasoningDecision := entropy.MakeEntropyBasedReasoningDecision(
		result.Probabilities,
		categoryNames,
		categoryReasoningMap,
		float64(c.Config.CategoryModel.Threshold),
	)
	entropyLatency := time.Since(entropyStart).Seconds()

	c.recordEntropyMetrics(result.Probabilities, reasoningDecision, entropyLatency)

	// Check confidence threshold for category determination
	if result.Confidence < c.Config.CategoryModel.Threshold {
		// Determine fallback category (default to "other" if not configured)
		fallbackCategory := c.Config.FallbackCategory
		if fallbackCategory == "" {
			fallbackCategory = "other"
		}

		logging.Debugf("Classification confidence (%.4f) below threshold (%.4f), falling back to category: %s",
			result.Confidence, c.Config.CategoryModel.Threshold, fallbackCategory)

		// Record the fallback category as a signal match
		metrics.RecordSignalMatch(config.SignalTypeKeyword, fallbackCategory)

		// Return fallback category instead of empty string to enable proper decision routing
		return fallbackCategory, float64(result.Confidence), reasoningDecision, nil
	}

	// Convert class index to category name and translate to generic
	categoryName, ok := c.CategoryMapping.GetCategoryFromIndex(result.Class)
	if !ok {
		// Determine fallback category (default to "other" if not configured)
		fallbackCategory := c.Config.FallbackCategory
		if fallbackCategory == "" {
			fallbackCategory = "other"
		}

		logging.Warnf("Class index %d not found in category mapping, falling back to: %s", result.Class, fallbackCategory)
		metrics.RecordSignalMatch(config.SignalTypeKeyword, fallbackCategory)
		return fallbackCategory, float64(result.Confidence), reasoningDecision, nil
	}
	genericCategory := c.translateMMLUToGeneric(categoryName)

	// Record the category as a signal match
	metrics.RecordSignalMatch(config.SignalTypeKeyword, genericCategory)

	logging.Debugf("Classified as category: %s (mmlu=%s), reasoning_decision: use=%t, confidence=%.3f, reason=%s",
		genericCategory, categoryName, reasoningDecision.UseReasoning, reasoningDecision.Confidence, reasoningDecision.DecisionReason)

	return genericCategory, float64(result.Confidence), reasoningDecision, nil
}

func (c *Classifier) recordEntropyMetrics(probabilities []float32, reasoningDecision entropy.ReasoningDecision, entropyLatency float64) {
	// Calculate entropy value for metrics
	entropyValue := entropy.CalculateEntropy(probabilities)

	// Determine top category for metrics
	topCategory := "none"
	if len(reasoningDecision.TopCategories) > 0 {
		topCategory = reasoningDecision.TopCategories[0].Category
	}

	// Validate probability distribution quality
	probSum := float32(0.0)
	for _, prob := range probabilities {
		probSum += prob
	}

	// Record probability distribution quality checks
	if probSum >= 0.99 && probSum <= 1.01 {
		metrics.RecordProbabilityDistributionQuality("sum_check", "valid")
	} else {
		metrics.RecordProbabilityDistributionQuality("sum_check", "invalid")
		logging.Warnf("Probability distribution sum is %.3f (should be ~1.0)", probSum)
	}

	// Check for negative probabilities
	hasNegative := false
	for _, prob := range probabilities {
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
	entropyResult := entropy.AnalyzeEntropy(probabilities)
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
}
