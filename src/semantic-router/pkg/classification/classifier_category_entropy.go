package classification

import (
	"fmt"
	"strings"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// ModalityClassificationResult holds the result of modality signal classification
type ModalityClassificationResult struct {
	Modality   string  // "AR", "DIFFUSION", or "BOTH"
	Confidence float32 // Confidence score (0.0-1.0)
	Method     string  // Detection method used: "classifier", "keyword", or "hybrid"
}

// classifyModality determines the response modality for a text prompt.
// It supports three configurable methods via ModalityDetectionConfig:
//   - "classifier": ML-based (mmBERT-32K) — errors if model not loaded
//   - "keyword":    Configurable keyword matching — requires keywords in config
//   - "hybrid":     Classifier when available + keyword confirmation/fallback (default)
func (c *Classifier) classifyModality(text string, detectionConfig *config.ModalityDetectionConfig) ModalityClassificationResult {
	if text == "" {
		return ModalityClassificationResult{Modality: "AR", Confidence: 1.0, Method: "default"}
	}

	method := detectionConfig.GetMethod()

	switch method {
	case config.ModalityDetectionClassifier:
		return c.classifyModalityByClassifier(text, detectionConfig)
	case config.ModalityDetectionKeyword:
		return c.classifyModalityByKeyword(text, detectionConfig)
	case config.ModalityDetectionHybrid:
		return c.classifyModalityHybrid(text, detectionConfig)
	default:
		logging.Errorf("[ModalitySignal] BUG: unknown detection method %q — defaulting to AR", method)
		return ModalityClassificationResult{Modality: "AR", Confidence: 0.0, Method: "error/unknown-method"}
	}
}

// classifyModalityByClassifier uses the mmBERT-32K ML classifier exclusively.
func (c *Classifier) classifyModalityByClassifier(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	result, err := candle_binding.ClassifyMmBert32KModality(text)
	if err == nil {
		logging.Debugf("[ModalitySignal] Classifier: %s (confidence=%.3f) for prompt: %.80s",
			result.Modality, result.Confidence, text)
		return ModalityClassificationResult{
			Modality:   result.Modality,
			Confidence: result.Confidence,
			Method:     "classifier",
		}
	}

	logging.Errorf("[ModalitySignal] Classifier unavailable: %v — defaulting to AR", err)
	return ModalityClassificationResult{Modality: "AR", Confidence: 0.0, Method: "classifier/error"}
}

// classifyModalityByKeyword uses keyword patterns from config to detect modality.
func (c *Classifier) classifyModalityByKeyword(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	if cfg == nil || len(cfg.Keywords) == 0 {
		logging.Warnf("[ModalitySignal] Keyword detection requested but no keywords configured — defaulting to AR")
		return ModalityClassificationResult{Modality: "AR", Confidence: 0.5, Method: "keyword/no-config"}
	}

	lowerContent := strings.ToLower(text)

	// Check if any configured image keyword matches
	hasImageIntent := false
	for _, kw := range cfg.Keywords {
		if strings.Contains(lowerContent, strings.ToLower(kw)) {
			hasImageIntent = true
			break
		}
	}

	if !hasImageIntent {
		return ModalityClassificationResult{Modality: "AR", Confidence: 0.8, Method: "keyword"}
	}

	// Image intent detected — check if it's BOTH using both_keywords from config
	if len(cfg.BothKeywords) > 0 {
		for _, kw := range cfg.BothKeywords {
			if strings.Contains(lowerContent, strings.ToLower(kw)) {
				logging.Debugf("[ModalitySignal] Keyword: BOTH detected (image + both_keyword %q) for: %.80s", kw, text)
				return ModalityClassificationResult{Modality: "BOTH", Confidence: 0.75, Method: "keyword"}
			}
		}
	}

	logging.Debugf("[ModalitySignal] Keyword: DIFFUSION detected for: %.80s", text)
	return ModalityClassificationResult{Modality: "DIFFUSION", Confidence: 0.8, Method: "keyword"}
}

// classifyModalityHybrid uses the ML classifier as primary, with keyword matching as
// fallback (when classifier is unavailable) or confirmation (when classifier confidence is low).
func (c *Classifier) classifyModalityHybrid(text string, cfg *config.ModalityDetectionConfig) ModalityClassificationResult {
	confThreshold := cfg.GetConfidenceThreshold()

	// Try classifier first
	classifierResult, err := candle_binding.ClassifyMmBert32KModality(text)
	if err == nil && classifierResult.Confidence >= confThreshold {
		logging.Debugf("[ModalitySignal] Hybrid(classifier): %s (confidence=%.3f, threshold=%.2f) for: %.80s",
			classifierResult.Modality, classifierResult.Confidence, confThreshold, text)
		return ModalityClassificationResult{
			Modality:   classifierResult.Modality,
			Confidence: classifierResult.Confidence,
			Method:     "hybrid/classifier",
		}
	}

	if err == nil {
		// Classifier available but low confidence - use keyword to confirm/override
		keywordResult := c.classifyModalityByKeyword(text, cfg)

		if classifierResult.Modality == keywordResult.Modality {
			logging.Infof("[ModalitySignal] Hybrid(agree): %s (classifier=%.3f, keyword=%.3f) for: %.80s",
				classifierResult.Modality, classifierResult.Confidence, keywordResult.Confidence, text)
			return ModalityClassificationResult{
				Modality:   classifierResult.Modality,
				Confidence: (classifierResult.Confidence + keywordResult.Confidence) / 2,
				Method:     "hybrid/agree",
			}
		}

		lowerThreshold := confThreshold * cfg.GetLowerThresholdRatio()
		if classifierResult.Confidence >= lowerThreshold {
			logging.Infof("[ModalitySignal] Hybrid(classifier-preferred): %s (classifier=%.3f vs keyword=%s) for: %.80s",
				classifierResult.Modality, classifierResult.Confidence, keywordResult.Modality, text)
			return ModalityClassificationResult{
				Modality:   classifierResult.Modality,
				Confidence: classifierResult.Confidence,
				Method:     "hybrid/classifier-preferred",
			}
		}

		logging.Debugf("[ModalitySignal] Hybrid(keyword-override): %s (classifier=%s@%.3f too low) for: %.80s",
			keywordResult.Modality, classifierResult.Modality, classifierResult.Confidence, text)
		return ModalityClassificationResult{
			Modality:   keywordResult.Modality,
			Confidence: keywordResult.Confidence,
			Method:     "hybrid/keyword-override",
		}
	}

	// Classifier unavailable - fall back to keyword detection
	logging.Debugf("[ModalitySignal] Hybrid: classifier unavailable (%v), using keyword detection", err)
	return c.classifyModalityByKeyword(text, cfg)
}

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
