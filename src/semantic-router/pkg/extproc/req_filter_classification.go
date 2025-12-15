package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// performDecisionEvaluationAndModelSelection performs decision evaluation using DecisionEngine
// Returns (decisionName, confidence, reasoningDecision, selectedModel)
// This is the new approach that uses Decision-based routing with AND/OR rule combinations
// Decision evaluation is ALWAYS performed when decisions are configured (for plugin features like
// hallucination detection), but model selection only happens for auto models.
// If the request contains multimodal content (images), it uses multimodal classification instead of text-only.
func (r *OpenAIRouter) performDecisionEvaluationAndModelSelection(originalModel string, userContent string, nonUserMessages []string, ctx *RequestContext) (string, float64, entropy.ReasoningDecision, string) {
	var decisionName string
	var evaluationConfidence float64
	var reasoningDecision entropy.ReasoningDecision
	var selectedModel string

	// Check if request contains multimodal content (images) - images are already extracted and stored in context
	hasImages := len(ctx.Images) > 0

	// Check if there's content to evaluate
	if len(nonUserMessages) == 0 && userContent == "" && !hasImages {
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Check if decisions are configured
	if len(r.Config.Decisions) == 0 {
		if r.Config.IsAutoModelName(originalModel) {
			logging.Warnf("No decisions configured, using default model")
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
		}
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// If multimodal content exists, use multimodal classification
	if hasImages {
		logging.Infof("Detected multimodal request with images, using multimodal classification")

		// Get images from context (already extracted)
		images := ctx.Images

		contentType := "image"
		if userContent != "" || len(nonUserMessages) > 0 {
			contentType = "multimodal"
		}

		// Create multimodal classification request
		multimodalReq := services.MultimodalClassificationRequest{
			Text:        userContent,
			Images:      images,
			ContentType: contentType,
		}

		// Get classification service
		classificationSvc := services.GetGlobalClassificationService()
		if classificationSvc != nil {
			intentResp, err := classificationSvc.ClassifyMultimodal(multimodalReq)
			if err != nil {
				logging.Errorf("Multimodal classification failed: %v, falling back to text-only", err)
				// Fall through to text-only classification below
			} else if intentResp != nil && intentResp.Classification.Category != "" {
				// Use the category from multimodal classification to find matching decision
				categoryName := intentResp.Classification.Category
				logging.Infof("Multimodal classification result: category=%s, confidence=%.3f",
					categoryName, intentResp.Classification.Confidence)

				// Try to find a decision that matches this category
				// For now, use text-based decision engine with the category name as a hint
				// TODO: Enhance decision engine to support multimodal classification results directly
				evaluationText := userContent
				if evaluationText == "" && len(nonUserMessages) > 0 {
					evaluationText = strings.Join(nonUserMessages, " ")
				}
				if evaluationText == "" {
					evaluationText = categoryName // Use category as fallback text
				}

				// Perform decision evaluation with the text (multimodal already classified)
				result, err := r.Classifier.EvaluateDecisionWithEngine(evaluationText)
				if err != nil {
					logging.Errorf("Decision evaluation error after multimodal classification: %v", err)
					if r.Config.IsAutoModelName(originalModel) {
						return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
					}
					return "", 0.0, entropy.ReasoningDecision{}, ""
				}

				if result != nil && result.Decision != nil {
					// Use multimodal classification confidence if available
					confidence := float64(intentResp.Classification.Confidence)
					if result.Confidence > 0 {
						// Average or use the higher confidence
						if result.Confidence > confidence {
							confidence = result.Confidence
						}
					}

					// Store decision and continue with model selection
					ctx.VSRSelectedDecision = result.Decision
					ctx.VSRSelectedCategory = categoryName

					decisionName := result.Decision.Name
					logging.Infof("Decision Evaluation Result (multimodal): decision=%s, category=%s, confidence=%.3f",
						decisionName, categoryName, confidence)

					// Model selection for auto models
					if !r.Config.IsAutoModelName(originalModel) {
						return decisionName, confidence, entropy.ReasoningDecision{}, ""
					}

					// Select model from decision
					var selectedModel string
					var reasoningDecision entropy.ReasoningDecision
					if len(result.Decision.ModelRefs) > 0 {
						modelRef := result.Decision.ModelRefs[0]
						selectedModel = modelRef.Model
						if modelRef.LoRAName != "" {
							selectedModel = modelRef.LoRAName
						}

						if result.Decision.ModelRefs[0].UseReasoning != nil {
							useReasoning := *result.Decision.ModelRefs[0].UseReasoning
							reasoningDecision = entropy.ReasoningDecision{
								UseReasoning:     useReasoning,
								Confidence:       confidence,
								DecisionReason:   "multimodal_classification",
								FallbackStrategy: "multimodal_decision_based_routing",
								TopCategories: []entropy.CategoryProbability{
									{
										Category:    categoryName,
										Probability: float32(confidence),
									},
								},
							}
						}
					} else {
						selectedModel = r.Config.DefaultModel
					}

					return decisionName, confidence, reasoningDecision, selectedModel
				}
			}
		} else {
			logging.Warnf("Classification service not available, falling back to text-only classification")
		}
	}

	// Determine text to use for evaluation
	evaluationText := userContent
	if evaluationText == "" && len(nonUserMessages) > 0 {
		evaluationText = strings.Join(nonUserMessages, " ")
	}

	if evaluationText == "" {
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Perform decision evaluation using DecisionEngine
	// This is ALWAYS done when decisions are configured, regardless of model type,
	// because plugins (e.g., hallucination detection) depend on the matched decision
	result, err := r.Classifier.EvaluateDecisionWithEngine(evaluationText)
	if err != nil {
		logging.Errorf("Decision evaluation error: %v", err)
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
		}
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	if result == nil || result.Decision == nil {
		logging.Warnf("No decision matched")
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel
		}
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Store the selected decision in context for later use (e.g., plugins, header mutations)
	// This is critical for hallucination detection and other per-decision plugins
	ctx.VSRSelectedDecision = result.Decision

	// Extract domain category from matched rules (for VSRSelectedCategory header)
	// MatchedRules contains rule names like "domain:math", "keyword:thinking", etc.
	// We extract the first domain rule as the category
	categoryName := ""
	for _, rule := range result.MatchedRules {
		if strings.HasPrefix(rule, "domain:") {
			categoryName = strings.TrimPrefix(rule, "domain:")
			break
		}
	}
	// Store category in context for response headers
	ctx.VSRSelectedCategory = categoryName

	decisionName = result.Decision.Name
	evaluationConfidence = result.Confidence
	logging.Infof("Decision Evaluation Result: decision=%s, category=%s, confidence=%.3f, matched_rules=%v",
		decisionName, categoryName, evaluationConfidence, result.MatchedRules)

	// Model selection only happens for auto models
	// When a specific model is requested, we keep it but still apply decision plugins
	if !r.Config.IsAutoModelName(originalModel) {
		logging.Infof("Model %s explicitly specified, keeping original model (decision %s plugins will be applied)",
			originalModel, decisionName)
		return decisionName, evaluationConfidence, reasoningDecision, ""
	}

	// Select best model from the decision's ModelRefs (only for auto models)
	if len(result.Decision.ModelRefs) > 0 {
		modelRef := result.Decision.ModelRefs[0]
		// Use LoRA name if specified, otherwise use the base model name
		selectedModel = modelRef.Model
		if modelRef.LoRAName != "" {
			selectedModel = modelRef.LoRAName
			logging.Infof("Selected model from decision %s: %s (LoRA adapter for base model %s)",
				decisionName, selectedModel, modelRef.Model)
		} else {
			logging.Infof("Selected model from decision %s: %s", decisionName, selectedModel)
		}

		// Determine reasoning mode from the best model's configuration
		if result.Decision.ModelRefs[0].UseReasoning != nil {
			useReasoning := *result.Decision.ModelRefs[0].UseReasoning
			reasoningDecision = entropy.ReasoningDecision{
				UseReasoning:     useReasoning,
				Confidence:       evaluationConfidence,
				DecisionReason:   "decision_engine_evaluation",
				FallbackStrategy: "decision_based_routing",
				TopCategories: []entropy.CategoryProbability{
					{
						Category:    decisionName,
						Probability: float32(evaluationConfidence),
					},
				},
			}
			// Note: ReasoningEffort is handled separately in req_filter_reason.go
		}
	} else {
		// No model refs in decision, use default model
		selectedModel = r.Config.DefaultModel
		logging.Infof("No model refs in decision %s, using default model: %s", decisionName, selectedModel)
	}

	return decisionName, evaluationConfidence, reasoningDecision, selectedModel
}
