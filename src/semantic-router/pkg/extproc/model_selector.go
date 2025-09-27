package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/rules"
)

// classifyAndSelectBestModel chooses best models based on hybrid routing (rules + ML)
func (r *OpenAIRouter) classifyAndSelectBestModel(query string) string {
	return r.classifyAndSelectBestModelWithContext(context.Background(), query, nil, "")
}

// classifyAndSelectBestModelWithContext chooses best models using hybrid routing with full context
func (r *OpenAIRouter) classifyAndSelectBestModelWithContext(ctx context.Context, query string, headers map[string]string, originalModel string) string {
	// Use hybrid router for enhanced routing decisions
	if r.HybridRouter != nil {
		decision, err := r.HybridRouter.RouteRequest(ctx, query, nil, headers, originalModel)
		if err != nil {
			observability.Errorf("Hybrid routing failed, falling back to legacy classifier: %v", err)
			return r.Classifier.ClassifyAndSelectBestModel(query)
		}

		observability.Infof("Hybrid routing decision: model=%s, rule_matched=%v, reasoning=%v",
			decision.SelectedModel, decision.RuleMatched, decision.UseReasoning)

		return decision.SelectedModel
	}

	// Fallback to legacy classifier
	return r.Classifier.ClassifyAndSelectBestModel(query)
}

// getRoutingDecisionWithExplanation returns the full routing decision with explanation
func (r *OpenAIRouter) getRoutingDecisionWithExplanation(ctx context.Context, query string, headers map[string]string, originalModel string) (*rules.RoutingDecision, error) {
	if r.HybridRouter != nil {
		return r.HybridRouter.RouteRequest(ctx, query, nil, headers, originalModel)
	}

	// Create a basic decision for legacy routing
	selectedModel := r.Classifier.ClassifyAndSelectBestModel(query)
	categoryName, confidence, _ := r.Classifier.ClassifyCategory(query)

	decision := &rules.RoutingDecision{
		RuleMatched:   false,
		SelectedModel: selectedModel,
		UseReasoning:  false, // Would need to check category config
		Headers:       make(map[string]string),
		Explanation: rules.DecisionExplanation{
			DecisionType: "model_based",
			CategoryClassification: &rules.CategoryClassificationResult{
				Category:   categoryName,
				Confidence: float64(confidence),
			},
			Reasoning:  "Legacy model-based routing",
			Confidence: float64(confidence),
		},
	}

	return decision, nil
}

// findCategoryForClassification determines the category for the given text using classification
func (r *OpenAIRouter) findCategoryForClassification(query string) string {
	if len(r.CategoryDescriptions) == 0 {
		return ""
	}

	categoryName, _, err := r.Classifier.ClassifyCategory(query)
	if err != nil {
		observability.Errorf("Category classification error: %v", err)
		return ""
	}

	return categoryName
}
