package services

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (s *ClassificationService) getRecommendedModel(category string, _ float64) string {
	classifier, runtimeConfig := s.runtimeSnapshot()
	return recommendedModelFromRuntime(classifier, runtimeConfig, category)
}

func recommendedModelFromRuntime(
	classifier *classification.Classifier,
	runtimeConfig *config.RouterConfig,
	category string,
) string {
	if classifier != nil {
		model := classifier.SelectBestModelForCategory(category)
		if model != "" {
			return model
		}
	}
	if runtimeConfig == nil {
		return ""
	}
	if model := recommendedModelFromDecisions(runtimeConfig.Decisions, category); model != "" {
		return model
	}
	return runtimeConfig.DefaultModel
}

func recommendedModelFromDecisions(decisions []config.Decision, category string) string {
	for _, decision := range decisions {
		if !strings.EqualFold(decision.Name, category) {
			continue
		}
		if len(decision.ModelRefs) == 0 {
			return ""
		}
		modelRef := decision.ModelRefs[0]
		if modelRef.LoRAName != "" {
			return modelRef.LoRAName
		}
		return modelRef.Model
	}
	return ""
}

func (s *ClassificationService) getRoutingDecision(confidence float64, options *IntentOptions) string {
	threshold := 0.7
	if options != nil && options.ConfidenceThreshold > 0 {
		threshold = options.ConfidenceThreshold
	}

	if confidence >= threshold {
		return "high_confidence_specialized"
	}
	return "low_confidence_general"
}
