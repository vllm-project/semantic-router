package services

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (s *ClassificationService) getRecommendedModel(category string, _ float64) string {
	if s.classifier != nil {
		model := s.classifier.SelectBestModelForCategory(category)
		if model != "" {
			return model
		}
	}
	if s.config == nil {
		return ""
	}
	if model := recommendedModelFromDecisions(s.config.Decisions, category); model != "" {
		return model
	}
	return s.config.DefaultModel
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
