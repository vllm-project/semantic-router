package services

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func resolveIntentCategory(
	classifier *classification.Classifier,
	decisionResult *decision.DecisionResult,
	text string,
) (string, float64) {
	if decisionResult != nil && decisionResult.Decision != nil {
		return decisionResult.Decision.Name, decisionResult.Confidence
	}
	category, confidence, _, err := classifier.ClassifyCategoryWithEntropy(text)
	if err != nil {
		logging.Warnf(
			"Classification fallback failed: %v, using default 'other' category",
			err,
		)
		return "other", 0
	}
	return category, confidence
}
