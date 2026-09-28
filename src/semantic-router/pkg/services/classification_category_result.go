package services

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func resolveIntentCategory(
	ctx context.Context,
	classifier *classification.Classifier,
	decisionResult *decision.DecisionResult,
	signals *classification.SignalResults,
	text string,
) Classification {
	if decisionResult != nil && decisionResult.Decision != nil {
		return Classification{
			Category:            decisionResult.Decision.Name,
			Confidence:          decisionResult.Confidence,
			ConfidenceAvailable: confidenceAvailability(decisionResult.ConfidenceScored),
		}
	}
	if signals != nil && signals.DomainClassification != nil {
		result := signals.DomainClassification
		category := result.Category
		if !result.ConfidenceAvailable {
			category = "other"
		}
		return Classification{
			Category:            category,
			Confidence:          result.Confidence,
			ConfidenceAvailable: confidenceAvailability(result.ConfidenceAvailable),
		}
	}
	category, confidence, _, err := classifier.ClassifyCategoryWithEntropyContext(ctx, text)
	if err != nil {
		logging.Warnf(
			"Classification fallback failed: %v, using default 'other' category",
			err,
		)
		return Classification{Category: "other", ConfidenceAvailable: confidenceAvailability(false)}
	}
	return Classification{Category: category, Confidence: confidence, ConfidenceAvailable: confidenceAvailability(true)}
}
