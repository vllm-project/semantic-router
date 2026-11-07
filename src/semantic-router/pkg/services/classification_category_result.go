package services

import (
	"context"
	"strings"

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
	category, _ := resolveIntentCategoryWithError(ctx, classifier, decisionResult, signals, text)
	return category
}

// resolveIntentCategoryWithError retains inference failures for batch items;
// legacy single-input responses keep their established fallback representation.
func resolveIntentCategoryWithError(
	ctx context.Context,
	classifier *classification.Classifier,
	decisionResult *decision.DecisionResult,
	signals *classification.SignalResults,
	text string,
) (Classification, error) {
	if decisionResult != nil && decisionResult.Decision != nil {
		return Classification{
			Category:            decisionResult.Decision.Name,
			Confidence:          decisionResult.Confidence,
			ConfidenceAvailable: confidenceAvailability(decisionResult.ConfidenceScored),
		}, nil
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
		}, nil
	}
	// Image-only intent has no text for a text-only category fallback.
	if strings.TrimSpace(text) == "" {
		return Classification{Category: "other", ConfidenceAvailable: confidenceAvailability(false)}, nil
	}
	category, confidence, _, err := classifier.ClassifyCategoryWithEntropyContext(ctx, text)
	if err != nil {
		logging.Warnf(
			"Classification fallback failed: %v, using default 'other' category",
			err,
		)
		return Classification{Category: "other", ConfidenceAvailable: confidenceAvailability(false)}, err
	}
	return Classification{Category: category, Confidence: confidence, ConfidenceAvailable: confidenceAvailability(true)}, nil
}
