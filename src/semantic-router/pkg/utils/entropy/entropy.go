package entropy

import (
	"cmp"
	"math"
	"slices"
)

// EntropyResult contains the results of entropy-based analysis
type EntropyResult struct {
	Entropy           float64 // Shannon entropy of the probability distribution
	NormalizedEntropy float64 // Entropy normalized to [0,1] range
	Certainty         float64 // Inverse of normalized entropy (1 - normalized_entropy)
	UncertaintyLevel  string  // Human-readable uncertainty level
}

// CategoryProbability represents a category and its probability
type CategoryProbability struct {
	Category    string  // Category name
	Probability float32 // Probability for this category
}

// CalculateEntropy calculates Shannon entropy from a probability distribution
func CalculateEntropy(probabilities []float32) float64 {
	if len(probabilities) == 0 {
		return 0.0
	}

	entropy := 0.0
	for _, prob := range probabilities {
		if prob > 0 {
			entropy -= float64(prob) * math.Log2(float64(prob))
		}
	}

	return entropy
}

// CalculateNormalizedEntropy calculates normalized entropy (0 to 1 scale)
func CalculateNormalizedEntropy(probabilities []float32) float64 {
	if len(probabilities) <= 1 {
		return 0.0
	}

	entropy := CalculateEntropy(probabilities)
	maxEntropy := math.Log2(float64(len(probabilities)))

	if maxEntropy == 0 {
		return 0.0
	}

	return entropy / maxEntropy
}

// AnalyzeEntropy performs comprehensive entropy analysis
func AnalyzeEntropy(probabilities []float32) EntropyResult {
	entropy := CalculateEntropy(probabilities)
	normalizedEntropy := CalculateNormalizedEntropy(probabilities)
	certainty := 1.0 - normalizedEntropy

	// Determine uncertainty level
	var uncertaintyLevel string
	if normalizedEntropy >= 0.8 {
		uncertaintyLevel = "very_high"
	} else if normalizedEntropy >= 0.6 {
		uncertaintyLevel = "high"
	} else if normalizedEntropy >= 0.4 {
		uncertaintyLevel = "medium"
	} else if normalizedEntropy >= 0.2 {
		uncertaintyLevel = "low"
	} else {
		uncertaintyLevel = "very_low"
	}

	return EntropyResult{
		Entropy:           entropy,
		NormalizedEntropy: normalizedEntropy,
		Certainty:         certainty,
		UncertaintyLevel:  uncertaintyLevel,
	}
}

// ReasoningDecision contains the result of entropy-based reasoning decision
type ReasoningDecision struct {
	UseReasoning     bool                  // Whether to use reasoning
	Confidence       float64               // Confidence in the decision
	DecisionReason   string                // Human-readable reason for the decision
	FallbackStrategy string                // Strategy used for uncertain cases
	TopCategories    []CategoryProbability // Top predicted categories
}

// MakeEntropyBasedReasoningDecision implements the entropy-based reasoning strategy
func MakeEntropyBasedReasoningDecision(
	probabilities []float32,
	categoryNames []string,
	categoryReasoningMap map[string]bool,
	baseConfidenceThreshold float64,
) ReasoningDecision {

	if len(probabilities) == 0 || len(categoryNames) == 0 {
		return ReasoningDecision{
			UseReasoning:     false,
			Confidence:       0.0,
			DecisionReason:   "no_classification_data",
			FallbackStrategy: "default_no_reasoning",
		}
	}

	// Analyze entropy
	entropyResult := AnalyzeEntropy(probabilities)

	// Get top predicted categories with probabilities
	topCategories := getTopCategories(probabilities, categoryNames, 3)

	// Get the top prediction
	topCategory := topCategories[0]
	topConfidence := float64(topCategory.Probability)

	// Entropy-based decision logic
	switch entropyResult.UncertaintyLevel {
	case "very_high":
		// Very uncertain - use conservative default (enable reasoning for safety)
		return ReasoningDecision{
			UseReasoning:     true,
			Confidence:       0.3,
			DecisionReason:   "very_high_uncertainty_conservative_default",
			FallbackStrategy: "high_uncertainty_reasoning_enabled",
			TopCategories:    topCategories,
		}

	case "high":
		// High uncertainty - use weighted decision from top 2 categories
		topTwoCategories := topCategories
		if len(topTwoCategories) > 2 {
			topTwoCategories = topTwoCategories[:2]
		}
		decision := makeWeightedDecision(topTwoCategories, categoryReasoningMap, "high_uncertainty_weighted_decision")
		decision.FallbackStrategy = "top_two_categories_weighted"
		return decision

	case "medium":
		// Medium uncertainty - trust top category if above threshold, otherwise weighted
		if topConfidence >= baseConfidenceThreshold {
			if useReasoning, exists := categoryReasoningMap[topCategory.Category]; exists {
				return ReasoningDecision{
					UseReasoning:     useReasoning,
					Confidence:       topConfidence * 0.8, // Reduce confidence due to medium uncertainty
					DecisionReason:   "medium_uncertainty_top_category_above_threshold",
					FallbackStrategy: "trust_top_category",
					TopCategories:    topCategories,
				}
			}
		}

		// Fall back to weighted decision
		return makeWeightedDecision(topCategories, categoryReasoningMap, "medium_uncertainty_weighted")

	case "low", "very_low":
		// Low uncertainty - trust the classification completely
		if useReasoning, exists := categoryReasoningMap[topCategory.Category]; exists {
			confidenceMultiplier := 0.9
			if entropyResult.UncertaintyLevel == "very_low" {
				confidenceMultiplier = 0.95
			}

			return ReasoningDecision{
				UseReasoning:     useReasoning,
				Confidence:       topConfidence * confidenceMultiplier,
				DecisionReason:   entropyResult.UncertaintyLevel + "_uncertainty_trust_classification",
				FallbackStrategy: "trust_top_category",
				TopCategories:    topCategories,
			}
		}

		// Category not in reasoning map - default to no reasoning
		return ReasoningDecision{
			UseReasoning:     false,
			Confidence:       topConfidence * 0.8,
			DecisionReason:   "category_not_in_reasoning_map",
			FallbackStrategy: "unknown_category_default",
			TopCategories:    topCategories,
		}

	default:
		// Unknown uncertainty level - conservative default
		return ReasoningDecision{
			UseReasoning:     false,
			Confidence:       0.5,
			DecisionReason:   "unknown_uncertainty_level",
			FallbackStrategy: "conservative_default",
			TopCategories:    topCategories,
		}
	}
}

// Helper function to get top N categories with their probabilities
func getTopCategories(probabilities []float32, categoryNames []string, topN int) []CategoryProbability {
	if len(probabilities) != len(categoryNames) {
		return []CategoryProbability{}
	}

	// Create category-probability pairs
	pairs := make([]CategoryProbability, len(probabilities))
	for i, prob := range probabilities {
		pairs[i] = CategoryProbability{
			Category:    categoryNames[i],
			Probability: prob,
		}
	}

	// Sort by probability (descending)
	slices.SortFunc(pairs, func(a, b CategoryProbability) int {
		return cmp.Compare(b.Probability, a.Probability) // Note: b, a for descending
	})

	// Return top N
	n := min(topN, len(pairs))
	return pairs[:n]
}

// Helper function to make weighted decision from top categories
func makeWeightedDecision(topCategories []CategoryProbability, categoryReasoningMap map[string]bool, reason string) ReasoningDecision {
	weightedReasoningScore := 0.0
	totalWeight := 0.0

	for _, cat := range topCategories {
		if useReasoning, exists := categoryReasoningMap[cat.Category]; exists {
			weight := float64(cat.Probability)
			if useReasoning {
				weightedReasoningScore += weight
			}
			totalWeight += weight
		}
	}

	useReasoning := false
	confidence := 0.5
	if totalWeight > 0 {
		reasoningRatio := weightedReasoningScore / totalWeight
		useReasoning = reasoningRatio > 0.5
		confidence = math.Abs(reasoningRatio-0.5) + 0.5
	}

	return ReasoningDecision{
		UseReasoning:     useReasoning,
		Confidence:       confidence,
		DecisionReason:   reason,
		FallbackStrategy: "weighted_decision",
		TopCategories:    topCategories,
	}
}

// Helper function to get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
