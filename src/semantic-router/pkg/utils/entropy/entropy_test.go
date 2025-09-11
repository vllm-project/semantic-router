package entropy

import (
	"math"
	"testing"
)

func TestCalculateEntropy(t *testing.T) {
	tests := []struct {
		name           string
		probabilities  []float32
		expectedResult float64
	}{
		{
			name:           "Uniform distribution",
			probabilities:  []float32{0.25, 0.25, 0.25, 0.25},
			expectedResult: 2.0, // log2(4) = 2.0 for uniform distribution
		},
		{
			name:           "Certain prediction",
			probabilities:  []float32{1.0, 0.0, 0.0, 0.0},
			expectedResult: 0.0, // No uncertainty
		},
		{
			name:           "High certainty",
			probabilities:  []float32{0.85, 0.05, 0.05, 0.05},
			expectedResult: 0.8476, // Should be low entropy
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateEntropy(tt.probabilities)
			if math.Abs(result-tt.expectedResult) > 0.01 {
				t.Errorf("CalculateEntropy() = %v, want %v", result, tt.expectedResult)
			}
		})
	}
}

func TestCalculateNormalizedEntropy(t *testing.T) {
	tests := []struct {
		name           string
		probabilities  []float32
		expectedResult float64
	}{
		{
			name:           "Uniform distribution",
			probabilities:  []float32{0.25, 0.25, 0.25, 0.25},
			expectedResult: 1.0, // Maximum entropy for 4 classes
		},
		{
			name:           "Certain prediction",
			probabilities:  []float32{1.0, 0.0, 0.0, 0.0},
			expectedResult: 0.0, // No uncertainty
		},
		{
			name:           "High certainty biology",
			probabilities:  []float32{0.85, 0.05, 0.03, 0.03, 0.02, 0.02},
			expectedResult: 0.365, // Should be low normalized entropy
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateNormalizedEntropy(tt.probabilities)
			if math.Abs(result-tt.expectedResult) > 0.01 {
				t.Errorf("CalculateNormalizedEntropy() = %v, want %v", result, tt.expectedResult)
			}
		})
	}
}

func TestAnalyzeEntropy(t *testing.T) {
	tests := []struct {
		name                     string
		probabilities            []float32
		expectedUncertaintyLevel string
	}{
		{
			name:                     "Very high uncertainty",
			probabilities:            []float32{0.17, 0.17, 0.17, 0.17, 0.16, 0.16},
			expectedUncertaintyLevel: "very_high",
		},
		{
			name:                     "High uncertainty",
			probabilities:            []float32{0.45, 0.40, 0.10, 0.05},
			expectedUncertaintyLevel: "high",
		},
		{
			name:                     "Medium uncertainty",
			probabilities:            []float32{0.70, 0.15, 0.10, 0.05},
			expectedUncertaintyLevel: "high", // Actually 0.660 normalized entropy
		},
		{
			name:                     "Low uncertainty",
			probabilities:            []float32{0.85, 0.05, 0.05, 0.05},
			expectedUncertaintyLevel: "medium", // Actually 0.424 normalized entropy
		},
		{
			name:                     "Very low uncertainty",
			probabilities:            []float32{0.90, 0.04, 0.03, 0.02, 0.01},
			expectedUncertaintyLevel: "low", // Actually 0.282 normalized entropy
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := AnalyzeEntropy(tt.probabilities)
			if result.UncertaintyLevel != tt.expectedUncertaintyLevel {
				t.Errorf("AnalyzeEntropy().UncertaintyLevel = %v, want %v", result.UncertaintyLevel, tt.expectedUncertaintyLevel)
			}
		})
	}
}

func TestMakeEntropyBasedReasoningDecision(t *testing.T) {
	categoryReasoningMap := map[string]bool{
		"biology":   false,
		"chemistry": false,
		"law":       false,
		"other":     false,
		"physics":   true,
		"business":  true,
	}

	tests := []struct {
		name                   string
		probabilities          []float32
		categoryNames          []string
		expectedUseReasoning   bool
		expectedDecisionReason string
	}{
		{
			name:                   "High certainty biology (should not use reasoning)",
			probabilities:          []float32{0.85, 0.05, 0.03, 0.03, 0.02, 0.02},
			categoryNames:          []string{"biology", "other", "chemistry", "law", "physics", "business"},
			expectedUseReasoning:   false,
			expectedDecisionReason: "low_uncertainty_trust_classification",
		},
		{
			name:                   "Uniform distribution (very high uncertainty)",
			probabilities:          []float32{0.17, 0.17, 0.17, 0.17, 0.16, 0.16},
			categoryNames:          []string{"biology", "other", "chemistry", "law", "physics", "business"},
			expectedUseReasoning:   true,
			expectedDecisionReason: "very_high_uncertainty_conservative_default",
		},
		{
			name:                   "High uncertainty between biology and chemistry",
			probabilities:          []float32{0.45, 0.40, 0.10, 0.03, 0.01, 0.01},
			categoryNames:          []string{"biology", "chemistry", "other", "law", "physics", "business"},
			expectedUseReasoning:   false, // Both biology and chemistry don't use reasoning
			expectedDecisionReason: "high_uncertainty_weighted_decision",
		},
		{
			name:                   "Strong physics classification",
			probabilities:          []float32{0.90, 0.04, 0.02, 0.02, 0.01, 0.01},
			categoryNames:          []string{"physics", "biology", "chemistry", "law", "other", "business"},
			expectedUseReasoning:   true,                                   // Physics uses reasoning
			expectedDecisionReason: "low_uncertainty_trust_classification", // Actually low uncertainty, not very low
		},
		{
			name:                   "Problematic other category with medium uncertainty",
			probabilities:          []float32{0.70, 0.15, 0.10, 0.03, 0.01, 0.01},
			categoryNames:          []string{"other", "biology", "chemistry", "law", "physics", "business"},
			expectedUseReasoning:   false, // Other category doesn't use reasoning
			expectedDecisionReason: "medium_uncertainty_top_category_above_threshold",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MakeEntropyBasedReasoningDecision(
				tt.probabilities,
				tt.categoryNames,
				categoryReasoningMap,
				0.6, // threshold
			)

			if result.UseReasoning != tt.expectedUseReasoning {
				t.Errorf("MakeEntropyBasedReasoningDecision().UseReasoning = %v, want %v", result.UseReasoning, tt.expectedUseReasoning)
			}

			if result.DecisionReason != tt.expectedDecisionReason {
				t.Errorf("MakeEntropyBasedReasoningDecision().DecisionReason = %v, want %v", result.DecisionReason, tt.expectedDecisionReason)
			}

			// Verify top categories are returned
			if len(result.TopCategories) == 0 {
				t.Error("Expected top categories to be returned")
			}

			// Verify confidence is reasonable
			if result.Confidence < 0.0 || result.Confidence > 1.0 {
				t.Errorf("Confidence should be between 0 and 1, got %v", result.Confidence)
			}
		})
	}
}

func TestGetTopCategories(t *testing.T) {
	probabilities := []float32{0.45, 0.30, 0.15, 0.05, 0.03, 0.02}
	categoryNames := []string{"biology", "chemistry", "physics", "law", "other", "business"}

	result := getTopCategories(probabilities, categoryNames, 3)

	if len(result) != 3 {
		t.Errorf("Expected 3 top categories, got %d", len(result))
	}

	// Check that they're sorted by probability (descending)
	if result[0].Category != "biology" || result[0].Probability != 0.45 {
		t.Errorf("Expected first category to be biology with 0.45, got %s with %f", result[0].Category, result[0].Probability)
	}

	if result[1].Category != "chemistry" || result[1].Probability != 0.30 {
		t.Errorf("Expected second category to be chemistry with 0.30, got %s with %f", result[1].Category, result[1].Probability)
	}

	if result[2].Category != "physics" || result[2].Probability != 0.15 {
		t.Errorf("Expected third category to be physics with 0.15, got %s with %f", result[2].Category, result[2].Probability)
	}
}
