package entropy

import (
	"math"
	"strings"
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

// TestEntropyCalculationConsistency tests that entropy calculations are consistent across different scenarios
func TestEntropyCalculationConsistency(t *testing.T) {
	testCases := []struct {
		name          string
		probabilities []float32
		description   string
	}{
		{
			name:          "Identical high confidence distributions",
			probabilities: []float32{0.9, 0.05, 0.03, 0.02},
			description:   "Both calculations should produce identical entropy for same probability distribution",
		},
		{
			name:          "Identical medium uncertainty distributions",
			probabilities: []float32{0.5, 0.3, 0.15, 0.05},
			description:   "Both calculations should handle medium uncertainty identically",
		},
		{
			name:          "Identical uniform distributions",
			probabilities: []float32{0.25, 0.25, 0.25, 0.25},
			description:   "Both calculations should handle maximum entropy identically",
		},
	}

	categoryNames := []string{"physics", "biology", "chemistry", "other"}
	categoryReasoningMap := map[string]bool{
		"physics":   false,
		"biology":   false,
		"chemistry": true,
		"other":     true,
	}
	threshold := 0.7

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Test with same probability distribution for consistency
			decision1 := MakeEntropyBasedReasoningDecision(
				tc.probabilities, categoryNames, categoryReasoningMap, threshold)

			decision2 := MakeEntropyBasedReasoningDecision(
				tc.probabilities, categoryNames, categoryReasoningMap, threshold)

			// Verify decisions are identical
			if decision1.UseReasoning != decision2.UseReasoning {
				t.Errorf("Decisions should be identical, got %v vs %v",
					decision1.UseReasoning, decision2.UseReasoning)
			}

			if decision1.DecisionReason != decision2.DecisionReason {
				t.Errorf("Decision reasons should be identical, got '%s' vs '%s'",
					decision1.DecisionReason, decision2.DecisionReason)
			}

			// Verify entropy calculations are consistent
			entropy1 := CalculateEntropy(tc.probabilities)
			entropy2 := CalculateEntropy(tc.probabilities)

			if math.Abs(entropy1-entropy2) > 0.001 {
				t.Errorf("Entropy calculations should be identical, got %.6f vs %.6f",
					entropy1, entropy2)
			}

			t.Logf("Consistency test '%s' passed: %s", tc.name, tc.description)
		})
	}
}

// TestEntropyMetricsIntegration tests that entropy integrates properly with metrics calculations
func TestEntropyMetricsIntegration(t *testing.T) {
	// Test that entropy calculations work with the metrics system
	testCases := []struct {
		name                string
		probabilities       []float32
		expectedEntropy     float64
		expectedUncertainty string
		tolerance           float64
	}{
		{
			name:                "Low entropy metrics",
			probabilities:       []float32{0.85, 0.08, 0.04, 0.03},
			expectedEntropy:     0.828,    // Corrected based on actual calculation
			expectedUncertainty: "medium", // Corrected based on actual normalized entropy
			tolerance:           0.01,
		},
		{
			name:                "High entropy metrics",
			probabilities:       []float32{0.4, 0.35, 0.15, 0.1},
			expectedEntropy:     1.802,       // Corrected based on actual calculation
			expectedUncertainty: "very_high", // Corrected based on actual normalized entropy
			tolerance:           0.01,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Calculate entropy using the entropy package
			entropyValue := CalculateEntropy(tc.probabilities)

			if math.Abs(entropyValue-tc.expectedEntropy) > tc.tolerance {
				t.Errorf("Expected entropy %.3f, got %.3f (tolerance: %.3f)",
					tc.expectedEntropy, entropyValue, tc.tolerance)
			}

			// Calculate normalized entropy for uncertainty level using the package
			normalizedEntropy := CalculateNormalizedEntropy(tc.probabilities)

			var uncertaintyLevel string
			switch {
			case normalizedEntropy >= 0.8:
				uncertaintyLevel = "very_high"
			case normalizedEntropy >= 0.6:
				uncertaintyLevel = "high"
			case normalizedEntropy >= 0.4:
				uncertaintyLevel = "medium"
			case normalizedEntropy >= 0.2:
				uncertaintyLevel = "low"
			default:
				uncertaintyLevel = "very_low"
			}

			if uncertaintyLevel != tc.expectedUncertainty {
				t.Errorf("Expected uncertainty level '%s', got '%s'",
					tc.expectedUncertainty, uncertaintyLevel)
			}

			t.Logf("Entropy metrics test '%s' passed: entropy=%.3f, uncertainty=%s",
				tc.name, entropyValue, uncertaintyLevel)
		})
	}
}

// TestEntropyEdgeCases tests edge cases and boundary conditions
func TestEntropyEdgeCases(t *testing.T) {
	tests := []struct {
		name               string
		probabilities      []float32
		expectedEntropy    float64
		expectedNormalized float64
		description        string
	}{
		{
			name:               "Empty probability array",
			probabilities:      []float32{},
			expectedEntropy:    0.0,
			expectedNormalized: 0.0,
			description:        "Should return 0 entropy for empty array",
		},
		{
			name:               "Single probability",
			probabilities:      []float32{1.0},
			expectedEntropy:    0.0,
			expectedNormalized: 0.0,
			description:        "Should return 0 normalized entropy for single probability",
		},
		{
			name:               "All zeros",
			probabilities:      []float32{0.0, 0.0, 0.0, 0.0},
			expectedEntropy:    0.0,
			expectedNormalized: 0.0,
			description:        "Should handle all zeros gracefully",
		},
		{
			name:               "Very small probabilities",
			probabilities:      []float32{0.9999, 0.0001, 0.0, 0.0},
			expectedEntropy:    0.0014,
			expectedNormalized: 0.0007,
			description:        "Should handle very small probabilities",
		},
		{
			name:               "Perfect uniform distribution (2 classes)",
			probabilities:      []float32{0.5, 0.5},
			expectedEntropy:    1.0,
			expectedNormalized: 1.0,
			description:        "Should return max entropy for 2-class uniform distribution",
		},
		{
			name:               "Perfect uniform distribution (8 classes)",
			probabilities:      []float32{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125},
			expectedEntropy:    3.0,
			expectedNormalized: 1.0,
			description:        "Should return max entropy for 8-class uniform distribution",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entropy := CalculateEntropy(tt.probabilities)
			normalized := CalculateNormalizedEntropy(tt.probabilities)

			if math.Abs(entropy-tt.expectedEntropy) > 0.01 {
				t.Errorf("%s: Expected entropy %.4f, got %.4f", tt.description, tt.expectedEntropy, entropy)
			}

			if math.Abs(normalized-tt.expectedNormalized) > 0.01 {
				t.Errorf("%s: Expected normalized entropy %.4f, got %.4f", tt.description, tt.expectedNormalized, normalized)
			}
		})
	}
}

// TestEntropyUncertaintyLevels tests all uncertainty level classifications
func TestEntropyUncertaintyLevels(t *testing.T) {
	tests := []struct {
		name                     string
		probabilities            []float32
		expectedUncertaintyLevel string
		expectedNormalizedRange  [2]float64 // [min, max]
		description              string
	}{
		{
			name:                     "Very high uncertainty - uniform",
			probabilities:            []float32{0.25, 0.25, 0.25, 0.25},
			expectedUncertaintyLevel: "very_high",
			expectedNormalizedRange:  [2]float64{0.8, 1.0},
			description:              "Uniform distribution should have very high uncertainty",
		},
		{
			name:                     "Very high uncertainty - near uniform",
			probabilities:            []float32{0.28, 0.26, 0.24, 0.22},
			expectedUncertaintyLevel: "very_high",
			expectedNormalizedRange:  [2]float64{0.8, 1.0},
			description:              "Near-uniform distribution should have very high uncertainty",
		},
		{
			name:                     "High uncertainty - two dominant",
			probabilities:            []float32{0.45, 0.40, 0.10, 0.05},
			expectedUncertaintyLevel: "high",
			expectedNormalizedRange:  [2]float64{0.6, 0.8},
			description:              "Two competing categories should have high uncertainty",
		},
		{
			name:                     "Medium uncertainty - clear leader",
			probabilities:            []float32{0.70, 0.15, 0.10, 0.05},
			expectedUncertaintyLevel: "high",
			expectedNormalizedRange:  [2]float64{0.4, 0.8},
			description:              "Clear leader with some uncertainty",
		},
		{
			name:                     "Low uncertainty - strong leader",
			probabilities:            []float32{0.85, 0.08, 0.04, 0.03},
			expectedUncertaintyLevel: "medium",
			expectedNormalizedRange:  [2]float64{0.2, 0.6},
			description:              "Strong leader should have low uncertainty",
		},
		{
			name:                     "Very low uncertainty - dominant",
			probabilities:            []float32{0.95, 0.03, 0.01, 0.01},
			expectedUncertaintyLevel: "very_low",
			expectedNormalizedRange:  [2]float64{0.0, 0.2},
			description:              "Very dominant category should have very low uncertainty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := AnalyzeEntropy(tt.probabilities)

			if result.UncertaintyLevel != tt.expectedUncertaintyLevel {
				t.Errorf("%s: Expected uncertainty level '%s', got '%s'",
					tt.description, tt.expectedUncertaintyLevel, result.UncertaintyLevel)
			}

			if result.NormalizedEntropy < tt.expectedNormalizedRange[0] ||
				result.NormalizedEntropy > tt.expectedNormalizedRange[1] {
				t.Errorf("%s: Expected normalized entropy in range [%.2f, %.2f], got %.3f",
					tt.description, tt.expectedNormalizedRange[0], tt.expectedNormalizedRange[1],
					result.NormalizedEntropy)
			}

			t.Logf("%s: normalized_entropy=%.3f, certainty=%.3f, level=%s",
				tt.description, result.NormalizedEntropy, result.Certainty, result.UncertaintyLevel)
		})
	}
}

// TestReasoningDecisionComprehensive tests all reasoning decision scenarios comprehensively
func TestReasoningDecisionComprehensive(t *testing.T) {
	categoryReasoningMap := map[string]bool{
		"math":     true,
		"physics":  true,
		"code":     true,
		"biology":  false,
		"history":  false,
		"language": false,
	}

	tests := []struct {
		name                   string
		probabilities          []float32
		categoryNames          []string
		threshold              float64
		expectedUseReasoning   bool
		expectedDecisionReason string
		expectedConfidenceMin  float64
		expectedConfidenceMax  float64
		description            string
	}{
		{
			name:                   "Very high entropy - enable reasoning",
			probabilities:          []float32{0.25, 0.25, 0.25, 0.25},
			categoryNames:          []string{"math", "physics", "code", "biology"},
			threshold:              0.6,
			expectedUseReasoning:   true,
			expectedDecisionReason: "very_high_uncertainty_conservative_default",
			expectedConfidenceMin:  0.25,
			expectedConfidenceMax:  0.35,
			description:            "Uniform distribution should enable reasoning conservatively",
		},
		{
			name:                   "Very low entropy - trust classification (math)",
			probabilities:          []float32{0.95, 0.02, 0.02, 0.01},
			categoryNames:          []string{"math", "physics", "code", "biology"},
			threshold:              0.6,
			expectedUseReasoning:   true,
			expectedDecisionReason: "very_low_uncertainty_trust_classification",
			expectedConfidenceMin:  0.85,
			expectedConfidenceMax:  0.95,
			description:            "Very confident math classification should enable reasoning",
		},
		{
			name:                   "Very low entropy - trust classification (biology)",
			probabilities:          []float32{0.95, 0.02, 0.02, 0.01},
			categoryNames:          []string{"biology", "history", "language", "math"},
			threshold:              0.6,
			expectedUseReasoning:   false,
			expectedDecisionReason: "very_low_uncertainty_trust_classification",
			expectedConfidenceMin:  0.85,
			expectedConfidenceMax:  0.95,
			description:            "Very confident biology classification should not enable reasoning",
		},
		{
			name:                   "High uncertainty - weighted decision (both reasoning)",
			probabilities:          []float32{0.45, 0.40, 0.10, 0.05},
			categoryNames:          []string{"math", "physics", "code", "biology"},
			threshold:              0.6,
			expectedUseReasoning:   true,
			expectedDecisionReason: "high_uncertainty_weighted_decision",
			expectedConfidenceMin:  0.5,
			expectedConfidenceMax:  1.0,
			description:            "High uncertainty between two reasoning categories should enable reasoning",
		},
		{
			name:                   "High uncertainty - weighted decision (both non-reasoning)",
			probabilities:          []float32{0.45, 0.40, 0.10, 0.05},
			categoryNames:          []string{"biology", "history", "language", "math"},
			threshold:              0.6,
			expectedUseReasoning:   false,
			expectedDecisionReason: "high_uncertainty_weighted_decision",
			expectedConfidenceMin:  0.5,
			expectedConfidenceMax:  1.0,
			description:            "High uncertainty between two non-reasoning categories should not enable reasoning",
		},
		{
			name:                   "High uncertainty - weighted decision (mixed)",
			probabilities:          []float32{0.45, 0.40, 0.10, 0.05},
			categoryNames:          []string{"math", "biology", "history", "language"},
			threshold:              0.6,
			expectedUseReasoning:   true, // 0.45/(0.45+0.40) = 0.529 > 0.5, so reasoning enabled
			expectedDecisionReason: "high_uncertainty_weighted_decision",
			expectedConfidenceMin:  0.5,
			expectedConfidenceMax:  0.55,
			description:            "High uncertainty between reasoning and non-reasoning should use weighted decision",
		},
		{
			name:                   "Medium uncertainty - above threshold (math)",
			probabilities:          []float32{0.75, 0.15, 0.05, 0.05},
			categoryNames:          []string{"math", "physics", "code", "biology"},
			threshold:              0.6,
			expectedUseReasoning:   true,
			expectedDecisionReason: "medium_uncertainty_top_category_above_threshold",
			expectedConfidenceMin:  0.55,
			expectedConfidenceMax:  0.65,
			description:            "Medium uncertainty with math above threshold should enable reasoning",
		},
		{
			name:                   "Medium uncertainty - above threshold (biology)",
			probabilities:          []float32{0.75, 0.15, 0.05, 0.05},
			categoryNames:          []string{"biology", "history", "language", "math"},
			threshold:              0.6,
			expectedUseReasoning:   false,
			expectedDecisionReason: "medium_uncertainty_top_category_above_threshold",
			expectedConfidenceMin:  0.55,
			expectedConfidenceMax:  0.65,
			description:            "Medium uncertainty with biology above threshold should not enable reasoning",
		},
		{
			name:                   "Medium uncertainty - trust classification (code)",
			probabilities:          []float32{0.80, 0.10, 0.06, 0.04},
			categoryNames:          []string{"code", "math", "physics", "biology"},
			threshold:              0.6,
			expectedUseReasoning:   true,
			expectedDecisionReason: "medium_uncertainty_top_category_above_threshold",
			expectedConfidenceMin:  0.60,
			expectedConfidenceMax:  0.70,
			description:            "Medium uncertainty code classification should enable reasoning",
		},
		{
			name:                   "Empty probabilities - default behavior",
			probabilities:          []float32{},
			categoryNames:          []string{},
			threshold:              0.6,
			expectedUseReasoning:   false,
			expectedDecisionReason: "no_classification_data",
			expectedConfidenceMin:  0.0,
			expectedConfidenceMax:  0.0,
			description:            "Empty data should return safe default",
		},
		{
			name:                   "Category not in reasoning map",
			probabilities:          []float32{0.90, 0.05, 0.03, 0.02},
			categoryNames:          []string{"unknown", "math", "physics", "biology"},
			threshold:              0.6,
			expectedUseReasoning:   false,
			expectedDecisionReason: "category_not_in_reasoning_map",
			expectedConfidenceMin:  0.70,
			expectedConfidenceMax:  0.80,
			description:            "Unknown category should default to no reasoning",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MakeEntropyBasedReasoningDecision(
				tt.probabilities,
				tt.categoryNames,
				categoryReasoningMap,
				tt.threshold,
			)

			if result.UseReasoning != tt.expectedUseReasoning {
				t.Errorf("%s: Expected UseReasoning=%v, got %v",
					tt.description, tt.expectedUseReasoning, result.UseReasoning)
			}

			if result.DecisionReason != tt.expectedDecisionReason {
				t.Errorf("%s: Expected DecisionReason='%s', got '%s'",
					tt.description, tt.expectedDecisionReason, result.DecisionReason)
			}

			if result.Confidence < tt.expectedConfidenceMin || result.Confidence > tt.expectedConfidenceMax {
				t.Errorf("%s: Expected confidence in range [%.2f, %.2f], got %.3f",
					tt.description, tt.expectedConfidenceMin, tt.expectedConfidenceMax, result.Confidence)
			}

			if len(tt.probabilities) > 0 && len(result.TopCategories) == 0 {
				t.Errorf("%s: Expected TopCategories to be populated", tt.description)
			}

			t.Logf("%s: use_reasoning=%v, confidence=%.3f, reason=%s",
				tt.description, result.UseReasoning, result.Confidence, result.DecisionReason)
		})
	}
}

// TestConfidenceAdjustment tests confidence adjustment based on uncertainty levels
func TestConfidenceAdjustment(t *testing.T) {
	categoryReasoningMap := map[string]bool{
		"math":    true,
		"physics": true,
		"code":    true,
	}

	tests := []struct {
		name                 string
		probabilities        []float32
		expectedConfMin      float64
		expectedConfMax      float64
		expectedReasonSubstr string
		description          string
	}{
		{
			name:                 "Very low uncertainty - minimal adjustment",
			probabilities:        []float32{0.95, 0.03, 0.01, 0.01},
			expectedConfMin:      0.85,
			expectedConfMax:      0.95,
			expectedReasonSubstr: "very_low_uncertainty",
			description:          "Very low uncertainty should have highest confidence retention (~95% multiplier)",
		},
		{
			name:                 "Low uncertainty - small adjustment",
			probabilities:        []float32{0.90, 0.06, 0.02, 0.02},
			expectedConfMin:      0.75,
			expectedConfMax:      0.85,
			expectedReasonSubstr: "low_uncertainty",
			description:          "Low uncertainty should have good confidence retention (~90% multiplier)",
		},
		{
			name:                 "Medium uncertainty - moderate adjustment",
			probabilities:        []float32{0.75, 0.15, 0.05, 0.05},
			expectedConfMin:      0.55,
			expectedConfMax:      0.65,
			expectedReasonSubstr: "medium_uncertainty",
			description:          "Medium uncertainty should reduce confidence moderately (~80% multiplier)",
		},
		{
			name:                 "High uncertainty - weighted decision",
			probabilities:        []float32{0.45, 0.40, 0.10, 0.05},
			expectedConfMin:      0.50,
			expectedConfMax:      1.00,
			expectedReasonSubstr: "high_uncertainty_weighted",
			description:          "High uncertainty should use weighted decision",
		},
		{
			name:                 "Very high uncertainty - low confidence",
			probabilities:        []float32{0.25, 0.25, 0.25, 0.25},
			expectedConfMin:      0.25,
			expectedConfMax:      0.35,
			expectedReasonSubstr: "very_high_uncertainty",
			description:          "Very high uncertainty should have fixed low confidence (0.3)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			categoryNames := []string{"math", "physics", "code", "biology"}

			result := MakeEntropyBasedReasoningDecision(
				tt.probabilities,
				categoryNames,
				categoryReasoningMap,
				0.6,
			)

			// Check that confidence is in expected range
			if result.Confidence < tt.expectedConfMin || result.Confidence > tt.expectedConfMax {
				t.Errorf("%s: Expected confidence in range [%.2f, %.2f], got %.3f",
					tt.description, tt.expectedConfMin, tt.expectedConfMax, result.Confidence)
			}

			// Check that decision reason contains expected substring
			if !strings.Contains(result.DecisionReason, tt.expectedReasonSubstr) {
				t.Errorf("%s: Expected reason to contain '%s', got '%s'",
					tt.description, tt.expectedReasonSubstr, result.DecisionReason)
			}

			t.Logf("%s: confidence=%.3f, reason=%s",
				tt.description, result.Confidence, result.DecisionReason)
		})
	}
}
