package classification

import (
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestEmbeddingClassifier_SoftMatching tests the soft matching feature
func TestEmbeddingClassifier_SoftMatching(t *testing.T) {
	// Create test rules
	rules := []config.EmbeddingRule{
		{
			Name:                      "rule_a",
			Candidates:                []string{"candidate_a1", "candidate_a2"},
			SimilarityThreshold:       0.75,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "rule_b",
			Candidates:                []string{"candidate_b1", "candidate_b2"},
			SimilarityThreshold:       0.75,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "rule_c",
			Candidates:                []string{"candidate_c1", "candidate_c2"},
			SimilarityThreshold:       0.75,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
	}

	// Test case 1: No hard match, soft matching disabled
	t.Run("NoHardMatch_SoftMatchingDisabled", func(t *testing.T) {
		// Mock the embedding function to return controlled similarities
		originalFunc := getEmbeddingWithModelType
		defer func() { getEmbeddingWithModelType = originalFunc }()

		// Mock embeddings: all scores below threshold (0.75)
		// rule_a: 0.60, rule_b: 0.65, rule_c: 0.72
		mockEmbeddings := map[string][]float32{
			"query":        makeEmbedding(1.0, 0.0, 0.0),
			"candidate_a1": makeEmbedding(0.60, 0.0, 0.0),
			"candidate_a2": makeEmbedding(0.55, 0.0, 0.0),
			"candidate_b1": makeEmbedding(0.65, 0.0, 0.0),
			"candidate_b2": makeEmbedding(0.60, 0.0, 0.0),
			"candidate_c1": makeEmbedding(0.72, 0.0, 0.0),
			"candidate_c2": makeEmbedding(0.70, 0.0, 0.0),
		}

		getEmbeddingWithModelType = func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
			if emb, ok := mockEmbeddings[text]; ok {
				return &candle_binding.EmbeddingOutput{Embedding: emb}, nil
			}
			return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(0.0)}, nil
		}

		softMatchingDisabled := false
		hnswConfig := config.HNSWConfig{
			PreloadEmbeddings:  true,
			EnableSoftMatching: &softMatchingDisabled,
			MinScoreThreshold:  0.5,
		}

		classifier, err := NewEmbeddingClassifier(rules, hnswConfig)
		if err != nil {
			t.Fatalf("Failed to create classifier: %v", err)
		}

		ruleName, score, err := classifier.Classify("query")
		if err != nil {
			t.Fatalf("Classify failed: %v", err)
		}

		// Should return empty since no hard match and soft matching disabled
		if ruleName != "" {
			t.Errorf("Expected no match, got rule: %s with score: %.2f", ruleName, score)
		}
	})

	// Test case 2: No hard match, soft matching enabled, should return rule_c (0.72)
	t.Run("NoHardMatch_SoftMatchingEnabled", func(t *testing.T) {
		// Mock the embedding function to return controlled similarities
		originalFunc := getEmbeddingWithModelType
		defer func() { getEmbeddingWithModelType = originalFunc }()

		// Same mock embeddings as above
		mockEmbeddings := map[string][]float32{
			"query":        makeEmbedding(1.0, 0.0, 0.0),
			"candidate_a1": makeEmbedding(0.60, 0.0, 0.0),
			"candidate_a2": makeEmbedding(0.55, 0.0, 0.0),
			"candidate_b1": makeEmbedding(0.65, 0.0, 0.0),
			"candidate_b2": makeEmbedding(0.60, 0.0, 0.0),
			"candidate_c1": makeEmbedding(0.72, 0.0, 0.0),
			"candidate_c2": makeEmbedding(0.70, 0.0, 0.0),
		}

		getEmbeddingWithModelType = func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
			if emb, ok := mockEmbeddings[text]; ok {
				return &candle_binding.EmbeddingOutput{Embedding: emb}, nil
			}
			return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(0.0)}, nil
		}

		softMatchingEnabled := true
		hnswConfig := config.HNSWConfig{
			PreloadEmbeddings:  true,
			EnableSoftMatching: &softMatchingEnabled,
			MinScoreThreshold:  0.5,
		}

		classifier, err := NewEmbeddingClassifier(rules, hnswConfig)
		if err != nil {
			t.Fatalf("Failed to create classifier: %v", err)
		}

		ruleName, score, err := classifier.Classify("query")
		if err != nil {
			t.Fatalf("Classify failed: %v", err)
		}

		// Should return rule_c with score 0.72 (highest score)
		if ruleName != "rule_c" {
			t.Errorf("Expected rule_c, got: %s", ruleName)
		}
		if score < 0.71 || score > 0.73 {
			t.Errorf("Expected score ~0.72, got: %.2f", score)
		}
	})
}

// Helper function to create a simple embedding vector
func makeEmbedding(values ...float32) []float32 {
	// Pad to 768 dimensions (standard embedding size)
	result := make([]float32, 768)
	for i, v := range values {
		if i < len(result) {
			result[i] = v
		}
	}
	return result
}
