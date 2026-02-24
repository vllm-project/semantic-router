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

// TestEmbeddingClassifier_ClassifyAll tests that ClassifyAll returns multiple matched rules
func TestEmbeddingClassifier_ClassifyAll(t *testing.T) {
	rules := []config.EmbeddingRule{
		{
			Name:                      "ai",
			Candidates:                []string{"machine learning", "neural network"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "programming",
			Candidates:                []string{"python code", "software development"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
		{
			Name:                      "cooking",
			Candidates:                []string{"recipe", "ingredients"},
			SimilarityThreshold:       0.70,
			AggregationMethodConfiged: config.AggregationMethodMax,
		},
	}

	t.Run("MultipleHardMatches", func(t *testing.T) {
		originalFunc := getEmbeddingWithModelType
		defer func() { getEmbeddingWithModelType = originalFunc }()

		// Query is similar to "ai" and "programming" candidates, but NOT "cooking"
		// Note: cosineSimilarity() computes dot product only (no L2 normalization),
		// so values must be chosen such that dot(query, candidate) >= threshold (0.70)
		mockEmbeddings := map[string][]float32{
			"TensorFlow pipeline":  makeEmbedding(0.90, 0.85, 0.10),
			"machine learning":     makeEmbedding(0.85, 0.0, 0.0), // dot=0.90*0.85=0.765 ✓
			"neural network":       makeEmbedding(0.80, 0.0, 0.0), // dot=0.90*0.80=0.720 ✓
			"python code":          makeEmbedding(0.0, 0.90, 0.0), // dot=0.85*0.90=0.765 ✓
			"software development": makeEmbedding(0.0, 0.85, 0.0), // dot=0.85*0.85=0.723 ✓
			"recipe":               makeEmbedding(0.0, 0.0, 0.30), // dot=0.10*0.30=0.030 ✗
			"ingredients":          makeEmbedding(0.0, 0.0, 0.25), // dot=0.10*0.25=0.025 ✗
		}

		getEmbeddingWithModelType = func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
			if emb, ok := mockEmbeddings[text]; ok {
				return &candle_binding.EmbeddingOutput{Embedding: emb}, nil
			}
			return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(0.0)}, nil
		}

		hnswConfig := config.HNSWConfig{PreloadEmbeddings: true}
		classifier, err := NewEmbeddingClassifier(rules, hnswConfig)
		if err != nil {
			t.Fatalf("Failed to create classifier: %v", err)
		}

		matched, err := classifier.ClassifyAll("TensorFlow pipeline")
		if err != nil {
			t.Fatalf("ClassifyAll failed: %v", err)
		}

		// Should return 2 hard matches: "ai" and "programming", but NOT "cooking"
		if len(matched) != 2 {
			t.Fatalf("Expected 2 matches, got %d: %+v", len(matched), matched)
		}

		ruleNames := map[string]bool{}
		for _, m := range matched {
			ruleNames[m.RuleName] = true
			if m.Method != "hard" {
				t.Errorf("Expected hard match for %s, got %s", m.RuleName, m.Method)
			}
			if m.Score < 0.70 {
				t.Errorf("Expected score >= 0.70 for %s, got %.4f", m.RuleName, m.Score)
			}
		}
		if !ruleNames["ai"] {
			t.Error("Expected 'ai' rule to match")
		}
		if !ruleNames["programming"] {
			t.Error("Expected 'programming' rule to match")
		}
		if ruleNames["cooking"] {
			t.Error("'cooking' should NOT match")
		}
	})

	t.Run("ClassifyAll_ConsistentWithClassify", func(t *testing.T) {
		// When only one rule matches, Classify and ClassifyAll should agree
		originalFunc := getEmbeddingWithModelType
		defer func() { getEmbeddingWithModelType = originalFunc }()

		mockEmbeddings := map[string][]float32{
			"query":                makeEmbedding(1.0, 0.0, 0.0),
			"machine learning":     makeEmbedding(0.85, 0.0, 0.0),
			"neural network":       makeEmbedding(0.80, 0.0, 0.0),
			"python code":          makeEmbedding(0.30, 0.0, 0.0),
			"software development": makeEmbedding(0.25, 0.0, 0.0),
			"recipe":               makeEmbedding(0.10, 0.0, 0.0),
			"ingredients":          makeEmbedding(0.05, 0.0, 0.0),
		}

		getEmbeddingWithModelType = func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
			if emb, ok := mockEmbeddings[text]; ok {
				return &candle_binding.EmbeddingOutput{Embedding: emb}, nil
			}
			return &candle_binding.EmbeddingOutput{Embedding: makeEmbedding(0.0)}, nil
		}

		hnswConfig := config.HNSWConfig{PreloadEmbeddings: true}
		classifier, err := NewEmbeddingClassifier(rules, hnswConfig)
		if err != nil {
			t.Fatalf("Failed to create classifier: %v", err)
		}

		// ClassifyAll should return just "ai"
		matched, err := classifier.ClassifyAll("query")
		if err != nil {
			t.Fatalf("ClassifyAll failed: %v", err)
		}
		if len(matched) != 1 || matched[0].RuleName != "ai" {
			t.Fatalf("Expected single 'ai' match, got: %+v", matched)
		}

		// Classify should return the same rule and score
		ruleName, score, err := classifier.Classify("query")
		if err != nil {
			t.Fatalf("Classify failed: %v", err)
		}
		if ruleName != matched[0].RuleName {
			t.Errorf("Classify returned %q but ClassifyAll returned %q", ruleName, matched[0].RuleName)
		}
		if score != matched[0].Score {
			t.Errorf("Classify score %.4f != ClassifyAll score %.4f", score, matched[0].Score)
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
