package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestComplexityClassifier_ClassifyDetailedWithImageUsesPrototypeBanks(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"analyze the root cause":        makeEmbedding(1.0, 0.0),
		"trace the failure":             makeEmbedding(0.99, 0.08),
		"quick summary":                 makeEmbedding(0.0, 1.0),
		"simple rewrite":                makeEmbedding(0.08, 0.99),
		"please analyze the root cause": makeEmbedding(0.995, 0.04),
	})

	classifier, err := NewComplexityClassifier([]config.ComplexityRule{
		{
			Name:      "needs_reasoning",
			Threshold: 0.2,
			Hard: config.ComplexityCandidates{
				Candidates: []string{"analyze the root cause", "trace the failure"},
			},
			Easy: config.ComplexityCandidates{
				Candidates: []string{"quick summary", "simple rewrite"},
			},
		},
	}, "qwen3", config.PrototypeScoringConfig{
		ClusterSimilarityThreshold: 0.98,
		MaxPrototypes:              2,
	})
	if err != nil {
		t.Fatalf("failed to create complexity classifier: %v", err)
	}

	results, err := classifier.ClassifyDetailedWithImage("please analyze the root cause", "")
	if err != nil {
		t.Fatalf("ClassifyDetailedWithImage failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected one complexity result, got %+v", results)
	}
	if results[0].Difficulty != "hard" {
		t.Fatalf("expected hard difficulty, got %+v", results[0])
	}
	if results[0].TextHardScore <= results[0].TextEasyScore {
		t.Fatalf("expected hard prototype score to beat easy score, got %+v", results[0])
	}
	if results[0].SignalSource != "text" {
		t.Fatalf("expected text signal source, got %+v", results[0])
	}
	if results[0].Confidence <= 0.2 {
		t.Fatalf("expected confidence > 0.2, got %+v", results[0])
	}
}
