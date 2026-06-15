package classification

import (
	"runtime"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestComplexityWorkerCountSerializesDefaultCandleRuntime(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "")

	if got := complexityWorkerCount(60); got != 1 {
		t.Fatalf("complexityWorkerCount() = %d, want 1 for default candle runtime", got)
	}
}

func TestComplexityWorkerCountSerializesExplicitCandleRuntime(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "candle")

	if got := complexityWorkerCount(60); got != 1 {
		t.Fatalf("complexityWorkerCount() = %d, want 1 for explicit candle runtime", got)
	}
}

func TestComplexityWorkerCountBoundsNonCandleRuntime(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", "openvino")

	got := complexityWorkerCount(3)
	if got != 3 {
		t.Fatalf("complexityWorkerCount() = %d, want task-count bound 3", got)
	}

	got = complexityWorkerCount(1000)
	want := runtime.NumCPU() * 2
	if got != want {
		t.Fatalf("complexityWorkerCount() = %d, want CPU bound %d", got, want)
	}
}

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

func TestComplexityClassifier_ClassifyDetailedWithImageWithoutCache(t *testing.T) {
	stubEmbeddingLookup(t, map[string][]float32{
		"hard text":     makeEmbedding(0.5, 0.5),
		"easy text":     makeEmbedding(0.5, 0.5),
		"visual query":  makeEmbedding(0.5, 0.5),
		"fallback text": makeEmbedding(0.5, 0.5),
	})
	stubMultiModalImageLookup(t, map[string][]float32{
		"hard-image":    makeEmbedding(1.0, 0.0),
		"easy-image":    makeEmbedding(0.0, 1.0),
		"request-image": makeEmbedding(1.0, 0.0),
	})

	originalMMText := getMultiModalTextEmbedding
	getMultiModalTextEmbedding = func(text string, targetDim int) ([]float32, error) {
		return makeEmbedding(0.5, 0.5), nil
	}
	t.Cleanup(func() { getMultiModalTextEmbedding = originalMMText })

	classifier, err := NewComplexityClassifier([]config.ComplexityRule{
		{
			Name:      "visual_reasoning",
			Threshold: 0.2,
			Hard: config.ComplexityCandidates{
				Candidates:      []string{"hard text"},
				ImageCandidates: []string{"hard-image"},
			},
			Easy: config.ComplexityCandidates{
				Candidates:      []string{"easy text"},
				ImageCandidates: []string{"easy-image"},
			},
		},
	}, "qwen3", config.PrototypeScoringConfig{
		ClusterSimilarityThreshold: 0.98,
		MaxPrototypes:              2,
	})
	if err != nil {
		t.Fatalf("failed to create complexity classifier: %v", err)
	}

	results, err := classifier.ClassifyDetailedWithImage("visual query", "request-image")
	if err != nil {
		t.Fatalf("ClassifyDetailedWithImage failed without cache: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected one complexity result, got %+v", results)
	}
	if results[0].Difficulty != "hard" {
		t.Fatalf("expected image-backed hard difficulty, got %+v", results[0])
	}
	if results[0].SignalSource != "image" {
		t.Fatalf("expected image signal source, got %+v", results[0])
	}
}
