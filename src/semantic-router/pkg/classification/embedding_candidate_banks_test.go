package classification

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestEmbeddingCandidateBanksUseOwnedModalityAndMargin(t *testing.T) {
	same := testImageURI("shared")
	var mu sync.Mutex
	textCalls, imageCalls := map[string]int{}, map[string]int{}
	provider := &testEmbeddingProvider{
		text: func(_ context.Context, value string, options embedding.Options) ([]float32, error) {
			if options.Dimension != 2 {
				return nil, fmt.Errorf("dimension: %d", options.Dimension)
			}
			mu.Lock()
			textCalls[value]++
			mu.Unlock()
			if value == same {
				return []float32{0, 1}, nil
			}
			return []float32{1, 0}, nil
		},
		image: func(_ context.Context, value []byte, dim int) ([]float32, error) {
			if dim != 2 {
				return nil, fmt.Errorf("dimension: %d", dim)
			}
			mu.Lock()
			imageCalls[string(value)]++
			mu.Unlock()
			if string(value) == "shared" {
				return []float32{1, 0}, nil
			}
			return []float32{0.6, 0.8}, nil
		},
	}
	rules := []config.EmbeddingRule{{Name: "contrastive", QueryModality: config.QueryModalityImage, Candidates: []string{same}, ImageCandidates: []string{same}, NegativeCandidates: []string{same}, NegativeImageCandidates: []string{testImageURI("negative")}, SimilarityThreshold: 0.3, AggregationMethodConfiged: config.AggregationMethodMax}}
	c, err := NewEmbeddingClassifierWithProvider(rules, config.HNSWConfig{TargetDimension: 2, PreloadEmbeddings: true}, provider)
	if err != nil {
		t.Fatal(err)
	}
	if err = c.WarmupCandidateEmbeddings(); err != nil {
		t.Fatal(err)
	}
	if textCalls[same] != 1 || imageCalls["shared"] != 1 || imageCalls["negative"] != 1 {
		t.Fatalf("actual-modality dedup: text=%v image=%v", textCalls, imageCalls)
	}
	result, err := c.ClassifyDetailedMultimodal(config.QueryModalityImage, same)
	if err != nil {
		t.Fatal(err)
	}
	score := result.Scores[0]
	if math.Abs(score.Score-0.4) > 1e-6 || score.PositiveScore != 1 || math.Abs(score.NegativeScore-0.6) > 1e-6 || len(result.Matches) != 1 {
		t.Fatalf("wrong margin: %+v", result)
	}
	// A second classifier owns a different provider and never inherits bank state.
	other, err := NewEmbeddingClassifierWithProvider(rules, config.HNSWConfig{TargetDimension: 2, PreloadEmbeddings: true}, &testEmbeddingProvider{})
	if err != nil {
		t.Fatal(err)
	}
	if err = other.WarmupCandidateEmbeddings(); err == nil {
		t.Fatal("unprepared provider unexpectedly borrowed candidate vectors")
	}
}

func TestEmbeddingCandidateBankFailureIsAtomic(t *testing.T) {
	for _, badDimension := range []bool{false, true} {
		p := &testEmbeddingProvider{
			text: func(context.Context, string, embedding.Options) ([]float32, error) { return []float32{1, 0}, nil },
			image: func(context.Context, []byte, int) ([]float32, error) {
				if badDimension {
					return []float32{1, 0, 0}, nil
				}
				return nil, fmt.Errorf("decode failed")
			},
		}
		c, err := NewEmbeddingClassifierWithProvider([]config.EmbeddingRule{{Name: "mixed", Candidates: []string{"text"}, NegativeImageCandidates: []string{testImageURI("bad")}}}, config.HNSWConfig{PreloadEmbeddings: true}, p)
		if err != nil {
			t.Fatal(err)
		}
		if err = c.WarmupCandidateEmbeddings(); err == nil {
			t.Fatal("expected error")
		}
		if len(c.candidateEmbeddings) != 0 || len(c.imageCandidateEmbeddings) != 0 || c.preloadComplete {
			t.Fatal("failed preload published a partial bank")
		}
	}
}

func TestEmbeddingMarginConfidencePreservesRawScore(t *testing.T) {
	c := &Classifier{}
	result := &SignalResults{SignalValues: map[string]float64{}, SignalConfidences: map[string]float64{}}
	c.recordEmbeddingResult(result, &EmbeddingClassificationResult{Scores: []EmbeddingRuleScore{{Name: "margin", Score: 0.02, PositiveScore: 0.82, NegativeScore: 0.8, Contrastive: true}}, Matches: []MatchedRule{{RuleName: "margin", Score: 0.02, Method: "hard"}}}, 0, 0)
	if result.SignalValues["embedding:margin"] != 0.02 || result.SignalValues["embedding:margin:positive"] != 0.82 || result.SignalValues["embedding:margin:negative"] != 0.8 {
		t.Fatalf("lost raw evidence: %+v", result.SignalValues)
	}
	if math.Abs(result.SignalConfidences["embedding:margin"]-0.505) > 1e-12 {
		t.Fatal("raw margin leaked into bounded confidence")
	}
	if math.Abs(embeddingSignalConfidence(0.02, true)-0.505) > 1e-12 || embeddingSignalConfidence(-2, true) != 0 || embeddingSignalConfidence(2, true) != 1 {
		t.Fatal("invalid margin confidence mapping")
	}
}
