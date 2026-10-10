package classification

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestReaskClassifier_ClassifyStopsEmbeddingAfterAllRulesFail(t *testing.T) {
	embeddingCalls := 0
	provider := newTestTextProvider(func(text string) ([]float32, error) {
		embeddingCalls++
		if text == "current" {
			return makeEmbedding(1, 0), nil
		}
		return makeEmbedding(0, 1), nil
	})

	priorUserTurns := make([]string, 50)
	for index := range priorUserTurns {
		priorUserTurns[index] = fmt.Sprintf("unrelated earlier question %d", index)
	}

	classifier, err := NewReaskClassifierWithProvider([]config.ReaskRule{
		{
			Name:          "likely_dissatisfied",
			Threshold:     0.9,
			LookbackTurns: 1,
		},
		{
			Name:          "default_threshold",
			LookbackTurns: 1,
		},
	}, "test-model", provider)
	if err != nil {
		t.Fatalf("NewReaskClassifierWithProvider(, provider) error = %v", err)
	}

	matches, err := classifier.Classify("current", priorUserTurns)
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 0 {
		t.Fatalf("expected no matches, got %+v", matches)
	}
	if embeddingCalls != 2 {
		t.Fatalf("embedding calls = %d, want 2", embeddingCalls)
	}
}

func TestReaskClassifier_ClassifyContinuesUntilLowestThresholdFails(t *testing.T) {
	embeddingCalls := make(map[string]int)
	embeddings := map[string][]float32{
		"current":                       makeEmbedding(1, 0),
		"recent permissive match":       makeEmbedding(0.8, 0.6),
		"older all-rules failure":       makeEmbedding(0, 1),
		"oldest should not be embedded": makeEmbedding(1, 0),
	}
	provider := newTestTextProvider(func(text string) ([]float32, error) {
		embeddingCalls[text]++
		return embeddings[text], nil
	})

	classifier, err := NewReaskClassifierWithProvider([]config.ReaskRule{
		{
			Name:          "strict",
			Threshold:     0.9,
			LookbackTurns: 1,
		},
		{
			Name:          "permissive",
			Threshold:     0.7,
			LookbackTurns: 1,
		},
	}, "test-model", provider)
	if err != nil {
		t.Fatalf("NewReaskClassifierWithProvider(, provider) error = %v", err)
	}

	matches, err := classifier.Classify("current", []string{
		"oldest should not be embedded",
		"older all-rules failure",
		"recent permissive match",
	})
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 1 || matches[0].RuleName != "permissive" {
		t.Fatalf("matches = %+v, want permissive", matches)
	}
	if embeddingCalls["recent permissive match"] != 1 {
		t.Fatalf("recent permissive match embedding calls = %d, want 1", embeddingCalls["recent permissive match"])
	}
	if embeddingCalls["older all-rules failure"] != 1 {
		t.Fatalf("older all-rules failure embedding calls = %d, want 1", embeddingCalls["older all-rules failure"])
	}
	if embeddingCalls["oldest should not be embedded"] != 0 {
		t.Fatalf("oldest embedding calls = %d, want 0", embeddingCalls["oldest should not be embedded"])
	}
}
