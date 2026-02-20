package classification

import (
	"errors"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPreferenceClassifier_ContrastiveFewShot(t *testing.T) {
	reset := SetEmbeddingFuncForTests(func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		lower := strings.ToLower(text)
		switch {
		case strings.Contains(lower, "bug") || strings.Contains(lower, "fix"):
			return &candle_binding.EmbeddingOutput{Embedding: []float32{0.2, 0.9}}, nil
		case strings.Contains(lower, "code"):
			return &candle_binding.EmbeddingOutput{Embedding: []float32{0.9, 0.1}}, nil
		default:
			return &candle_binding.EmbeddingOutput{Embedding: []float32{0.5, 0.5}}, nil
		}
	})
	defer reset()

	rules := []config.PreferenceRule{
		{Name: "code_generation", Description: "Writes code", Examples: []string{"write python code"}},
		{Name: "bug_fixing", Description: "Finds and fixes bugs", Examples: []string{"fix this bug"}},
	}

	localCfg := &config.PreferenceModelConfig{
		UseContrastive: true,
		EmbeddingModel: "qwen3",
	}

	classifier, err := NewPreferenceClassifier(nil, rules, localCfg)
	if err != nil {
		t.Fatalf("failed to create contrastive preference classifier: %v", err)
	}

	conversation := `[{"role":"user","content":"can you help fix this bug?"}]`
	result, err := classifier.Classify(conversation)
	if err != nil {
		t.Fatalf("contrastive classification failed: %v", err)
	}

	if result.Preference != "bug_fixing" {
		t.Fatalf("expected bug_fixing preference, got %s", result.Preference)
	}

	if result.Confidence <= 0 {
		t.Fatalf("expected positive confidence score, got %f", result.Confidence)
	}
}

func TestContrastivePreferenceClassifier_UsesDescriptionsWhenNoExamples(t *testing.T) {
	reset := SetEmbeddingFuncForTests(func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		switch text {
		case "Writes code":
			return &candle_binding.EmbeddingOutput{Embedding: []float32{1, 0}}, nil
		case "Fixes bugs":
			return &candle_binding.EmbeddingOutput{Embedding: []float32{0, 1}}, nil
		case "please write code":
			return &candle_binding.EmbeddingOutput{Embedding: []float32{1, 0}}, nil
		default:
			return &candle_binding.EmbeddingOutput{Embedding: []float32{0.1, 0.1}}, nil
		}
	})
	defer reset()

	rules := []config.PreferenceRule{
		{Name: "code_generation", Description: "Writes code"},
		{Name: "bug_fixing", Description: "Fixes bugs"},
	}

	localCfg := &config.PreferenceModelConfig{
		UseContrastive: true,
	}

	classifier, err := NewPreferenceClassifier(nil, rules, localCfg)
	if err != nil {
		t.Fatalf("failed to create contrastive classifier: %v", err)
	}

	conversation := `[{"role":"user","content":"please write code"}]`
	result, err := classifier.Classify(conversation)
	if err != nil {
		t.Fatalf("classification failed: %v", err)
	}

	if result.Preference != "code_generation" {
		t.Fatalf("expected code_generation, got %s", result.Preference)
	}
}

func TestContrastivePreferenceClassifier_MaxExamplesLimit(t *testing.T) {
	reset := SetEmbeddingFuncForTests(func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		if strings.Contains(text, "extra") {
			return nil, errors.New("should not embed extra example")
		}
		return &candle_binding.EmbeddingOutput{Embedding: []float32{1, 0}}, nil
	})
	defer reset()

	rules := []config.PreferenceRule{
		{Name: "code_generation", Description: "Writes code", Examples: []string{"first example", "extra example"}},
	}

	localCfg := &config.PreferenceModelConfig{UseContrastive: true, MaxExamples: 1}

	if _, err := NewPreferenceClassifier(nil, rules, localCfg); err != nil {
		t.Fatalf("unexpected error creating classifier with max_examples: %v", err)
	}
}

func TestContrastivePreferenceClassifier_NoExamplesError(t *testing.T) {
	rules := []config.PreferenceRule{{Name: "code_generation"}}
	localCfg := &config.PreferenceModelConfig{UseContrastive: true}

	if _, err := NewPreferenceClassifier(nil, rules, localCfg); err == nil {
		t.Fatalf("expected error when no descriptions or examples are provided")
	}
}

func TestContrastivePreferenceClassifier_EmptyText(t *testing.T) {
	reset := SetEmbeddingFuncForTests(func(text string, modelType string, targetDim int) (*candle_binding.EmbeddingOutput, error) {
		return &candle_binding.EmbeddingOutput{Embedding: []float32{0.1, 0.2}}, nil
	})
	defer reset()

	rules := []config.PreferenceRule{{Name: "code_generation", Description: "Writes code"}}
	localCfg := &config.PreferenceModelConfig{UseContrastive: true}

	classifier, err := NewPreferenceClassifier(nil, rules, localCfg)
	if err != nil {
		t.Fatalf("failed to create contrastive classifier: %v", err)
	}

	if _, err := classifier.Classify(""); err == nil {
		t.Fatalf("expected error for empty text")
	}
}
