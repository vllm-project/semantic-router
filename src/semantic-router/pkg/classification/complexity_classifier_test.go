package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMatchComplexityRules(t *testing.T) {
	min := 0.2
	max := 0.8

	classifier := &Classifier{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					ComplexityRules: []config.ComplexityRule{
						{Name: "low", Min: nil, Max: &min},
						{Name: "mid", Min: &min, Max: &max},
						{Name: "high", Min: &max, Max: nil},
					},
				},
			},
		},
	}

	matched := classifier.matchComplexityRules(0.5)
	if len(matched) != 1 || matched[0] != "mid" {
		t.Fatalf("expected mid complexity match, got %v", matched)
	}

	matched = classifier.matchComplexityRules(0.9)
	if len(matched) != 1 || matched[0] != "high" {
		t.Fatalf("expected high complexity match, got %v", matched)
	}
}

func TestExtractRepoID(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "repo id",
			input:    "ilya-kolchinsky/PromptComplexityEstimator",
			expected: "ilya-kolchinsky/PromptComplexityEstimator",
		},
		{
			name:     "hf url",
			input:    "https://huggingface.co/ilya-kolchinsky/PromptComplexityEstimator",
			expected: "ilya-kolchinsky/PromptComplexityEstimator",
		},
		{
			name:     "hf url with models path",
			input:    "https://huggingface.co/models/ilya-kolchinsky/PromptComplexityEstimator",
			expected: "ilya-kolchinsky/PromptComplexityEstimator",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractRepoID(tt.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.expected {
				t.Fatalf("expected %s, got %s", tt.expected, got)
			}
		})
	}
}
