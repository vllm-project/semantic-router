package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestKeywordClassifier(t *testing.T) {
	rules := []config.KeywordRule{
		{
			Category: "test-category-1",
			Operator: "AND",
			Keywords: []string{"keyword1", "keyword2"},
		},
		{
			Category:      "test-category-2",
			Operator:      "OR",
			Keywords:      []string{"keyword3", "keyword4"},
			CaseSensitive: true,
		},
		{
			Category: "test-category-3",
			Operator: "NOR",
			Keywords: []string{"keyword5", "keyword6"},
		},
	}

	tests := []struct {
		name     string
		text     string
		expected string
	}{
		{
			name:     "AND match",
			text:     "this text contains keyword1 and keyword2",
			expected: "test-category-1",
		},
		{
			name: "AND no match",
			// This text does not match the AND rule. It also does not contain "keyword5" or "keyword6",
			// so the NOR rule will match as a fallback.
			text:     "this text contains keyword1 but not the other",
			expected: "test-category-3",
		},
		{
			name:     "OR match",
			text:     "this text contains keyword3",
			expected: "test-category-2",
		},
		{
			name: "OR no match",
			// This text does not match the OR rule. It also does not contain "keyword5" or "keyword6",
			// so the NOR rule will match as a fallback.
			text:     "this text contains nothing of interest",
			expected: "test-category-3",
		},
		{
			name:     "NOR match",
			text:     "this text is clean",
			expected: "test-category-3",
		},
		{
			name: "NOR no match",
			// This text contains "keyword5", so the NOR rule will NOT match.
			// Since no other rules match, the result should be empty.
			text:     "this text contains keyword5",
			expected: "",
		},
		{
			name: "Case sensitive no match",
			// This text does not match the case-sensitive OR rule. It also does not contain "keyword5" or "keyword6",
			// so the NOR rule will match as a fallback.
			text:     "this text contains KEYWORD3",
			expected: "test-category-3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a new classifier for each test to ensure a clean slate
			classifier := NewKeywordClassifier(rules)
			category, _, err := classifier.Classify(tt.text)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if category != tt.expected {
				t.Errorf("expected category %q, but got %q", tt.expected, category)
			}
		})
	}
}
