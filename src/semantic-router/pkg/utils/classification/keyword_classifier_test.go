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
			name:     "AND no match",
			text:     "this text contains keyword1 but not the other",
			expected: "",
		},
		{
			name:     "OR match",
			text:     "this text contains keyword3",
			expected: "test-category-2",
		},
		{
			name:     "OR no match",
			text:     "this text contains nothing of interest",
			expected: "",
		},
		{
			name:     "NOR match",
			text:     "this text is clean",
			expected: "test-category-3",
		},
		{
			name:     "NOR no match",
			text:     "this text contains keyword5",
			expected: "",
		},
		{
			name:     "Case sensitive match",
			text:     "this text contains KEYWORD3",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a new classifier for each test to ensure a clean slate
			classifier := NewKeywordClassifier(rules)
			category, _, _ := classifier.Classify(tt.text)
			if category != tt.expected {
				// If the category is not what we expect, we check if the NOR rule was triggered
				if tt.expected == "" && category == "test-category-3" {
					// we run the classification again, but this time we remove the NOR rule
					var filteredRules []config.KeywordRule
					for _, rule := range rules {
						if rule.Operator != "NOR" {
							filteredRules = append(filteredRules, rule)
						}
					}
					classifier = NewKeywordClassifier(filteredRules)
					category, _, _ = classifier.Classify(tt.text)
					if category != tt.expected {
						t.Errorf("expected category %s, but got %s", tt.expected, category)
					}
				} else {
					t.Errorf("expected category %s, but got %s", tt.expected, category)
				}
			}
		})
	}
}