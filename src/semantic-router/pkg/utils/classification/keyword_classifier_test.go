package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestKeywordClassifier(t *testing.T) {
	tests := []struct {
		name        string
		text        string
		expected    string
		rules       []config.KeywordRule // Rules specific to this test case
		expectError bool                 // Whether NewKeywordClassifier is expected to return an error
	}{
		{
			name:     "AND match",
			text:     "this text contains keyword1 and keyword2",
			expected: "test-category-1",
			rules: []config.KeywordRule{
				{
					Category: "test-category-1",
					Operator: "AND",
					Keywords: []string{"keyword1", "keyword2"},
				},
				{
					Category: "test-category-3",
					Operator: "NOR",
					Keywords: []string{"keyword5", "keyword6"},
				},
			},
		},
		{
			name:     "AND no match",
			text:     "this text contains keyword1 but not the other",
			expected: "test-category-3", // Falls through to NOR
			rules: []config.KeywordRule{
				{
					Category: "test-category-1",
					Operator: "AND",
					Keywords: []string{"keyword1", "keyword2"},
				},
				{
					Category: "test-category-3",
					Operator: "NOR",
					Keywords: []string{"keyword5", "keyword6"},
				},
			},
		},
		{
			name:     "OR match",
			text:     "this text contains keyword3",
			expected: "test-category-2",
			rules: []config.KeywordRule{
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
			},
		},
		{
			name:     "OR no match",
			text:     "this text contains nothing of interest",
			expected: "test-category-3", // Falls through to NOR
			rules: []config.KeywordRule{
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
			},
		},
		{
			name:     "NOR match",
			text:     "this text is clean",
			expected: "test-category-3",
			rules: []config.KeywordRule{
				{
					Category: "test-category-3",
					Operator: "NOR",
					Keywords: []string{"keyword5", "keyword6"},
				},
			},
		},
		{
			name:     "NOR no match",
			text:     "this text contains keyword5",
			expected: "", // Fails NOR, and no other rules match
			rules: []config.KeywordRule{
				{
					Category: "test-category-3",
					Operator: "NOR",
					Keywords: []string{"keyword5", "keyword6"},
				},
			},
		},
		{
			name:     "Case sensitive no match",
			text:     "this text contains KEYWORD3",
			expected: "test-category-3", // Fails case-sensitive OR, falls through to NOR
			rules: []config.KeywordRule{
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
			},
		},
		{
			name:     "Regex word boundary - partial match should not match",
			text:     "this is a secretary meeting",
			expected: "test-category-3", // "secret" rule (test-category-secret) won't match, falls through to NOR
			rules: []config.KeywordRule{
				{
					Category:      "test-category-secret",
					Operator:      "OR",
					Keywords:      []string{"secret"},
					CaseSensitive: false,
				},
				{
					Category: "test-category-3",
					Operator: "NOR",
					Keywords: []string{"keyword5", "keyword6"},
				},
			},
		},
		{
			name:     "Regex word boundary - exact match should match",
			text:     "this is a secret meeting",
			expected: "test-category-secret", // Should match new "secret" rule
			rules: []config.KeywordRule{
				{
					Category:      "test-category-secret",
					Operator:      "OR",
					Keywords:      []string{"secret"},
					CaseSensitive: false,
				},
				{
					Category: "test-category-3",
					Operator: "NOR",
					Keywords: []string{"keyword5", "keyword6"},
				},
			},
		},
		{
			name:     "Regex QuoteMeta - dot literal",
			text:     "this is version 1.0",
			expected: "test-category-dot", // Should match new "1.0" rule
			rules: []config.KeywordRule{
				{
					Category:      "test-category-dot",
					Operator:      "OR",
					Keywords:      []string{"1.0"},
					CaseSensitive: false,
				},
				{
					Category: "test-category-3",
					Operator: "NOR",
					Keywords: []string{"keyword5", "keyword6"},
				},
			},
		},
		{
			name:     "Regex QuoteMeta - asterisk literal",
			text:     "match this text with a * wildcard",
			expected: "test-category-asterisk", // Should match new "*" rule
			rules: []config.KeywordRule{
				{
					Category:      "test-category-asterisk",
					Operator:      "OR",
					Keywords:      []string{"*"},
					CaseSensitive: false,
				},
				{
					Category: "test-category-3",
					Operator: "NOR",
					Keywords: []string{"keyword5", "keyword6"},
				},
			},
		},
		{
			name: "Unsupported operator should return error",
			rules: []config.KeywordRule{
				{
					Category: "bad-operator",
					Operator: "UNKNOWN", // Invalid operator
					Keywords: []string{"test"},
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			classifier, err := NewKeywordClassifier(tt.rules)

			if tt.expectError {
				if err == nil {
					t.Fatalf("expected an error during initialization, but got none")
				}
				return // Test passed if error was expected and received
			}

			if err != nil {
				t.Fatalf("Failed to initialize KeywordClassifier: %v", err)
			}

			category, _, err := classifier.Classify(tt.text)
			if err != nil {
				t.Fatalf("unexpected error from Classify: %v", err)
			}
			if category != tt.expected {
				t.Errorf("expected category %q, but got %q", tt.expected, category)
			}
		})
	}
}
