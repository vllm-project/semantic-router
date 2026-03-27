package classification

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type keywordClassifierTestCase struct {
	name        string
	text        string
	expected    string
	rules       []config.KeywordRule
	expectError bool
}

var keywordClassifierTests = []keywordClassifierTestCase{
	{
		name:     "AND match",
		text:     "this text contains keyword1 and keyword2",
		expected: "test-category-1",
		rules: []config.KeywordRule{
			{Name: "test-category-1", Operator: "AND", Keywords: []string{"keyword1", "keyword2"}},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "AND no match",
		text:     "this text contains keyword1 but not the other",
		expected: "test-category-3",
		rules: []config.KeywordRule{
			{Name: "test-category-1", Operator: "AND", Keywords: []string{"keyword1", "keyword2"}},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "OR match",
		text:     "this text contains keyword3",
		expected: "test-category-2",
		rules: []config.KeywordRule{
			{Name: "test-category-2", Operator: "OR", Keywords: []string{"keyword3", "keyword4"}, CaseSensitive: true},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "OR no match",
		text:     "this text contains nothing of interest",
		expected: "test-category-3",
		rules: []config.KeywordRule{
			{Name: "test-category-2", Operator: "OR", Keywords: []string{"keyword3", "keyword4"}, CaseSensitive: true},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "NOR match",
		text:     "this text is clean",
		expected: "test-category-3",
		rules: []config.KeywordRule{
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "NOR no match",
		text:     "this text contains keyword5",
		expected: "",
		rules: []config.KeywordRule{
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "Case sensitive no match",
		text:     "this text contains KEYWORD3",
		expected: "test-category-3",
		rules: []config.KeywordRule{
			{Name: "test-category-2", Operator: "OR", Keywords: []string{"keyword3", "keyword4"}, CaseSensitive: true},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "Regex word boundary - partial match should not match",
		text:     "this is a secretary meeting",
		expected: "test-category-3",
		rules: []config.KeywordRule{
			{Name: "test-category-secret", Operator: "OR", Keywords: []string{"secret"}},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "Regex word boundary - exact match should match",
		text:     "this is a secret meeting",
		expected: "test-category-secret",
		rules: []config.KeywordRule{
			{Name: "test-category-secret", Operator: "OR", Keywords: []string{"secret"}},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "Regex QuoteMeta - dot literal",
		text:     "this is version 1.0",
		expected: "test-category-dot",
		rules: []config.KeywordRule{
			{Name: "test-category-dot", Operator: "OR", Keywords: []string{"1.0"}},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "Regex QuoteMeta - asterisk literal",
		text:     "match this text with a * wildcard",
		expected: "test-category-asterisk",
		rules: []config.KeywordRule{
			{Name: "test-category-asterisk", Operator: "OR", Keywords: []string{"*"}},
			{Name: "test-category-3", Operator: "NOR", Keywords: []string{"keyword5", "keyword6"}},
		},
	},
	{
		name:     "Explicit regex method matches AWS key pattern",
		text:     "My AWS key is AKIAIOSFODNN7EXAMPLE and it should be caught",
		expected: "secrets",
		rules: []config.KeywordRule{
			{
				Name:          "secrets",
				Operator:      "OR",
				Method:        "regex",
				Keywords:      []string{`AKIA[0-9A-Z]{16}`},
				CaseSensitive: false,
			},
		},
	},
	{
		name: "Unsupported operator should return error",
		rules: []config.KeywordRule{
			{Name: "bad-operator", Operator: "UNKNOWN", Keywords: []string{"test"}},
		},
		expectError: true,
	},
}

func TestKeywordClassifier(t *testing.T) {
	for _, tt := range keywordClassifierTests {
		t.Run(tt.name, func(t *testing.T) {
			classifier, err := NewKeywordClassifier(tt.rules)

			if tt.expectError {
				if err == nil {
					t.Fatalf("expected an error during initialization, but got none")
				}
				return
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

type keywordBenchmarkScenario struct {
	name  string
	rules []config.KeywordRule
	text  string
}

func keywordBenchmarkScenarios() []keywordBenchmarkScenario {
	manyKeywords := make([]string, 100)
	for i := 0; i < len(manyKeywords); i++ {
		manyKeywords[i] = fmt.Sprintf("keyword%d", i)
	}

	return []keywordBenchmarkScenario{
		{
			name:  "Regex_AND_Match",
			rules: []config.KeywordRule{{Name: "cat-and", Operator: "AND", Keywords: []string{"apple", "banana"}}},
			text:  "I like apple and banana",
		},
		{
			name:  "Regex_OR_Match",
			rules: []config.KeywordRule{{Name: "cat-or", Operator: "OR", Keywords: []string{"orange", "grape"}, CaseSensitive: true}},
			text:  "I prefer orange juice",
		},
		{
			name:  "Regex_NOR_Match",
			rules: []config.KeywordRule{{Name: "cat-nor", Operator: "NOR", Keywords: []string{"disallowed"}}},
			text:  "This text is clean",
		},
		{
			name:  "Regex_No_Match",
			rules: []config.KeywordRule{{Name: "cat-nor", Operator: "NOR", Keywords: []string{"disallowed"}}},
			text:  "Something else entirely with disallowed words",
		},
		{
			name:  "Regex_LongKeywords",
			rules: []config.KeywordRule{{Name: "long-kw", Operator: "OR", Keywords: []string{"supercalifragilisticexpialidocious", "pneumonoultramicroscopicsilicovolcanoconiosis"}}},
			text:  "This text contains supercalifragilisticexpialidocious and other long words.",
		},
		{
			name:  "Regex_ShortText",
			rules: []config.KeywordRule{{Name: "short-text", Operator: "OR", Keywords: []string{"short"}}},
			text:  "short",
		},
		{
			name:  "Regex_LongText",
			rules: []config.KeywordRule{{Name: "long-text", Operator: "OR", Keywords: []string{"endword"}}},
			text:  strings.Repeat("word ", 1000) + "endword",
		},
		{
			name:  "Regex_ManyKeywords",
			rules: []config.KeywordRule{{Name: "many-kw", Operator: "OR", Keywords: manyKeywords}},
			text:  "This text contains keyword99",
		},
		{
			name:  "Regex_ComplexKeywords",
			rules: []config.KeywordRule{{Name: "complex-kw", Operator: "OR", Keywords: []string{"user.name@domain.com", `C:\Program Files\`}}},
			text:  `Please send to user.name@domain.com or check C:\Program Files\`,
		},
	}
}

func BenchmarkKeywordClassifierRegex(b *testing.B) {
	for _, scenario := range keywordBenchmarkScenarios() {
		b.Run(scenario.name, func(b *testing.B) {
			classifier, err := NewKeywordClassifier(scenario.rules)
			if err != nil {
				b.Fatalf("Failed to initialize KeywordClassifier: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _ = classifier.Classify(scenario.text)
			}
		})
	}
}
