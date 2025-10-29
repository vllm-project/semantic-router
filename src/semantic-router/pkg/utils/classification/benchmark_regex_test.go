package classification

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// --- Current Regex Implementation ---
// This uses the currently modified keyword_classifier.go with regex matching.

func BenchmarkKeywordClassifierRegex(b *testing.B) {
	rulesConfig := []config.KeywordRule{
		{Category: "cat-and", Operator: "AND", Keywords: []string{"apple", "banana"}, CaseSensitive: false},
		{Category: "cat-or", Operator: "OR", Keywords: []string{"orange", "grape"}, CaseSensitive: true},
		{Category: "cat-nor", Operator: "NOR", Keywords: []string{"disallowed"}, CaseSensitive: false},
	}

	testTextAndMatch := "I like apple and banana"
	testTextOrMatch := "I prefer orange juice"
	testTextNorMatch := "This text is clean"
	testTextNoMatch := "Something else entirely with disallowed words" // To fail all above for final no match

	classifierRegex, err := NewKeywordClassifier(rulesConfig)
	if err != nil {
		b.Fatalf("Failed to initialize KeywordClassifier: %v", err)
	}

	b.Run("Regex_AND_Match", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierRegex.Classify(testTextAndMatch)
		}
	})
	b.Run("Regex_OR_Match", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierRegex.Classify(testTextOrMatch)
		}
	})
	b.Run("Regex_NOR_Match", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierRegex.Classify(testTextNorMatch)
		}
	})
	b.Run("Regex_No_Match", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierRegex.Classify(testTextNoMatch)
		}
	})

	// Scenario: Keywords with varying lengths
	rulesConfigLongKeywords := []config.KeywordRule{
		{Category: "long-kw", Operator: "OR", Keywords: []string{"supercalifragilisticexpialidocious", "pneumonoultramicroscopicsilicovolcanoconiosis"}, CaseSensitive: false},
	}
	classifierLongKeywords, err := NewKeywordClassifier(rulesConfigLongKeywords)
	if err != nil {
		b.Fatalf("Failed to initialize classifierLongKeywords: %v", err)
	}
	b.Run("Regex_LongKeywords", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierLongKeywords.Classify("This text contains supercalifragilisticexpialidocious and other long words.")
		}
	})

	// Scenario: Texts with varying lengths
	rulesConfigShortText := []config.KeywordRule{
		{Category: "short-text", Operator: "OR", Keywords: []string{"short"}, CaseSensitive: false},
	}
	classifierShortText, err := NewKeywordClassifier(rulesConfigShortText)
	if err != nil {
		b.Fatalf("Failed to initialize classifierShortText: %v", err)
	}
	b.Run("Regex_ShortText", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierShortText.Classify("short")
		}
	})

	rulesConfigLongText := []config.KeywordRule{
		{Category: "long-text", Operator: "OR", Keywords: []string{"endword"}, CaseSensitive: false},
	}
	classifierLongText, err := NewKeywordClassifier(rulesConfigLongText)
	if err != nil {
		b.Fatalf("Failed to initialize classifierLongText: %v", err)
	}
	longText := strings.Repeat("word ", 1000) + "endword" // Text of ~5000 characters
	b.Run("Regex_LongText", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierLongText.Classify(longText)
		}
	})

	// Scenario: Rules with a larger number of keywords
	manyKeywords := make([]string, 100)
	for i := 0; i < 100; i++ {
		manyKeywords[i] = fmt.Sprintf("keyword%d", i)
	}
	rulesConfigManyKeywords := []config.KeywordRule{
		{Category: "many-kw", Operator: "OR", Keywords: manyKeywords, CaseSensitive: false},
	}
	classifierManyKeywords, err := NewKeywordClassifier(rulesConfigManyKeywords)
	if err != nil {
		b.Fatalf("Failed to initialize classifierManyKeywords: %v", err)
	}
	b.Run("Regex_ManyKeywords", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierManyKeywords.Classify("This text contains keyword99")
		}
	})

	// Scenario: Keywords with many escaped characters
	rulesConfigComplexKeywords := []config.KeywordRule{
		{Category: "complex-kw", Operator: "OR", Keywords: []string{"user.name@domain.com", "C:\\Program Files\\"}, CaseSensitive: false},
	}
	classifierComplexKeywords, err := NewKeywordClassifier(rulesConfigComplexKeywords)
	if err != nil {
		b.Fatalf("Failed to initialize classifierComplexKeywords: %v", err)
	}
	b.Run("Regex_ComplexKeywords", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = classifierComplexKeywords.Classify("Please send to user.name@domain.com or check C:\\Program Files\\")
		}
	})
}
