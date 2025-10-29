package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
			classifierRegex.Classify(testTextAndMatch)
		}
	})
	b.Run("Regex_OR_Match", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			classifierRegex.Classify(testTextOrMatch)
		}
	})
	b.Run("Regex_NOR_Match", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			classifierRegex.Classify(testTextNorMatch)
		}
	})
	b.Run("Regex_No_Match", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			classifierRegex.Classify(testTextNoMatch)
		}
	})
}
