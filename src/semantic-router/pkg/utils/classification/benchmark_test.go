package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func BenchmarkKeywordClassifier(b *testing.B) {
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

	classifier := NewKeywordClassifier(rules)

	b.Run("AND match", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _, _ = classifier.Classify("this text contains keyword1 and keyword2")
		}
	})

	b.Run("OR match", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _, _ = classifier.Classify("this text contains keyword3")
		}
	})

	b.Run("NOR match", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _, _ = classifier.Classify("this text is clean")
		}
	})

	b.Run("No match", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _, _ = classifier.Classify("this text contains keyword5")
		}
	})
}
