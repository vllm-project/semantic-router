package candle_binding

import (
	"fmt"
	"testing"
)

func BenchmarkRegexProvider_Scan(b *testing.B) {
	cfg := RegexProviderConfig{
		MaxPatterns:      100,
		MaxPatternLength: 1000,
		MaxInputLength:   10000,
		DefaultTimeoutMs: 1000,
		Patterns: []RegexPattern{
			{ID: "email", Pattern: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`},
			{ID: "word", Pattern: "hello"},
			{ID: "case", Pattern: "World", Flags: "i"},
		},
	}
	rp, err := NewRegexProvider(cfg)
	if err != nil {
		b.Fatalf("failed to create regex provider: %v", err)
	}

	input := "my email is test@example.com, say hello to the beautiful World"

	b.Run("SinglePattern", func(b *testing.B) {
		singlePatternCfg := RegexProviderConfig{
			MaxPatterns:      1,
			MaxPatternLength: 100,
			MaxInputLength:   1000,
			DefaultTimeoutMs: 100,
			Patterns: []RegexPattern{
				{ID: "email", Pattern: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`},
			},
		}
		singleRp, _ := NewRegexProvider(singlePatternCfg)
		for i := 0; i < b.N; i++ {
			_, _ = singleRp.Scan(input)
		}
	})

	b.Run("MultiPattern", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = rp.Scan(input)
		}
	})

	b.Run("LargeInput", func(b *testing.B) {
		largeInput := ""
		for i := 0; i < 100; i++ {
			largeInput += fmt.Sprintf("email%d@example.com ", i)
		}
		for i := 0; i < b.N; i++ {
			_, _ = rp.Scan(largeInput)
		}
	})
}
