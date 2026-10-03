package lexical

import (
	"fmt"
	"strings"
	"testing"
)

// The scenarios of the binding comparison in the stores performance record:
// thirty single-keyword rules that never match a long prompt (nothing
// short-circuits), so every rule scores the whole text.
func benchmarkRules(b *testing.B, ngram bool, text string) {
	matchers := make([]func(*Text) (Match, bool), 30)
	for i := range matchers {
		rule := Rule{Name: fmt.Sprintf("rule_%02d", i), Operator: "OR", Keywords: []string{fmt.Sprintf("keyword%02d", i)}, Threshold: 0.1}
		if ngram {
			rule.Threshold = 0.4
			m, err := NewNgram(rule, 3)
			if err != nil {
				b.Fatal(err)
			}
			matchers[i] = m.Match
		} else {
			m, err := NewBM25(rule)
			if err != nil {
				b.Fatal(err)
			}
			matchers[i] = m.Match
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		analysis := NewText(text)
		for _, match := range matchers {
			_, _ = match(analysis)
		}
	}
}

var longEnglish = strings.Repeat("Please help me debug this function because the deployment is urgent and the service keeps failing. ", 20)

func BenchmarkBM25ThirtyRulesLongEnglish(b *testing.B)  { benchmarkRules(b, false, longEnglish) }
func BenchmarkNgramThirtyRulesLongEnglish(b *testing.B) { benchmarkRules(b, true, longEnglish) }
