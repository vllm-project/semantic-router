//go:build record_binding_fixtures

package bindingfixtures

import (
	"fmt"
	"strings"
	"testing"

	nlp_binding "github.com/vllm-project/semantic-router/nlp-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification/lexical"
)

// Side-by-side latency of nlp-binding and the pure-Go rules on the same rules
// and texts, as the keyword signal calls them (first match and all matches).

type scenario struct {
	name  string
	ngram bool
	rules []rule
	text  string
}

func scenarios() []scenario {
	pool := []string{"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
		"kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
		"uniform", "victor", "whiskey", "xray", "yankee", "zulu", "anode", "binary", "carbon", "dynamo"}
	many := func(n int, ngram bool) []rule {
		rules := make([]rule, n)
		for i := range rules {
			rules[i] = rule{Name: fmt.Sprintf("rule_%02d", i), Operator: "OR", Keywords: []string{pool[i%len(pool)]}, Threshold: 0.1, Arity: 3}
			if ngram {
				rules[i].Threshold = 0.4
			}
		}
		return rules
	}
	chinese := strings.Repeat("我們的 RAG 系統在過去兩週召回率從 0.82 掉到 0.61，但 embedding 模型沒有換、索引也沒有重建。請以系統化的方式描述：要怎麼確認問題、怎麼量化影響。\n\n", 15)
	english := strings.Repeat("Please help me debug this function because the deployment is urgent and the service keeps failing. ", 20)
	codeMedical := []rule{
		{Name: "code_keywords", Operator: "OR", Keywords: []string{"code", "function", "implement", "debug", "algorithm", "compile", "syntax", "variable"}, Threshold: 0.1, Arity: 3},
		{Name: "medical_keywords", Operator: "OR", Keywords: []string{"diagnosis", "treatment", "symptoms", "prescription", "surgery", "patient", "medication"}, Threshold: 0.1, Arity: 3},
	}
	urgent := []rule{{Name: "urgent", Operator: "OR", Keywords: []string{"urgent", "immediate", "asap", "emergency"}, Threshold: 0.4, Arity: 3}}
	return []scenario{
		{"bm25-30rules-long-chinese", false, many(30, false), chinese},
		{"bm25-5rules-long-chinese", false, many(5, false), chinese},
		{"bm25-30rules-short-english", false, many(30, false), "What is the time complexity of binary search?"},
		{"bm25-code-medical-prompt", false, codeMedical, "Can you help me debug this function and explain the algorithm's time complexity?"},
		{"bm25-30rules-long-english", false, many(30, false), english},
		{"ngram-urgent-typo", true, urgent, "emergancy: the server is down, need immediat help asap"},
		{"ngram-code-medical-prompt", true, codeMedical, "Can you help me debug this function and explain the algorithm's time complexity?"},
		{"ngram-30rules-long-english", true, many(30, true), english},
	}
}

func BenchmarkKeywordRules(b *testing.B) {
	for _, sc := range scenarios() {
		b.Run(sc.name+"/binding", func(b *testing.B) {
			var classify func(string) nlp_binding.MatchResult
			var classifyAll func(string) []nlp_binding.MatchResult
			if sc.ngram {
				c := nlp_binding.NewNgramClassifier()
				defer c.Free()
				for _, r := range sc.rules {
					if err := c.AddRule(r.Name, r.Operator, r.Keywords, r.Threshold, r.CaseSensitive, r.Arity); err != nil {
						b.Fatal(err)
					}
				}
				classify, classifyAll = c.Classify, c.ClassifyAll
			} else {
				c := nlp_binding.NewBM25Classifier()
				defer c.Free()
				for _, r := range sc.rules {
					if err := c.AddRule(r.Name, r.Operator, r.Keywords, r.Threshold, r.CaseSensitive); err != nil {
						b.Fatal(err)
					}
				}
				classify, classifyAll = c.Classify, c.ClassifyAll
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = classify(sc.text)
				_ = classifyAll(sc.text)
			}
		})
		b.Run(sc.name+"/go", func(b *testing.B) {
			matchers := make([]func(*lexical.Text) (lexical.Match, bool), len(sc.rules))
			for i, r := range sc.rules {
				rule := lexical.Rule{Name: r.Name, Operator: r.Operator, Keywords: r.Keywords, Threshold: r.Threshold, CaseSensitive: r.CaseSensitive}
				if sc.ngram {
					m, err := lexical.NewNgram(rule, r.Arity)
					if err != nil {
						b.Fatal(err)
					}
					matchers[i] = m.Match
				} else {
					m, err := lexical.NewBM25(rule)
					if err != nil {
						b.Fatal(err)
					}
					matchers[i] = m.Match
				}
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// First match, then all matches, each with its own analysis as the
				// two binding calls tokenize separately.
				text := lexical.NewText(sc.text)
				for _, match := range matchers {
					if _, ok := match(text); ok {
						break
					}
				}
				text = lexical.NewText(sc.text)
				for _, match := range matchers {
					_, _ = match(text)
				}
			}
		})
	}
}
