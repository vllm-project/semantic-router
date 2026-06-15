package classification

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// makeBM25Rules builds n distinct BM25 rules whose keywords do NOT appear in
// the test prompt, so first-match short-circuit inside Rust is never taken.
// This is the realistic shape of the production hot path: many signals, few
// (or zero) matches per request.
func makeBM25Rules(n int) []config.KeywordRule {
	pool := []string{
		"alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
		"golf", "hotel", "india", "juliet", "kilo", "lima",
		"mike", "november", "oscar", "papa", "quebec", "romeo",
		"sierra", "tango", "uniform", "victor", "whiskey", "xray",
		"yankee", "zulu", "anode", "binary", "carbon", "dynamo",
	}
	rules := make([]config.KeywordRule, n)
	for i := 0; i < n; i++ {
		rules[i] = config.KeywordRule{
			Name:          fmt.Sprintf("rule_%02d", i),
			Operator:      "OR",
			Method:        "bm25",
			Keywords:      []string{pool[i%len(pool)]},
			BM25Threshold: 0.1,
		}
	}
	return rules
}

// TestBM25_NoCallAmplificationOnMixedRules verifies that with N BM25 rules
// configured, calling Classify still returns the correct match. This is a
// behaviour regression guard for the cache fix: the cache must not change
// observable results.
func TestBM25_CorrectnessAcrossManyRules(t *testing.T) {
	rules := makeBM25Rules(30)
	// Inject a rule that *does* match, in the middle of the list.
	rules[20] = config.KeywordRule{
		Name:          "matching_rule",
		Operator:      "OR",
		Method:        "bm25",
		Keywords:      []string{"observability", "telemetry"},
		BM25Threshold: 0.1,
	}

	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	defer kc.Free()

	got, _, err := kc.ClassifyWithKeywords("we need better observability for this service")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if got != "matching_rule" {
		t.Fatalf("expected matching_rule, got %q", got)
	}
}

// TestBM25_FirstMatchPriorityPreserved ensures that when multiple BM25 rules
// could match, the one declared first in config wins (matches existing
// first-match-by-config-order semantics).
func TestBM25_FirstMatchPriorityPreserved(t *testing.T) {
	rules := []config.KeywordRule{
		{
			Name: "a_first", Operator: "OR", Method: "bm25",
			Keywords: []string{"observability"}, BM25Threshold: 0.1,
		},
		{
			Name: "b_second", Operator: "OR", Method: "bm25",
			Keywords: []string{"observability"}, BM25Threshold: 0.1,
		},
	}
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	defer kc.Free()

	got, _, err := kc.ClassifyWithKeywords("observability matters")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if got != "a_first" {
		t.Fatalf("expected a_first to win by config order, got %q", got)
	}
}

// TestMixedMethodOrderPreserved ensures that interleaving regex/bm25/ngram
// rules still respects ruleOrder for first-match semantics, with the cache.
func TestMixedMethodOrderPreserved(t *testing.T) {
	rules := []config.KeywordRule{
		{
			Name: "bm25_a", Operator: "OR", Method: "bm25",
			Keywords: []string{"alpha"}, BM25Threshold: 0.1,
		},
		{
			Name: "regex_b", Operator: "OR", Method: "regex",
			Keywords: []string{"observability"},
		},
		{
			Name: "bm25_c", Operator: "OR", Method: "bm25",
			Keywords: []string{"telemetry"}, BM25Threshold: 0.1,
		},
	}
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		t.Fatalf("NewKeywordClassifier: %v", err)
	}
	defer kc.Free()

	// Prompt matches regex_b only — should return regex_b, not bm25_a or bm25_c.
	got, _, err := kc.ClassifyWithKeywords("observability is key")
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if got != "regex_b" {
		t.Fatalf("expected regex_b, got %q", got)
	}
}

// longChinesePrompt builds a ~6.7KB Traditional-Chinese prompt similar to the
// production "L-ZH" benchmark cell.
func longChinesePrompt() string {
	chunk := "我們的內部知識庫搜尋最近開始出現相關性下降的問題：top-3 命中率從 0.78 " +
		"掉到 0.61，但 embedding 模型沒有換、索引也沒有重建。同事懷疑是新加入的 " +
		"中英文混用文件影響了向量分布。請以系統化的方式描述：要怎麼確認問題、" +
		"怎麼量化影響、以及在不重訓模型的前提下有哪些補救策略？請按照診斷步驟、" +
		"量化方法、緩解方案三段回答。\n\n"
	return strings.Repeat(chunk, 15)
}

// BenchmarkBM25_LongChinese_30Rules measures the realistic worst case from the
// signal-latency report: N=30 BM25 rules, long Chinese prompt, no rule matches
// (so Rust never short-circuits). On unfixed upstream this took ~54ms per
// classify in production metrics. With the per-call cache it should drop into
// the low-millisecond range.
func BenchmarkBM25_LongChinese_30Rules(b *testing.B) {
	rules := makeBM25Rules(30)
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		b.Fatalf("NewKeywordClassifier: %v", err)
	}
	defer kc.Free()
	prompt := longChinesePrompt()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = kc.ClassifyWithKeywords(prompt)
	}
}

// BenchmarkBM25_LongChinese_5Rules is the lower-N reference point.
func BenchmarkBM25_LongChinese_5Rules(b *testing.B) {
	rules := makeBM25Rules(5)
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		b.Fatalf("NewKeywordClassifier: %v", err)
	}
	defer kc.Free()
	prompt := longChinesePrompt()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = kc.ClassifyWithKeywords(prompt)
	}
}

// BenchmarkBM25_ShortEnglish_30Rules confirms short prompts stay fast (the
// case where the N^2 bug was previously invisible).
func BenchmarkBM25_ShortEnglish_30Rules(b *testing.B) {
	rules := makeBM25Rules(30)
	kc, err := NewKeywordClassifier(rules)
	if err != nil {
		b.Fatalf("NewKeywordClassifier: %v", err)
	}
	defer kc.Free()
	prompt := "What is the time complexity of binary search?"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = kc.ClassifyWithKeywords(prompt)
	}
}
