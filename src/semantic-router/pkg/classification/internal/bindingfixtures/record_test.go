//go:build record_binding_fixtures

// Package bindingfixtures records the BM25 and n-gram keyword fixtures from
// nlp-binding before the binding is removed. It is temporary: the fixtures it
// writes are the parity reference for the pure-Go classifiers.
//
//	go test -tags record_binding_fixtures ./pkg/classification/internal/bindingfixtures \
//	  -run TestRecordKeywordFixtures -args -out keyword_binding_fixtures.json
package bindingfixtures

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
	"unicode/utf8"

	nlp_binding "github.com/vllm-project/semantic-router/nlp-binding"
)

var out = flag.String("out", "", "fixture output file")

type rule struct {
	Name          string   `json:"name"`
	Operator      string   `json:"operator"`
	Keywords      []string `json:"keywords"`
	Threshold     float32  `json:"threshold"`
	CaseSensitive bool     `json:"case_sensitive"`
	Arity         int      `json:"arity,omitempty"`
}

type match struct {
	Rule          string    `json:"rule"`
	Keywords      []string  `json:"keywords"`
	Scores        []float32 `json:"scores"`
	MatchCount    int       `json:"match_count"`
	TotalKeywords int       `json:"total_keywords"`
}

type textResult struct {
	Text  string  `json:"text"`
	First *match  `json:"first"`
	All   []match `json:"all"`
	// Stable is false when repeated runs over fresh classifiers disagreed
	// (the binding iterates hash sets with per-instance random state).
	Stable bool `json:"stable"`
}

type fixtureCase struct {
	Name   string       `json:"name"`
	Method string       `json:"method"`
	Rules  []rule       `json:"rules"`
	Texts  []textResult `json:"texts"`
}

type fixtureFile struct {
	Binding string        `json:"binding"`
	Note    string        `json:"note"`
	Cases   []fixtureCase `json:"cases"`
}

func TestRecordKeywordFixtures(t *testing.T) {
	if *out == "" {
		t.Skip("-out is required")
	}
	texts := corpus()
	file := fixtureFile{
		Binding: "nlp-binding (bm25 2.3.2, ngrammatic 0.7.0)",
		Note:    "Matched keywords keep the binding's order; ties in score may appear in any order.",
	}
	for _, c := range ruleSets() {
		recorded := fixtureCase{Name: c.name, Method: c.method, Rules: c.rules}
		for _, text := range texts {
			recorded.Texts = append(recorded.Texts, classify(t, c.method, c.rules, text))
		}
		file.Cases = append(file.Cases, recorded)
	}
	data, err := json.MarshalIndent(file, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(*out, append(data, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, c := range file.Cases {
		total += len(c.Texts)
	}
	t.Logf("recorded %d cases, %d texts", len(file.Cases), total)
}

func classify(t *testing.T, method string, rules []rule, text string) textResult {
	const runs = 3
	var first textResult
	for run := 0; run < runs; run++ {
		var result textResult
		switch method {
		case "bm25":
			c := nlp_binding.NewBM25Classifier()
			for _, r := range rules {
				if err := c.AddRule(r.Name, r.Operator, r.Keywords, r.Threshold, r.CaseSensitive); err != nil {
					t.Fatal(err)
				}
			}
			result = convert(text, c.Classify(text), c.ClassifyAll(text))
			c.Free()
		case "ngram":
			c := nlp_binding.NewNgramClassifier()
			for _, r := range rules {
				if err := c.AddRule(r.Name, r.Operator, r.Keywords, r.Threshold, r.CaseSensitive, r.Arity); err != nil {
					t.Fatal(err)
				}
			}
			result = convert(text, c.Classify(text), c.ClassifyAll(text))
			c.Free()
		}
		if run == 0 {
			first = result
			first.Stable = true
			continue
		}
		if canonical(result) != canonical(first) {
			first.Stable = false
		}
	}
	return first
}

func convert(text string, first nlp_binding.MatchResult, all []nlp_binding.MatchResult) textResult {
	result := textResult{Text: text, All: []match{}}
	if first.Matched {
		m := toMatch(first)
		result.First = &m
	}
	for _, r := range all {
		if r.Matched {
			result.All = append(result.All, toMatch(r))
		}
	}
	return result
}

func toMatch(r nlp_binding.MatchResult) match {
	keywords := append([]string{}, r.MatchedKeywords...)
	scores := append([]float32{}, r.Scores...)
	return match{Rule: r.RuleName, Keywords: keywords, Scores: scores, MatchCount: r.MatchCount, TotalKeywords: r.TotalKeywords}
}

// canonical sorts keyword/score pairs so runs that differ only in tie order compare equal.
func canonical(result textResult) string {
	all := append([]match{}, result.All...)
	if result.First != nil {
		all = append(all, *result.First)
	}
	var parts []string
	for _, m := range all {
		pairs := make([]string, len(m.Keywords))
		for i := range m.Keywords {
			var bits uint32
			if i < len(m.Scores) {
				bits = math.Float32bits(m.Scores[i])
			}
			pairs[i] = fmt.Sprintf("%s=%08x", m.Keywords[i], bits)
		}
		sort.Strings(pairs)
		parts = append(parts, fmt.Sprintf("%s|%d|%d|%s", m.Rule, m.MatchCount, m.TotalKeywords, strings.Join(pairs, ",")))
	}
	return strings.Join(parts, ";")
}

type ruleSet struct {
	name   string
	method string
	rules  []rule
}

func ruleSets() []ruleSet {
	code := []string{"code", "function", "implement", "debug", "algorithm", "compile", "syntax", "variable"}
	medical := []string{"diagnosis", "treatment", "symptoms", "prescription", "surgery", "patient", "medication"}
	urgent := []string{"urgent", "immediate", "asap", "emergency"}
	phrases := []string{"machine learning", "neural network", "deep learning model", "gradient descent", "how to"}
	mixed := []string{"The", "running", "connections", "can't", "e-mail", "foo_bar", "3.14", "U.S.A.", "naïve", "café", "Straße", "привет", "北京", "🍕", "x"}
	dupes := []string{"code", "Code", "code", "debug", "debugging", "debugged"}
	stop := []string{"the", "and", "is", "of", "you're", "how"}

	var sets []ruleSet
	for _, threshold := range []float32{0.1, 0.5, 1.0, 2.5} {
		sets = append(sets, ruleSet{fmt.Sprintf("bm25-code-medical-t%g", threshold), "bm25", []rule{
			{Name: "code_keywords", Operator: "OR", Keywords: code, Threshold: threshold},
			{Name: "medical_keywords", Operator: "OR", Keywords: medical, Threshold: threshold},
		}})
	}
	sets = append(sets,
		ruleSet{"bm25-operators", "bm25", []rule{
			{Name: "all_code", Operator: "AND", Keywords: []string{"debug", "function"}, Threshold: 0.1},
			{Name: "no_medical", Operator: "NOR", Keywords: medical, Threshold: 0.1},
			{Name: "urgent", Operator: "OR", Keywords: urgent, Threshold: 0.1},
		}},
		ruleSet{"bm25-phrases", "bm25", []rule{{Name: "ml", Operator: "OR", Keywords: phrases, Threshold: 0.1}}},
		ruleSet{"bm25-mixed-unicode", "bm25", []rule{
			{Name: "mixed", Operator: "OR", Keywords: mixed, Threshold: 0.1},
			{Name: "mixed_cs", Operator: "OR", Keywords: mixed, Threshold: 0.3, CaseSensitive: true},
		}},
		ruleSet{"bm25-duplicates-stopwords", "bm25", []rule{
			{Name: "dupes", Operator: "OR", Keywords: dupes, Threshold: 0.1},
			{Name: "stop", Operator: "OR", Keywords: stop, Threshold: 0.1},
			{Name: "stop_and", Operator: "AND", Keywords: stop, Threshold: 0.1},
		}},
		ruleSet{"bm25-single", "bm25", []rule{{Name: "one", Operator: "OR", Keywords: []string{"invoice"}, Threshold: 0.01}}},
	)
	for _, arity := range []int{2, 3, 4, 5} {
		for _, threshold := range []float32{0.3, 0.4, 0.6, 0.85} {
			sets = append(sets, ruleSet{fmt.Sprintf("ngram-a%d-t%g", arity, threshold), "ngram", []rule{
				{Name: "urgent_keywords", Operator: "OR", Keywords: urgent, Threshold: threshold, Arity: arity},
				{Name: "code_keywords", Operator: "OR", Keywords: code, Threshold: threshold, Arity: arity},
			}})
		}
	}
	sets = append(sets,
		ruleSet{"ngram-operators", "ngram", []rule{
			{Name: "all_urgent", Operator: "AND", Keywords: []string{"urgent", "asap"}, Threshold: 0.4, Arity: 3},
			{Name: "no_code", Operator: "NOR", Keywords: code, Threshold: 0.4, Arity: 3},
			{Name: "medical", Operator: "OR", Keywords: medical, Threshold: 0.4, Arity: 3},
		}},
		ruleSet{"ngram-phrases", "ngram", []rule{{Name: "ml", Operator: "OR", Keywords: phrases, Threshold: 0.3, Arity: 3}}},
		ruleSet{"ngram-mixed-unicode", "ngram", []rule{
			{Name: "mixed", Operator: "OR", Keywords: mixed, Threshold: 0.4, Arity: 3},
			{Name: "mixed_cs", Operator: "OR", Keywords: mixed, Threshold: 0.4, Arity: 2, CaseSensitive: true},
		}},
		ruleSet{"ngram-duplicates-many", "ngram", []rule{
			{Name: "dupes", Operator: "OR", Keywords: dupes, Threshold: 0.3, Arity: 3},
			{Name: "many", Operator: "OR", Keywords: append(append(append([]string{}, code...), medical...), urgent...), Threshold: 0.2, Arity: 2},
		}},
	)
	return sets
}

func corpus() []string {
	texts := []string{
		"", " ", "a", "x", "the", "the and is of", "?!", "...",
		"Help me debug this function",
		"I need to implement an algorithm",
		"Fix the syntax error in my code",
		"What are the symptoms of flu?",
		"The patient needs surgery",
		"Discuss treatment options for diagnosis",
		"What is the weather like today?",
		"How do I cook pasta?",
		"Help me code a medical app",
		"This is URGENT, please respond ASAP!",
		"emergancy: the server is down, need immediat help",
		"urgnet request for teh debuging of my fucntion",
		"Can you implment a sorting algorithim and compiel it?",
		"I'm running several connections; they're connected and connecting.",
		"Explain machine learning and neural networks, plus gradient descent.",
		"How to train a deep learning model on GPUs",
		"can't won't don't it's users' you're couldn't've",
		"e-mail foo_bar a:b 3.14 1,000,000 U.S.A. e.g. x_1 hello:world v2.0.1",
		"Code CODE code CoDe codes coding coded",
		"DEBUG Debugging debugged debugger debugs",
		"naïve café résumé façade jalapeño Straße Æneid œuvre Øresund Łódź",
		"привет мир, как дела? Москва",
		"北京欢迎你，我想学习编程。",
		"げんまい茶とカタカナのテスト",
		"مرحبا بالعالم",
		"नमस्ते दुनिया, कोड डिबग करें",
		"Γειά σου Κόσμε, ΟΔΟΣ",
		"I love 🍕 and 🚀 and 🍋 🔥",
		"“Smart quotes” — dashes – and … ellipsis",
		"\tspace\r\nstation\n space       station",
		"MiXeD CaSe ImPlEmEnT FuNcTiOn",
		"İstanbul ΣΊΣΥΦΟΣ final sigma: ΟΔΟΣ ΔΡΌΜΟΣ",
		"invoice invoices invoicing INVOICE",
		"surgery surgery surgery patient",
		"code-function debug_algorithm compile.syntax variable:code",
		"0123456789 42 1337",
		"the patient's diagnosis was surgery-free; medication & treatment",
	}
	long := strings.Repeat("Please help me debug this function because the deployment is urgent. ", 40)
	texts = append(texts, long)

	words := []string{
		"code", "function", "implement", "debug", "algorithm", "compile", "syntax", "variable",
		"diagnosis", "treatment", "symptoms", "prescription", "surgery", "patient", "medication",
		"urgent", "immediate", "asap", "emergency", "machine", "learning", "neural", "network",
		"weather", "pasta", "travel", "music", "the", "a", "of", "please", "help", "me", "with",
		"invoice", "café", "naïve", "привет", "北京", "🍕",
	}
	rng := rand.New(rand.NewSource(20261004))
	for i := 0; i < 220; i++ {
		n := 1 + rng.Intn(12)
		parts := make([]string, n)
		for j := range parts {
			word := words[rng.Intn(len(words))]
			switch rng.Intn(6) {
			case 0:
				word = typo(rng, word)
			case 1:
				word = strings.ToUpper(word)
			case 2:
				if r, size := utf8.DecodeRuneInString(word); size > 0 {
					word = strings.ToUpper(string(r)) + word[size:]
				}
			}
			parts[j] = word
		}
		separators := []string{" ", " ", " ", ", ", ". ", "-", "_", "! ", "? ", "\n"}
		var builder strings.Builder
		for j, part := range parts {
			if j > 0 {
				builder.WriteString(separators[rng.Intn(len(separators))])
			}
			builder.WriteString(part)
		}
		texts = append(texts, builder.String())
	}
	return texts
}

func typo(rng *rand.Rand, word string) string {
	runes := []rune(word)
	if len(runes) < 2 || !utf8.ValidString(word) {
		return word
	}
	i := rng.Intn(len(runes) - 1)
	switch rng.Intn(4) {
	case 0: // swap
		runes[i], runes[i+1] = runes[i+1], runes[i]
	case 1: // delete
		runes = append(runes[:i], runes[i+1:]...)
	case 2: // insert
		runes = append(runes[:i], append([]rune{rune('a' + rng.Intn(26))}, runes[i:]...)...)
	default: // substitute
		runes[i] = rune('a' + rng.Intn(26))
	}
	return string(runes)
}
