// Package lexical implements the BM25 and n-gram keyword rules in pure Go.
//
// Both reproduce the rules nlp-binding served (the bm25 2.3.2 and ngrammatic
// 0.7.0 crates), so configured thresholds keep their meaning: the same
// tokenizer, scores computed with the same float32 operations, and the same
// AND / OR / NOR semantics. Unlike the binding, keywords with equal scores
// are reported in declaration order instead of hash order. Fixtures recorded
// from the binding check the equivalence.
package lexical

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Rule is one keyword rule.
type Rule struct {
	Name          string
	Operator      string // AND, OR or NOR
	Keywords      []string
	Threshold     float32
	CaseSensitive bool
}

// Match describes a matching rule. A NOR rule matches with no keywords.
type Match struct {
	Keywords      []string
	Scores        []float32
	MatchCount    int
	TotalKeywords int
}

func (r Rule) validate() error {
	switch r.Operator {
	case "AND", "OR", "NOR":
	default:
		return fmt.Errorf("keyword rule %q: unsupported operator %q", r.Name, r.Operator)
	}
	if len(r.Keywords) == 0 {
		return fmt.Errorf("keyword rule %q has no keywords", r.Name)
	}
	return nil
}

// decide applies the rule operator to the matched keywords.
func (r Rule) decide(keywords []string, scores []float32) (Match, bool) {
	matched := false
	switch r.Operator {
	case "OR":
		matched = len(keywords) > 0
	case "AND":
		matched = len(keywords) == len(r.Keywords)
	case "NOR":
		return Match{TotalKeywords: len(r.Keywords)}, len(keywords) == 0
	}
	if !matched {
		return Match{}, false
	}
	return Match{Keywords: keywords, Scores: scores, MatchCount: len(keywords), TotalKeywords: len(r.Keywords)}, true
}

// normalizedKeyword is the keyword as the rule compares it.
func (r Rule) normalizedKeyword(keyword string) string {
	if r.CaseSensitive {
		return keyword
	}
	return toLowerRust(keyword)
}

// Text holds the analyses of one input that several rules share, so a
// request tokenizes its text once per variant however many rules it checks.
// It is not safe for concurrent use.
type Text struct {
	raw   string
	lower *string
	// [0] case-insensitive, [1] case-sensitive
	tokens [2]*[]string
	words  [2]*[]string
	grams  map[gramsKey]paddedGrams
}

type gramsKey struct {
	arity int
	text  string
}

// paddedGrams are the distinct n-grams of a space-padded text and the
// padded text's length in characters.
type paddedGrams struct {
	grams  []gramCount
	length int
}

// NewText prepares text for matching.
func NewText(text string) *Text { return &Text{raw: text} }

func variant(caseSensitive bool) int {
	if caseSensitive {
		return 1
	}
	return 0
}

// input is the text as a rule reads it: lowercased unless case-sensitive.
func (t *Text) input(caseSensitive bool) string {
	if caseSensitive {
		return t.raw
	}
	if t.lower == nil {
		lower := toLowerRust(t.raw)
		t.lower = &lower
	}
	return *t.lower
}

func (t *Text) bm25Tokens(caseSensitive bool) []string {
	slot := &t.tokens[variant(caseSensitive)]
	if *slot == nil {
		tokens := tokenize(t.input(caseSensitive))
		*slot = &tokens
	}
	return **slot
}

// ngrams returns the n-grams of text padded for arity, computed once per text
// and arity however many rules search it.
func (t *Text) ngrams(text string, arity int) paddedGrams {
	key := gramsKey{arity: arity, text: text}
	if cached, ok := t.grams[key]; ok {
		return cached
	}
	if t.grams == nil {
		t.grams = make(map[gramsKey]paddedGrams)
	}
	pad := strings.Repeat(" ", arity-1)
	padded := pad + text + pad
	computed := paddedGrams{grams: countGrams(padded, arity), length: utf8.RuneCountInString(padded)}
	t.grams[key] = computed
	return computed
}

func (t *Text) ngramWords(caseSensitive bool) []string {
	slot := &t.words[variant(caseSensitive)]
	if *slot == nil {
		words := strings.FieldsFunc(t.input(caseSensitive), func(r rune) bool {
			return !isAlphanumericRust(r) && r != '_' && r != '-'
		})
		*slot = &words
	}
	return **slot
}
