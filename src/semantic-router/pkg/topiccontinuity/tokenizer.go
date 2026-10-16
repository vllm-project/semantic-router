package topiccontinuity

import (
	"strings"
	"unicode"
)

// tokenizeTerms is a frozen copy of contextcompression.tokenizeTerms as of
// upstream commit ed174453c. It is part of this evaluator's versioned
// behavior: a later change to the context-compression tokenizer must not
// change topic-continuity results, so this copy is pinned by package-owned
// goldens and changes only together with EvaluatorVersion.
//
// Apostrophes are separators here, so "'parser'" and "parser" yield the same
// term without any extra trimming.
func tokenizeTerms(text string) []string {
	lower := []rune(strings.ToLower(text))
	terms := make([]string, 0, len(lower))
	word := make([]rune, 0)
	flushWord := func() {
		if len(word) > 1 {
			terms = append(terms, string(word))
		}
		word = word[:0]
	}
	var previousCJK rune
	for _, value := range lower {
		switch {
		case isCJK(value):
			flushWord()
			terms = append(terms, string(value))
			if previousCJK != 0 {
				terms = append(terms, string([]rune{previousCJK, value}))
			}
			previousCJK = value
		case unicode.IsLetter(value) || unicode.IsDigit(value) || value == '_':
			previousCJK = 0
			word = append(word, value)
		case unicode.IsSymbol(value):
			flushWord()
			previousCJK = 0
			terms = append(terms, string(value))
		default:
			flushWord()
			previousCJK = 0
		}
	}
	flushWord()
	return terms
}

func isCJK(value rune) bool {
	// Every Han, Hiragana, Katakana, and Hangul code point is at or above
	// U+1100, so this fast path is exactly equivalent to the table lookup.
	if value < 0x1100 {
		return false
	}
	return unicode.In(value, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul)
}

// termSet tokenizes every segment separately and drops stopwords.
func termSet(segments []textSegment) map[string]struct{} {
	out := make(map[string]struct{})
	for _, segment := range segments {
		for _, term := range tokenizeTerms(string(segment)) {
			if _, stop := stopwords[term]; stop {
				continue
			}
			out[term] = struct{}{}
		}
	}
	return out
}
