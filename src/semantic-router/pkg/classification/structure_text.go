package classification

import (
	"strings"
	"unicode"
)

// multilingualTextUnitCount counts content-bearing units without relying on
// whitespace tokenization. CJK runes count individually, while contiguous runs
// of non-CJK letters/digits count as one unit.
func multilingualTextUnitCount(text string) int {
	count := 0
	inWord := false

	for _, r := range text {
		switch {
		case isCJK(r):
			count++
			inWord = false
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			if !inWord {
				count++
				inWord = true
			}
		default:
			inWord = false
		}
	}

	return count
}

func keywordOccurrenceCount(text string, keywords []string, caseSensitive bool) int {
	if len(keywords) == 0 {
		return 0
	}
	candidate := text
	if !caseSensitive {
		candidate = strings.ToLower(text)
	}
	total := 0
	for _, keyword := range keywords {
		currentKeyword := keyword
		if !caseSensitive {
			currentKeyword = strings.ToLower(keyword)
		}
		total += countKeywordOccurrences(candidate, currentKeyword)
	}
	return total
}

func countKeywordOccurrences(text string, keyword string) int {
	if keyword == "" {
		return 0
	}

	requireBoundary := !containsCJK(keyword)
	count := 0
	searchFrom := 0

	for {
		idx := strings.Index(text[searchFrom:], keyword)
		if idx < 0 {
			return count
		}
		absoluteIdx := searchFrom + idx
		if !requireBoundary || keywordBoundaryMatch(text, absoluteIdx, absoluteIdx+len(keyword)) {
			count++
		}
		searchFrom = absoluteIdx + len(keyword)
	}
}

func keywordBoundaryMatch(text string, start int, end int) bool {
	if start > 0 {
		if prev, _ := utf8DecodeLastRuneInString(text[:start]); isBoundaryBlockingRune(prev) {
			return false
		}
	}
	if end < len(text) {
		if next, _ := utf8DecodeRuneInString(text[end:]); isBoundaryBlockingRune(next) {
			return false
		}
	}
	return true
}

func isBoundaryBlockingRune(r rune) bool {
	return (unicode.IsLetter(r) || unicode.IsDigit(r)) && !isCJK(r)
}

func containsCJK(text string) bool {
	for _, r := range text {
		if isCJK(r) {
			return true
		}
	}
	return false
}

func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

func utf8DecodeRuneInString(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return rune(0), 0
}

func utf8DecodeLastRuneInString(s string) (rune, int) {
	last := rune(0)
	size := 0
	for _, r := range s {
		last = r
		size = len(string(r))
	}
	return last, size
}
