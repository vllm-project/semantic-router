package classification

import (
	"unicode"
	"unicode/utf8"
)

// keywordBoundaryMatch reports whether the text between start and end is a
// word of its own: no letter or digit of a non-CJK script touches it. CJK
// text has no spaces between words, so a CJK neighbor never joins a word.
func keywordBoundaryMatch(text string, start int, end int) bool {
	if start > 0 {
		if prev, _ := utf8.DecodeLastRuneInString(text[:start]); isBoundaryBlockingRune(prev) {
			return false
		}
	}
	if end < len(text) {
		if next, _ := utf8.DecodeRuneInString(text[end:]); isBoundaryBlockingRune(next) {
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
