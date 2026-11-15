package classification

import (
	"unicode"
	"unicode/utf8"
)

// keywordBoundaryMatch reports whether the text between start and end is a
// word of its own: no letter or digit of a spaced script touches it. Scripts
// written without spaces between words never join a word, so their letters do
// not block a neighbor.
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
	return unicode.IsDigit(r) || (unicode.IsLetter(r) && !isUnspacedScript(r))
}

// Khmer coeng and Myanmar virama are stored before the consonant they stack
// onto, so a keyword starting right after one would split that cluster.
const (
	khmerCoeng    = '\u17D2'
	myanmarVirama = '\u1039'
)

// keywordSyllableMatch reports whether a match in an unspaced syllabic script
// stands as a whole syllable rather than splitting one: it must not end before
// a combining mark of the same cluster, nor start after a preposed vowel or a
// stacking sign that belongs to the next consonant.
func keywordSyllableMatch(text string, start, end int) bool {
	if end < len(text) {
		if next, _ := utf8.DecodeRuneInString(text[end:]); unicode.In(next, unicode.Mn, unicode.Mc) {
			return false
		}
	}
	if start > 0 {
		prev, _ := utf8.DecodeLastRuneInString(text[:start])
		if unicode.Is(unicode.Logical_Order_Exception, prev) || prev == khmerCoeng || prev == myanmarVirama {
			return false
		}
	}
	return true
}

func containsUnspacedScript(text string) bool {
	for _, r := range text {
		if isUnspacedScript(r) {
			return true
		}
	}
	return false
}

func containsSyllabicScript(text string) bool {
	for _, r := range text {
		if isSyllabicScript(r) {
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

// isUnspacedScript reports runes of scripts written without spaces between
// words. isCJK stays separate because it also drives the signal-window budget.
func isUnspacedScript(r rune) bool {
	return isCJK(r) || isSyllabicScript(r)
}

// isSyllabicScript reports letters of the unspaced syllabic scripts. Their
// digits are excluded so a digit or identifier keyword keeps word boundaries.
func isSyllabicScript(r rune) bool {
	return unicode.IsLetter(r) && unicode.In(r, unicode.Thai, unicode.Lao, unicode.Khmer, unicode.Myanmar)
}
