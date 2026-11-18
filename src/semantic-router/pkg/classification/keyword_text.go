package classification

import (
	"strings"
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

// Thai and Lao vowels written after their consonant that are letters, not
// combining marks: Thai SARA A, SARA AA, SARA AM and LAKKHANGYAO, and Lao A,
// AA, AM and SEMIVOWEL SIGN NYO. A keyword ending right before one would leave
// that vowel behind.
const followingVowels = "\u0E30\u0E32\u0E33\u0E45\u0EB0\u0EB2\u0EB3\u0EBD"

// keywordSyllableMatch reports whether a match of a keyword in an unspaced
// syllabic script counts. No edge may split a cluster: the match must not end
// before a combining mark, nor start after a preposed vowel or a stacking sign
// that belongs to the next consonant. Each edge also follows the rune of the
// keyword at that edge: a Thai or Lao edge must not end before a following
// vowel; a Latin letter, a digit or another spaced-script edge must be a word
// edge, as for a spaced-script keyword; a CJK edge has no further check.
func keywordSyllableMatch(text string, start, end int) bool {
	first, _ := utf8.DecodeRuneInString(text[start:end])
	last, _ := utf8.DecodeLastRuneInString(text[start:end])
	return keywordStartMatch(text, start, first) && keywordEndMatch(text, end, last)
}

func keywordStartMatch(text string, start int, first rune) bool {
	if start == 0 {
		return true
	}
	prev, _ := utf8.DecodeLastRuneInString(text[:start])
	if unicode.Is(unicode.Logical_Order_Exception, prev) || prev == khmerCoeng || prev == myanmarVirama {
		return false
	}
	if isCJK(first) || isSyllabicScript(first) {
		return true
	}
	return prev != '_' && !isBoundaryBlockingRune(prev)
}

func keywordEndMatch(text string, end int, last rune) bool {
	if end == len(text) {
		return true
	}
	next, _ := utf8.DecodeRuneInString(text[end:])
	if unicode.In(next, unicode.Mn, unicode.Mc) {
		return false
	}
	if isCJK(last) {
		return true
	}
	if isSyllabicScript(last) {
		return !strings.ContainsRune(followingVowels, next)
	}
	return next != '_' && !isBoundaryBlockingRune(next)
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

// isSyllabicScript reports letters and marks of the unspaced syllabic scripts.
// Their digits are excluded so a digit or identifier keyword keeps word
// boundaries.
func isSyllabicScript(r rune) bool {
	return (unicode.IsLetter(r) || unicode.IsMark(r)) &&
		unicode.In(r, unicode.Thai, unicode.Lao, unicode.Khmer, unicode.Myanmar)
}
