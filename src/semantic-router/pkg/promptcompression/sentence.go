package promptcompression

import (
	"strings"
	"unicode"
)

// isSentenceTerminator returns true for sentence-ending punctuation across scripts.
//
//	Latin/Cyrillic: . ! ?
//	CJK fullwidth:  。！？
//	Arabic:         ؟ (U+061F)
//	Devanagari:     । (U+0964) ॥ (U+0965)
//	Thai:           ฯ (U+0E2F, abbreviation) — Thai rarely has explicit sentence-end
//	Ethiopic:       ። (U+1362)
//	Armenian:       ։ (U+0589)
func isSentenceTerminator(r rune) bool {
	switch r {
	case '.', '!', '?',
		'\u3002', // 。 CJK fullwidth period
		'\uFF01', // ！ CJK fullwidth exclamation
		'\uFF1F', // ？ CJK fullwidth question
		'\u061F', // ؟  Arabic question mark
		'\u0964', // ।  Devanagari danda
		'\u0965', // ॥  Devanagari double danda
		'\u1362', // ።  Ethiopic full stop
		'\u0589': // ։  Armenian full stop
		return true
	}
	return false
}

// isTrailingTerminator returns true for punctuation that can follow a sentence
// terminator (e.g. "..." "?!" or fullwidth equivalents).
func isTrailingTerminator(r rune) bool {
	return isSentenceTerminator(r)
}

// SplitSentences segments text into sentences using punctuation-based heuristics.
// Supports Latin, CJK (Chinese/Japanese/Korean), Arabic, Devanagari, and other
// Unicode scripts. Handles abbreviations and decimal numbers for Latin text.
func SplitSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	var sentences []string
	runes := []rune(text)
	start := 0

	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if !isSentenceTerminator(r) {
			continue
		}

		// Skip decimal numbers like "3.14" (only for ASCII period)
		if r == '.' && i > 0 && unicode.IsDigit(runes[i-1]) &&
			i+1 < len(runes) && unicode.IsDigit(runes[i+1]) {
			continue
		}

		// Skip common abbreviations (single uppercase letter + period)
		if r == '.' && i > 0 && unicode.IsUpper(runes[i-1]) {
			if i-1 == start || (i-2 >= start && runes[i-2] == ' ') {
				if i+1 < len(runes) && runes[i+1] == ' ' && i+2 < len(runes) && unicode.IsUpper(runes[i+2]) {
					continue
				}
			}
		}

		// Consume trailing punctuation
		end := i + 1
		for end < len(runes) && isTrailingTerminator(runes[end]) {
			end++
		}

		sent := strings.TrimSpace(string(runes[start:end]))
		if sent != "" {
			sentences = append(sentences, sent)
		}
		for end < len(runes) && unicode.IsSpace(runes[end]) {
			end++
		}
		start = end
		i = end - 1
	}

	if start < len(runes) {
		tail := strings.TrimSpace(string(runes[start:]))
		if tail != "" {
			sentences = append(sentences, tail)
		}
	}

	return sentences
}

// isCJK returns true if the rune belongs to a CJK Unified Ideographs block,
// Hiragana, Katakana, Hangul, or CJK compatibility ranges.
func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

// CountTokensApprox estimates the BPE token count for mixed-script text.
//
// For whitespace-delimited words (Latin, Cyrillic, Arabic, etc.): ~1.3 tokens
// per word (Sennrich et al. 2016, "Neural Machine Translation of Rare Words
// with Subword Units").
//
// For CJK characters with no word boundaries: each ideograph ≈ 1.5 BPE tokens
// on average in multilingual BERT/GPT tokenizers, because common characters are
// single tokens while rare ones get split into byte-level pieces.
func CountTokensApprox(text string) int {
	if text == "" {
		return 0
	}

	var cjkRunes int
	var nonCJKWords int

	// Split on whitespace; within each field count CJK runes separately
	for _, field := range strings.Fields(text) {
		fieldRunes := []rune(field)
		hasCJK := false
		for _, r := range fieldRunes {
			if isCJK(r) {
				cjkRunes++
				hasCJK = true
			}
		}
		// Non-CJK portion of the field counts as a word
		if !hasCJK {
			nonCJKWords++
		} else {
			// Mixed field (e.g. "Python函数"): count non-CJK chars as partial word
			nonCJKCount := len(fieldRunes) - cjkRunesInField(fieldRunes)
			if nonCJKCount > 0 {
				nonCJKWords++
			}
		}
	}

	cjkTokens := float64(cjkRunes) * 1.5
	wordTokens := float64(nonCJKWords) * 1.3
	total := int(cjkTokens + wordTokens)
	if total == 0 && len(strings.Fields(text)) > 0 {
		total = 1
	}
	return total
}

func cjkRunesInField(runes []rune) int {
	n := 0
	for _, r := range runes {
		if isCJK(r) {
			n++
		}
	}
	return n
}

// TokenizeWords splits text into tokens suitable for bag-of-words scoring.
//
// For whitespace-delimited scripts (Latin, Cyrillic, Arabic): lowercased words
// with punctuation stripped.
//
// For CJK text without word boundaries: character bigrams (sliding window of 2).
// This follows McNamee & Mayfield (SIGIR 2004, "Character N-Gram Tokenization
// for European Language Text Retrieval") who showed character n-grams are
// competitive with word-level tokenization for information retrieval across
// languages. Bigrams naturally capture most Chinese/Japanese 2-character words
// (e.g. "调试" debug, "函数" function, "数据" data).
func TokenizeWords(text string) []string {
	text = strings.ToLower(text)
	var tokens []string

	for _, field := range strings.Fields(text) {
		cleaned := strings.TrimFunc(field, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		})
		if cleaned == "" {
			continue
		}

		runes := []rune(cleaned)
		cjkCount := 0
		for _, r := range runes {
			if isCJK(r) {
				cjkCount++
			}
		}

		if cjkCount == 0 {
			tokens = append(tokens, cleaned)
			continue
		}

		// Extract CJK bigrams and non-CJK words from mixed tokens
		var nonCJK []rune
		for j := 0; j < len(runes); j++ {
			if isCJK(runes[j]) {
				// Flush accumulated non-CJK text as a word
				if len(nonCJK) > 0 {
					tokens = append(tokens, string(nonCJK))
					nonCJK = nil
				}
				// Unigram (always emitted so single-char terms are captured)
				tokens = append(tokens, string(runes[j]))
				// Bigram
				if j+1 < len(runes) && isCJK(runes[j+1]) {
					tokens = append(tokens, string(runes[j:j+2]))
				}
			} else {
				nonCJK = append(nonCJK, runes[j])
			}
		}
		if len(nonCJK) > 0 {
			tokens = append(tokens, string(nonCJK))
		}
	}

	return tokens
}
