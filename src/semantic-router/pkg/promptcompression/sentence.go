package promptcompression

import (
	"strings"
	"unicode"
	"unicode/utf8"
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
//
// Operates directly on the UTF-8 byte string using utf8.DecodeRuneInString to
// avoid allocating a []rune copy of the full text.
func SplitSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	var sentences []string
	startByte := 0
	prevRune := rune(0)
	prevBytePos := 0

	for bytePos := 0; bytePos < len(text); {
		r, size := utf8.DecodeRuneInString(text[bytePos:])

		if !isSentenceTerminator(r) {
			prevRune = r
			prevBytePos = bytePos
			bytePos += size
			continue
		}

		if shouldSkipSentenceTerminator(text, startByte, bytePos, size, r, prevRune, prevBytePos) {
			prevRune = r
			prevBytePos = bytePos
			bytePos += size
			continue
		}

		endByte := consumeTrailingTerminators(text, bytePos+size)
		if sent := strings.TrimSpace(text[startByte:endByte]); sent != "" {
			sentences = append(sentences, sent)
		}

		endByte = consumeWhitespace(text, endByte)
		startByte = endByte
		bytePos = endByte
		prevRune = 0
		prevBytePos = endByte
	}

	if startByte < len(text) {
		if tail := strings.TrimSpace(text[startByte:]); tail != "" {
			sentences = append(sentences, tail)
		}
	}

	return sentences
}

// shouldSkipSentenceTerminator keeps periods that are part of numbers or abbreviations.
func shouldSkipSentenceTerminator(
	text string,
	startByte, bytePos, size int,
	r, prevRune rune,
	prevBytePos int,
) bool {
	if r != '.' {
		return false
	}
	return isDecimalPoint(text, bytePos, size, prevRune) ||
		isSingleLetterAbbreviation(text, startByte, bytePos, size, prevRune, prevBytePos)
}

// isDecimalPoint preserves decimal numbers such as 3.14.
func isDecimalPoint(text string, bytePos, size int, prevRune rune) bool {
	return unicode.IsDigit(prevRune) && unicode.IsDigit(runeAt(text, bytePos+size))
}

// isSingleLetterAbbreviation preserves the existing single-capital heuristic, e.g. "A. B".
func isSingleLetterAbbreviation(
	text string,
	startByte, bytePos, size int,
	prevRune rune,
	prevBytePos int,
) bool {
	if !unicode.IsUpper(prevRune) {
		return false
	}

	prevPrevR := runeAt(text, prevBytePos-utf8.RuneLen(prevRune))
	if prevBytePos != startByte && prevPrevR != ' ' {
		return false
	}

	nextR := runeAt(text, bytePos+size)
	nextNextR := runeAt(text, bytePos+size+utf8.RuneLen(nextR))
	return nextR == ' ' && unicode.IsUpper(nextNextR)
}

// consumeTrailingTerminators keeps repeated sentence marks like "?!" with the sentence.
func consumeTrailingTerminators(text string, pos int) int {
	for pos < len(text) {
		r, size := utf8.DecodeRuneInString(text[pos:])
		if !isTrailingTerminator(r) {
			break
		}
		pos += size
	}
	return pos
}

// consumeWhitespace advances the next sentence start past inter-sentence spacing.
func consumeWhitespace(text string, pos int) int {
	for pos < len(text) {
		r, size := utf8.DecodeRuneInString(text[pos:])
		if !unicode.IsSpace(r) {
			break
		}
		pos += size
	}
	return pos
}

// runeAt decodes the rune at byte position pos, returning 0 when pos is outside text.
func runeAt(text string, pos int) rune {
	if pos < 0 || pos >= len(text) {
		return 0
	}
	r, _ := utf8.DecodeRuneInString(text[pos:])
	return r
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
	hasFields := false

	// Split on whitespace lazily; within each field count CJK runes separately.
	for field := range strings.FieldsSeq(text) {
		hasFields = true
		fieldCJKRunes := 0
		hasNonCJK := false
		for _, r := range field {
			if isCJK(r) {
				fieldCJKRunes++
			} else {
				hasNonCJK = true
			}
		}

		cjkRunes += fieldCJKRunes
		// Non-CJK fields and mixed fields (e.g. "Python函数") each add one word.
		if fieldCJKRunes == 0 || hasNonCJK {
			nonCJKWords++
		}
	}

	cjkTokens := float64(cjkRunes) * 1.5
	wordTokens := float64(nonCJKWords) * 1.3
	total := int(cjkTokens + wordTokens)
	if total == 0 && hasFields {
		total = 1
	}
	return total
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
//
// Non-CJK fields retain the zero-copy substring fast path; mixed CJK fields
// lowercase non-CJK runs inline while producing CJK tokens.
func TokenizeWords(text string) []string {
	var tokens []string

	for _, field := range strings.Fields(text) {
		cleaned := trimTokenField(field)
		if cleaned == "" {
			continue
		}

		// Keep the common non-CJK path on the original substring.
		if !containsCJK(cleaned) {
			tokens = append(tokens, strings.ToLower(cleaned))
			continue
		}

		tokens = appendMixedFieldTokens(tokens, cleaned)
	}

	return tokens
}

// trimTokenField returns a zero-copy view without leading or trailing punctuation.
func trimTokenField(field string) string {
	start := 0
	for start < len(field) {
		r, size := utf8.DecodeRuneInString(field[start:])
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			break
		}
		start += size
	}

	end := len(field)
	for end > start {
		r, size := utf8.DecodeLastRuneInString(field[:end])
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			break
		}
		end -= size
	}
	return field[start:end]
}

func containsCJK(text string) bool {
	for _, r := range text {
		if isCJK(r) {
			return true
		}
	}
	return false
}

func appendMixedFieldTokens(tokens []string, field string) []string {
	var nonCJK []byte
	var prevCJK rune
	prevIsCJK := false

	for _, r := range field {
		if !isCJK(r) {
			nonCJK = utf8.AppendRune(nonCJK, unicode.ToLower(r))
			prevIsCJK = false
			continue
		}

		if len(nonCJK) > 0 {
			tokens = append(tokens, string(nonCJK))
			nonCJK = nonCJK[:0]
		}
		tokens = append(tokens, string(r))
		if prevIsCJK {
			tokens = append(tokens, string([]rune{prevCJK, r}))
		}
		prevCJK = r
		prevIsCJK = true
	}

	if len(nonCJK) > 0 {
		tokens = append(tokens, string(nonCJK))
	}
	return tokens
}
