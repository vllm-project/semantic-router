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
// Lowercasing is done inline per-rune to avoid allocating a full lowercase copy
// of the input string.
func TokenizeWords(text string) []string {
	var tokens []string
	for field := range strings.FieldsSeq(text) {
		tokens = appendFieldTokens(tokens, field)
	}
	return tokens
}

func appendFieldTokens(tokens []string, field string) []string {
	var nonCJK []byte
	var pending []byte
	// Pending runes replace the old cleanEnd trim: commit them only if another token rune follows.
	var prevCJK rune
	prevIsCJK := false
	started := false

	for _, r := range field {
		isTokenRune := unicode.IsLetter(r) || unicode.IsDigit(r)
		if !started && !isTokenRune {
			continue
		}
		started = true

		if !isTokenRune {
			pending = utf8.AppendRune(pending, unicode.ToLower(r))
			prevIsCJK = false
			continue
		}

		// A following token rune makes pending punctuation internal, so keep it.
		if len(pending) > 0 {
			nonCJK = append(nonCJK, pending...)
			pending = pending[:0]
		}

		if !isCJK(r) {
			nonCJK = utf8.AppendRune(nonCJK, unicode.ToLower(r))
			prevIsCJK = false
			continue
		}

		// CJK tokens stand alone; flush any accumulated non-CJK word before them.
		if len(nonCJK) > 0 {
			tokens = append(tokens, string(nonCJK))
			nonCJK = nonCJK[:0]
		}
		tokens = append(tokens, string(r))
		// Only adjacent CJK runes form bigrams; punctuation and non-CJK runes reset adjacency.
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
