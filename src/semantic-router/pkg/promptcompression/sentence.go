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

	// runeAt decodes the rune at byte position pos (returns 0 if out of range).
	runeAt := func(pos int) rune {
		if pos < 0 || pos >= len(text) {
			return 0
		}
		r, _ := utf8.DecodeRuneInString(text[pos:])
		return r
	}

	startByte := 0 // byte offset of current sentence start
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

		_ = prevBytePos // used implicitly via prevRune

		// Skip decimal numbers like "3.14"
		if r == '.' && unicode.IsDigit(prevRune) {
			nextR := runeAt(bytePos + size)
			if unicode.IsDigit(nextR) {
				prevRune = r
				prevBytePos = bytePos
				bytePos += size
				continue
			}
		}

		// Skip common abbreviations (single uppercase letter + period)
		if r == '.' && unicode.IsUpper(prevRune) {
			prevPrevR := runeAt(prevBytePos - utf8.RuneLen(prevRune))
			atStart := prevBytePos == startByte
			afterSpace := prevPrevR == ' '
			if atStart || afterSpace {
				nextR := runeAt(bytePos + size)
				nextNextR := runeAt(bytePos + size + utf8.RuneLen(nextR))
				if nextR == ' ' && unicode.IsUpper(nextNextR) {
					prevRune = r
					prevBytePos = bytePos
					bytePos += size
					continue
				}
			}
		}

		// Consume trailing terminators
		endByte := bytePos + size
		for endByte < len(text) {
			nr, ns := utf8.DecodeRuneInString(text[endByte:])
			if !isTrailingTerminator(nr) {
				break
			}
			endByte += ns
		}

		sent := strings.TrimSpace(text[startByte:endByte])
		if sent != "" {
			sentences = append(sentences, sent)
		}

		// Skip whitespace after sentence
		for endByte < len(text) {
			nr, ns := utf8.DecodeRuneInString(text[endByte:])
			if !unicode.IsSpace(nr) {
				break
			}
			endByte += ns
		}
		startByte = endByte
		bytePos = endByte
		prevRune = 0
		prevBytePos = endByte
	}

	if startByte < len(text) {
		tail := strings.TrimSpace(text[startByte:])
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
//
// Lowercasing is done inline per-rune to avoid allocating a full lowercase copy
// of the input string.
func TokenizeWords(text string) []string {
	var tokens []string

	for _, field := range strings.Fields(text) {
		// Trim leading/trailing non-letter/digit runes, lowercasing as we go.
		// First, find byte bounds of the "cleaned" substring.
		cleanStart := 0
		for cleanStart < len(field) {
			r, sz := utf8.DecodeRuneInString(field[cleanStart:])
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				break
			}
			cleanStart += sz
		}
		cleanEnd := len(field)
		for cleanEnd > cleanStart {
			r, sz := utf8.DecodeLastRuneInString(field[:cleanEnd])
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				break
			}
			cleanEnd -= sz
		}
		if cleanStart >= cleanEnd {
			continue
		}
		cleaned := field[cleanStart:cleanEnd]

		// Check if there's any CJK content.
		hasCJK := false
		for i := 0; i < len(cleaned); {
			r, sz := utf8.DecodeRuneInString(cleaned[i:])
			if isCJK(r) {
				hasCJK = true
				break
			}
			i += sz
		}

		if !hasCJK {
			tokens = append(tokens, strings.ToLower(cleaned))
			continue
		}

		// Mixed or pure CJK: extract bigrams and lowercased non-CJK words.
		var nonCJK []byte
		var prevCJK rune
		prevIsCJK := false

		for i := 0; i < len(cleaned); {
			r, sz := utf8.DecodeRuneInString(cleaned[i:])
			if isCJK(r) {
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
			} else {
				lr := unicode.ToLower(r)
				nonCJK = utf8.AppendRune(nonCJK, lr)
				prevIsCJK = false
			}
			i += sz
		}
		if len(nonCJK) > 0 {
			tokens = append(tokens, string(nonCJK))
		}
	}

	return tokens
}
