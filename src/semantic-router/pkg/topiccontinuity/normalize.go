package topiccontinuity

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// phraseToken is one normalized token of a phrase view. Offset is the byte
// offset of the token's first rune in the original segment; it is never taken
// from a lowercased or stripped string, whose byte lengths may differ. Run
// identifies the unmasked run the token came from: multi-token phrases never
// match across runs.
type phraseToken struct {
	Text   string
	Offset int
	Run    int
}

// byteRange is a half-open byte range [Start, End) over an original segment.
type byteRange struct {
	Start, End int
}

func (r byteRange) contains(offset int) bool {
	return offset >= r.Start && offset < r.End
}

// continuationView tokenizes a whole segment as one run. Nothing is stripped,
// so no delimiter can hide continuation evidence.
func continuationView(segment textSegment) []phraseToken {
	text := string(segment)
	return tokenizeRuns(text, []byteRange{{Start: 0, End: len(text)}})
}

// changeView tokenizes the segment with code and double-quoted regions masked.
// Masked ranges are phrase barriers: each unmasked run is tokenized
// separately and keeps original offsets.
func changeView(segment textSegment) []phraseToken {
	text := string(segment)
	return tokenizeRuns(text, unmaskedRuns(len(text), maskedRanges(text)))
}

// tokenizeRuns walks each run rune by rune over the original text, lowercases
// each rune individually, and records original offsets.
func tokenizeRuns(text string, runs []byteRange) []phraseToken {
	var tokens []phraseToken
	for runIndex, run := range runs {
		var builder strings.Builder
		start := -1
		flush := func() {
			if start >= 0 {
				if token, offset := trimApostrophes(builder.String(), start, text); token != "" {
					tokens = append(tokens, phraseToken{Text: token, Offset: offset, Run: runIndex})
				}
			}
			builder.Reset()
			start = -1
		}
		for offset := run.Start; offset < run.End; {
			value, width := utf8.DecodeRuneInString(text[offset:run.End])
			if isPhraseRune(value) {
				if start < 0 {
					start = offset
				}
				builder.WriteRune(unicode.ToLower(value))
			} else {
				flush()
			}
			offset += width
		}
		flush()
	}
	return tokens
}

func isPhraseRune(value rune) bool {
	return unicode.IsLetter(value) || unicode.IsDigit(value) || value == '\'' || value == '’'
}

// trimApostrophes removes ' and ’ from a token's edges, keeping them inside
// words ("isn't"). The returned offset moves past any trimmed leading runes,
// measured in the original text.
func trimApostrophes(token string, start int, original string) (string, int) {
	offset := start
	for token != "" {
		value, width := utf8.DecodeRuneInString(token)
		if value != '\'' && value != '’' {
			break
		}
		token = token[width:]
		_, originalWidth := utf8.DecodeRuneInString(original[offset:])
		offset += originalWidth
	}
	for token != "" {
		value, width := utf8.DecodeLastRuneInString(token)
		if value != '\'' && value != '’' {
			break
		}
		token = token[:len(token)-width]
	}
	return token, offset
}

// maskedRanges returns the code and double-quoted regions of a segment, in
// order: fenced code, backtick spans, ASCII double quotes, and curly double
// quotes. An unclosed delimiter masks to the end of the segment, which only
// ever hides change evidence.
func maskedRanges(text string) []byteRange {
	var ranges []byteRange
	for offset := 0; offset < len(text); {
		open, closer, width := maskOpener(text, offset)
		if open == "" {
			_, size := utf8.DecodeRuneInString(text[offset:])
			offset += size
			continue
		}
		end := len(text)
		if index := strings.Index(text[offset+width:], closer); index >= 0 {
			end = offset + width + index + len(closer)
		}
		ranges = append(ranges, byteRange{Start: offset, End: end})
		offset = end
	}
	return ranges
}

func maskOpener(text string, offset int) (open, closer string, width int) {
	switch {
	case strings.HasPrefix(text[offset:], "```"):
		return "```", "```", 3
	case text[offset] == '`':
		return "`", "`", 1
	case text[offset] == '"':
		return `"`, `"`, 1
	case strings.HasPrefix(text[offset:], "“"):
		return "“", "”", len("“")
	}
	return "", "", 0
}

func unmaskedRuns(length int, masked []byteRange) []byteRange {
	var runs []byteRange
	cursor := 0
	for _, mask := range masked {
		if mask.Start > cursor {
			runs = append(runs, byteRange{Start: cursor, End: mask.Start})
		}
		cursor = mask.End
	}
	if cursor < length {
		runs = append(runs, byteRange{Start: cursor, End: length})
	}
	return runs
}

// singleQuoteSpans finds paired single-quote spans on the original segment.
// A pair counts only when the opening quote starts the segment or follows
// whitespace or punctuation, the closing quote ends the segment or precedes
// whitespace or punctuation, and both sit on the same line. Apostrophes
// inside words never open a pair.
func singleQuoteSpans(text string) []byteRange {
	var spans []byteRange
	for offset := 0; offset < len(text); {
		value, width := utf8.DecodeRuneInString(text[offset:])
		closing, opens := quoteCloser(value)
		if !opens || !boundaryBefore(text, offset) {
			offset += width
			continue
		}
		end := findQuoteClose(text, offset+width, closing)
		if end < 0 {
			offset += width
			continue
		}
		spans = append(spans, byteRange{Start: offset, End: end})
		offset = end
	}
	return spans
}

func quoteCloser(value rune) (rune, bool) {
	switch value {
	case '\'':
		return '\'', true
	case '‘':
		return '’', true
	}
	return 0, false
}

// findQuoteClose returns the end offset of the closing quote, or -1.
func findQuoteClose(text string, from int, closing rune) int {
	for offset := from; offset < len(text); {
		value, width := utf8.DecodeRuneInString(text[offset:])
		if value == '\n' {
			return -1
		}
		if value == closing && offset > from && boundaryAfter(text, offset+width) {
			return offset + width
		}
		offset += width
	}
	return -1
}

func boundaryBefore(text string, offset int) bool {
	if offset == 0 {
		return true
	}
	value, _ := utf8.DecodeLastRuneInString(text[:offset])
	return unicode.IsSpace(value) || unicode.IsPunct(value)
}

func boundaryAfter(text string, offset int) bool {
	if offset >= len(text) {
		return true
	}
	value, _ := utf8.DecodeRuneInString(text[offset:])
	return unicode.IsSpace(value) || unicode.IsPunct(value)
}

// phraseAt reports whether phrase matches tokens starting at index, with all
// tokens in the same run.
func phraseAt(tokens []phraseToken, index int, phrase []string) bool {
	if index+len(phrase) > len(tokens) {
		return false
	}
	run := tokens[index].Run
	for i, word := range phrase {
		token := tokens[index+i]
		if token.Run != run || token.Text != word {
			return false
		}
	}
	return true
}

// matchAny returns the length of the first (longest) phrase matching at index.
func matchAny(tokens []phraseToken, index int, phrases [][]string) int {
	for _, phrase := range phrases {
		if phraseAt(tokens, index, phrase) {
			return len(phrase)
		}
	}
	return 0
}

// leadingAcknowledgements returns how many leading tokens are covered,
// without gaps, by acknowledgement phrases.
func leadingAcknowledgements(tokens []phraseToken) int {
	index := 0
	for index < len(tokens) {
		width := matchAny(tokens, index, acknowledgementPhrases)
		if width == 0 {
			break
		}
		index += width
	}
	return index
}
