package classification

import (
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// semanticSignalUnitLimit bounds one embedding/classifier forward pass. The
// request's full text still reaches exact-match, structure, and context signals;
// semantic signals receive representative head, middle, and tail views so a
// model with a large context window cannot force internal routing models to
// allocate against that same unbounded window. Units are quarter-token
// estimates calibrated on the mmBERT tokenizer.
const (
	semanticSignalUnitLimit     = 440 * 4
	piiSignalChunkBudget        = 128 * 4
	piiSignalChunkOverlapRunes  = 64
	jailbreakSignalChunkBudget  = 384 * 4
	jailbreakSignalOverlapRunes = 64
)

const signalWindowOmissionMarker = "\n... [content sampled for routing signal evaluation] ...\n"

var boundedSemanticSignalTypes = map[string]struct{}{
	config.SignalTypeEmbedding:    {},
	config.SignalTypeDomain:       {},
	config.SignalTypeFactCheck:    {},
	config.SignalTypeUserFeedback: {},
	config.SignalTypeReask:        {},
	config.SignalTypePreference:   {},
	config.SignalTypeLanguage:     {},
	config.SignalTypeComplexity:   {},
	config.SignalTypeModality:     {},
	config.SignalTypeKB:           {},
	config.SignalTypeEvent:        {},
}

func textForRoutingSignal(signalType, text string) string {
	if _, bounded := boundedSemanticSignalTypes[signalType]; !bounded {
		return text
	}
	return representativeSignalText(text, semanticSignalUnitLimit)
}

func representativeSignalText(text string, maxUnits int) string {
	runes := []rune(text)
	prefix := signalUnitPrefix(runes)
	if maxUnits <= 0 || prefix[len(runes)] <= maxUnits {
		return text
	}
	part := maxUnits / 3
	headEnd := signalUnitsForward(prefix, 0, part)
	tailStart := signalUnitsBackward(prefix, len(runes), part)
	center := len(runes) / 2
	middleStart := max(signalUnitsBackward(prefix, center, part/2), headEnd)
	middleEnd := min(signalUnitsForward(prefix, center, part-part/2), tailStart)
	return string(runes[:headEnd]) +
		signalWindowOmissionMarker +
		string(runes[middleStart:middleEnd]) +
		signalWindowOmissionMarker +
		string(runes[tailStart:])
}

func piiSignalChunks(text string) []string {
	return uniqueSignalChunks(
		securitySignalChunks(text, piiSignalChunkBudget, piiSignalChunkOverlapRunes),
	)
}

// piiSignalChunkSpans is piiSignalChunks with offsets, for callers that report
// entity positions. It does not de-duplicate: two identical chunks are at
// different offsets, and both of those positions are real.
func piiSignalChunkSpans(text string) []signalChunkSpan {
	return securitySignalChunkSpans(text, piiSignalChunkBudget, piiSignalChunkOverlapRunes)
}

func jailbreakSignalChunks(text string) []string {
	return uniqueSignalChunks(
		securitySignalChunks(text, jailbreakSignalChunkBudget, jailbreakSignalOverlapRunes),
	)
}

// signalChunkSpan is one chunk together with its byte offset in the text it was
// cut from. Callers that report positions back to a client need the offset to
// map a chunk-relative entity onto the original text.
type signalChunkSpan struct {
	Text      string
	StartByte int
}

// securitySignalChunks scans the entire input in bounded, overlapping pieces.
// Unlike semantic intent signals, security detection cannot safely discard the
// middle of a long prompt.
func securitySignalChunks(text string, budget, overlapRunes int) []string {
	spans := securitySignalChunkSpans(text, budget, overlapRunes)
	if spans == nil {
		return nil
	}
	chunks := make([]string, len(spans))
	for i, span := range spans {
		chunks[i] = span.Text
	}
	return chunks
}

// securitySignalChunkSpans is securitySignalChunks with each chunk's byte
// offset in text. An entity found at offset e in span s sits at
// s.StartByte + e in the original text.
func securitySignalChunkSpans(text string, budget, overlapRunes int) []signalChunkSpan {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}
	prefix := signalUnitPrefix(runes)
	if prefix[len(runes)] <= budget {
		return []signalChunkSpan{{Text: text, StartByte: 0}}
	}

	spans := make([]signalChunkSpan, 0, (len(runes)/1024)+1)
	// Chunk starts only ever move forward, so one cursor converts them to byte
	// offsets without a per-rune index.
	startByte, counted := 0, 0
	for start := 0; start < len(runes); {
		for ; counted < start; counted++ {
			startByte += utf8.RuneLen(runes[counted])
		}
		end := securitySignalChunkEnd(runes, prefix, start, budget)
		spans = append(spans, signalChunkSpan{
			Text:      string(runes[start:end]),
			StartByte: startByte,
		})
		if end == len(runes) {
			break
		}
		start = max(start+1, end-overlapRunes)
	}
	return spans
}

func securitySignalChunkEnd(runes []rune, prefix []int, start, budget int) int {
	end := max(signalUnitsForward(prefix, start, budget-1), start+1)
	if end >= len(runes) || unicode.IsSpace(runes[end]) {
		return end
	}
	for i := end; i > start+1 && i > end-64; i-- {
		if unicode.IsSpace(runes[i-1]) {
			return i
		}
	}
	return end
}

func securitySignalChunkUnits(runes []rune) int {
	return signalUnitPrefix(runes)[len(runes)]
}

func signalUnitPrefix(runes []rune) []int {
	prefix := make([]int, len(runes)+1)
	inWord, wordHasDigit := false, false
	for i, r := range runes {
		var units int
		units, inWord, wordHasDigit = signalRuneUnits(r, inWord, wordHasDigit)
		prefix[i+1] = prefix[i] + units
	}
	return prefix
}

func signalRuneUnits(r rune, inWord, wordHasDigit bool) (int, bool, bool) {
	switch {
	case unicode.IsSpace(r):
		return 0, false, false
	case isCJK(r):
		return 4, false, false
	case unicode.IsDigit(r):
		return 5, true, true
	case unicode.IsLetter(r):
		if inWord && wordHasDigit {
			return 4, true, true
		}
		units := 1
		if !unicode.Is(unicode.Latin, r) {
			units = 2
		}
		if !inWord {
			units++
		}
		return units, true, false
	case unicode.IsSymbol(r):
		return 5, false, false
	default:
		return 4, false, false
	}
}

func signalUnitsForward(prefix []int, start, budget int) int {
	end := start
	for end < len(prefix)-1 && prefix[end+1]-prefix[start] <= budget {
		end++
	}
	return end
}

func signalUnitsBackward(prefix []int, end, budget int) int {
	start := end
	for start > 0 && prefix[end]-prefix[start-1] <= budget {
		start--
	}
	return start
}

// uniqueSignalChunks removes exact duplicate inference work while preserving
// scan order. Generated logs, repeated quoted content, and padded eval inputs
// can contain hundreds of identical security windows; classifying the same
// bytes again cannot improve recall.
func uniqueSignalChunks(chunks []string) []string {
	if len(chunks) < 2 {
		return chunks
	}
	seen := make(map[string]struct{}, len(chunks))
	unique := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		if _, duplicate := seen[chunk]; duplicate {
			continue
		}
		seen[chunk] = struct{}{}
		unique = append(unique, chunk)
	}
	return unique
}
