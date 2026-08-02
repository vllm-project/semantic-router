package classification

import (
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// semanticSignalRuneLimit bounds one embedding/classifier forward pass. The
// request's full text still reaches exact-match, structure, and context signals;
// semantic signals receive representative head, middle, and tail views so a
// model with a large context window cannot force internal routing models to
// allocate against that same unbounded window.
const semanticSignalRuneLimit = 2 * 1024

// Classification tokenizers currently cap PII and jailbreak inference at 512
// tokens even when the underlying mmBERT model supports a longer context. PII
// is local token classification and loses entity confidence in long windows;
// jailbreak detection benefits from broader semantic context. Keep separate
// budgets instead of forcing both security signals through one compromise.
// Units are quarter-token estimates: ordinary runes cost one, word starts add
// one, and CJK runes cost six in total (about 1.5 tokens).
const (
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
	return representativeSignalText(text, semanticSignalRuneLimit)
}

func representativeSignalText(text string, maxRunes int) string {
	runes := []rune(text)
	if maxRunes <= 0 || len(runes) <= maxRunes {
		return text
	}
	headRunes := maxRunes / 3
	middleRunes := maxRunes / 3
	tailRunes := maxRunes - headRunes - middleRunes
	middleStart := (len(runes) - middleRunes) / 2
	return string(runes[:headRunes]) +
		signalWindowOmissionMarker +
		string(runes[middleStart:middleStart+middleRunes]) +
		signalWindowOmissionMarker +
		string(runes[len(runes)-tailRunes:])
}

func piiSignalChunks(text string) []string {
	return uniqueSignalChunks(
		securitySignalChunks(text, piiSignalChunkBudget, piiSignalChunkOverlapRunes),
	)
}

func jailbreakSignalChunks(text string) []string {
	return securitySignalChunks(text, jailbreakSignalChunkBudget, jailbreakSignalOverlapRunes)
}

// securitySignalChunks scans the entire input in bounded, overlapping pieces.
// Unlike semantic intent signals, security detection cannot safely discard the
// middle of a long prompt.
func securitySignalChunks(text string, budget, overlapRunes int) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}
	if securitySignalChunkUnits(runes) <= budget {
		return []string{text}
	}

	chunks := make([]string, 0, (len(runes)/1024)+1)
	for start := 0; start < len(runes); {
		end := securitySignalChunkEnd(runes, start, budget)
		chunks = append(chunks, string(runes[start:end]))
		if end == len(runes) {
			break
		}
		start = max(start+1, end-overlapRunes)
	}
	return chunks
}

func securitySignalChunkEnd(runes []rune, start, budget int) int {
	units := 0
	inWord := false
	end := start
	for end < len(runes) {
		nextUnits, nextInWord := securitySignalRuneUnits(runes[end], inWord)
		if end > start && units+nextUnits > budget {
			break
		}
		units += nextUnits
		inWord = nextInWord
		end++
	}
	return end
}

func securitySignalChunkUnits(runes []rune) int {
	units := 0
	inWord := false
	for _, r := range runes {
		var runeUnits int
		runeUnits, inWord = securitySignalRuneUnits(r, inWord)
		units += runeUnits
	}
	return units
}

func securitySignalRuneUnits(r rune, inWord bool) (int, bool) {
	if isCJK(r) {
		return 6, false
	}
	if unicode.IsLetter(r) || unicode.IsDigit(r) {
		if !inWord {
			return 2, true
		}
		return 1, true
	}
	return 1, false
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
