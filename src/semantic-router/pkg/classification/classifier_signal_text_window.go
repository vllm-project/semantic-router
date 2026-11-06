package classification

import (
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// semanticSignalUnitLimit bounds one embedding/classifier forward pass. The
// request's full text still reaches exact-match, structure, and context signals;
// semantic signals receive representative head, middle, and tail views so a
// model with a large context window cannot force internal routing models to
// allocate against that same unbounded window. Units are quarter-token
// estimates calibrated on the mmBERT tokenizer.
const semanticSignalUnitLimit = 440 * 4

const signalWindowOmissionMarker = "\n... [content sampled for routing signal evaluation] ...\n"

var boundedSemanticSignalTypes = map[string]struct{}{
	config.SignalTypeLanguage: {},
	config.SignalTypeEvent:    {},
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

// systemOneQuestionText is the text a systemone rule reads as the current
// turn: the request as it came.
const systemOneQuestionText = "systemone_request"
