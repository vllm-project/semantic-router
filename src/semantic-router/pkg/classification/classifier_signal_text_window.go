package classification

import (
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A model call made for a routing signal either samples the text or reads all
// of it, chosen by what a missed part of the text would cost.
//
//   - Sample. A signal that labels the request as a whole may read a head,
//     middle and tail view, which carries the request's purpose and bounds the
//     forward pass. textForRoutingSignal does this for
//     boundedSemanticSignalTypes and passes every other signal the full text.
//   - Read all. A question about whether something appears anywhere (PII, an
//     attack, an unsupported claim) must read all of the text, because a
//     dropped part is a silent miss. A systemone rule asks for that with
//     coverage: full, which reads the input in windows past one model input,
//     and on_unscanned decides what an answer that still missed part of it
//     means. coverage: truncate reads only the first tokens.
//
// semanticSignalUnitLimit bounds the three samples to 440 estimated tokens,
// which leaves room in a 512-token forward for the two omission markers and
// the special tokens. Units are quarter-token estimates calibrated on the
// mmBERT tokenizer.
const semanticSignalUnitLimit = 440 * 4

const signalWindowOmissionMarker = "\n... [content sampled for routing signal evaluation] ...\n"

var boundedSemanticSignalTypes = map[string]struct{}{
	config.SignalTypeLanguage: {},
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
