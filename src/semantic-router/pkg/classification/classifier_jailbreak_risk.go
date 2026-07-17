package classification

import (
	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// jailbreakPositiveLabel is the mapping label that denotes an actual jailbreak
// attempt (as opposed to benign / other classes).
const jailbreakPositiveLabel = "jailbreak"

// jailbreakRiskScore returns the probability that the input is a jailbreak — i.e.
// the probability mass on the jailbreak class — independent of which class the
// model actually predicts (argmax).
//
// When the full softmax distribution is available it returns the exact
// P(jailbreak). Otherwise it derives a conservative estimate from the
// predicted-class confidence: the confidence itself when the predicted class is
// jailbreak, or 1-confidence otherwise (an upper bound on P(jailbreak) that is
// exact for binary models and never under-reports risk).
//
// This avoids the misleading case where a confident benign prediction reports a
// high risk_score: the predicted-class confidence is P(benign), not P(jailbreak).
func jailbreakRiskScore(mapping *JailbreakMapping, result candle_binding.ClassResultWithProbs) float32 {
	if mapping != nil && len(result.Probabilities) > 0 {
		if idx, ok := mapping.GetIndexForJailbreakType(jailbreakPositiveLabel); ok &&
			idx >= 0 && idx < len(result.Probabilities) {
			return result.Probabilities[idx]
		}
	}

	if mapping != nil {
		if predicted, ok := mapping.GetJailbreakTypeFromIndex(result.Class); ok &&
			predicted == jailbreakPositiveLabel {
			return result.Confidence
		}
	}

	return 1 - result.Confidence
}
