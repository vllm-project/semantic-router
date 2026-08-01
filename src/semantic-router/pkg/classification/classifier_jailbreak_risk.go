package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

// CheckForJailbreakWithRisk analyzes text for jailbreak attempts and additionally
// returns a risk score equal to P(jailbreak class), independent of which class the
// model predicts. It mirrors CheckForJailbreak but is intended for callers (such as
// the security detection API) that report a risk score, so that a confident benign
// prediction produces a low risk score rather than a misleadingly high one.
func (c *Classifier) CheckForJailbreakWithRisk(text string) (bool, string, float32, float32, error) {
	threshold := c.Config.PromptGuard.Threshold

	if !c.IsJailbreakEnabled() {
		return false, "", 0.0, 0.0, fmt.Errorf("jailbreak detection is not enabled or properly configured")
	}

	if text == "" {
		return false, "", 0.0, 0.0, nil
	}

	result, err := c.classifyJailbreakWithProbs(text)
	if err != nil {
		return false, "", 0.0, 0.0, fmt.Errorf("jailbreak classification failed: %w", err)
	}

	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(result.Class)
	if !ok {
		return false, "", 0.0, 0.0, fmt.Errorf("unknown jailbreak class index: %d", result.Class)
	}

	isJailbreak := result.Confidence >= threshold && jailbreakType == jailbreakPositiveLabel
	riskScore := jailbreakRiskScore(c.JailbreakMapping, result)

	if isJailbreak {
		logging.Warnf("JAILBREAK DETECTED: '%s' (confidence: %.3f, risk: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, riskScore, threshold)
	}

	return isJailbreak, jailbreakType, result.Confidence, riskScore, nil
}

// classifyJailbreakWithProbs runs jailbreak inference, preferring the full
// probability distribution when the configured backend supports it and otherwise
// falling back to argmax-only classification.
func (c *Classifier) classifyJailbreakWithProbs(text string) (candle_binding.ClassResultWithProbs, error) {
	if probInf, ok := c.jailbreakInference.(JailbreakProbInference); ok {
		return probInf.ClassifyWithProbs(text)
	}

	result, err := c.jailbreakInference.Classify(text)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, err
	}
	return candle_binding.ClassResultWithProbs{Class: result.Class, Confidence: result.Confidence}, nil
}
