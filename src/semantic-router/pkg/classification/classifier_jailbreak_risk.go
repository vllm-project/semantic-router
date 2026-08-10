package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// jailbreakPositiveLabel is the default mapping label that denotes an actual
// jailbreak attempt (as opposed to benign / other classes), used when
// prompt_guard.positive_labels is not configured.
const jailbreakPositiveLabel = "jailbreak"

// resolvePositiveLabels returns the configured positive_labels, or the
// single-label legacy default when unset.
func resolvePositiveLabels(configured []string) []string {
	if len(configured) > 0 {
		return configured
	}
	return []string{jailbreakPositiveLabel}
}

// isPositiveJailbreakLabel reports whether label counts as a positive (unsafe)
// jailbreak classification, per the configured positive_labels (defaulting to
// jailbreakPositiveLabel when unset).
func isPositiveJailbreakLabel(configured []string, label string) bool {
	for _, l := range resolvePositiveLabels(configured) {
		if l == label {
			return true
		}
	}
	return false
}

// validateJailbreakPositiveLabels enforces a fail-fast contract: when
// positive_labels is explicitly configured, at least one entry must exist in
// the jailbreak mapping's actual label set, so a misconfigured label (e.g. a
// custom model's positive class named "malicious" instead of "jailbreak")
// can never silently mean "this model's positive class is never detected."
// It is a no-op when positive_labels is unset, since the legacy default
// ("jailbreak" missing from a mapping) is an existing, tolerated case handled
// by jailbreakRiskScore's conservative fallback.
func validateJailbreakPositiveLabels(configured []string, mapping *JailbreakMapping) error {
	if len(configured) == 0 || mapping == nil {
		return nil
	}
	for _, label := range configured {
		if _, ok := mapping.GetIndexForJailbreakType(label); ok {
			return nil
		}
	}
	knownLabels := make([]string, 0, len(mapping.LabelToIdx))
	for label := range mapping.LabelToIdx {
		knownLabels = append(knownLabels, label)
	}
	return fmt.Errorf("prompt_guard.positive_labels %v: none match the jailbreak_mapping labels %v", configured, knownLabels)
}

// jailbreakRiskScore returns the probability that the input is a jailbreak — i.e.
// the probability mass on the positive_labels classes — independent of which
// class the model actually predicts (argmax).
//
// When the full softmax distribution is available it returns the exact summed
// P(positive_labels). Otherwise it derives a conservative estimate from the
// predicted-class confidence: the confidence itself when the predicted class is
// one of positive_labels, or 1-confidence otherwise (an upper bound on risk that
// is exact for binary models and never under-reports risk).
//
// This avoids the misleading case where a confident benign prediction reports a
// high risk_score: the predicted-class confidence is P(benign), not P(jailbreak).
func jailbreakRiskScore(mapping *JailbreakMapping, positiveLabels []string, result candle_binding.ClassResultWithProbs) float32 {
	labels := resolvePositiveLabels(positiveLabels)

	if mapping != nil && len(result.Probabilities) > 0 {
		var sum float32
		matched := false
		for _, label := range labels {
			if idx, ok := mapping.GetIndexForJailbreakType(label); ok &&
				idx >= 0 && idx < len(result.Probabilities) {
				sum += result.Probabilities[idx]
				matched = true
			}
		}
		if matched {
			return sum
		}
	}

	if mapping != nil {
		if predicted, ok := mapping.GetJailbreakTypeFromIndex(result.Class); ok &&
			isPositiveJailbreakLabel(labels, predicted) {
			return result.Confidence
		}
	}

	return 1 - result.Confidence
}

// isJailbreakRiskAboveThreshold reports whether result's summed positive-label
// risk score meets or exceeds threshold, returning that score alongside the
// decision. Shared by CheckForJailbreakWithRisk and the signal-evaluation
// path (findBestJailbreakMatch) so both always threshold the exact same
// value - see jailbreakRiskScore's doc comment for why this must stay
// independent of which class wins argmax.
func isJailbreakRiskAboveThreshold(mapping *JailbreakMapping, positiveLabels []string, result candle_binding.ClassResultWithProbs, threshold float32) (bool, float32) {
	riskScore := jailbreakRiskScore(mapping, positiveLabels, result)
	return riskScore >= threshold, riskScore
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

	result, err := c.jailbreakInference.Classify(text)
	if err != nil {
		return false, "", 0.0, 0.0, fmt.Errorf("jailbreak classification failed: %w", err)
	}

	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(result.Class)
	if !ok {
		return false, "", 0.0, 0.0, fmt.Errorf("unknown jailbreak class index: %d", result.Class)
	}

	isJailbreak, riskScore := isJailbreakRiskAboveThreshold(c.JailbreakMapping, c.Config.PromptGuard.PositiveLabels, result, threshold)

	if isJailbreak {
		logging.Warnf("JAILBREAK DETECTED: '%s' (confidence: %.3f, risk: %.3f, threshold: %.3f)",
			jailbreakType, result.Confidence, riskScore, threshold)
	}

	return isJailbreak, jailbreakType, result.Confidence, riskScore, nil
}
