package classification

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// errNothingScored marks a scan that produced neither a result nor a failure,
// which is what an empty text does: there was nothing to score, and that is not
// a detection and not a failure.
var errNothingScored = errors.New("no text was scored")

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
func jailbreakRiskScore(mapping *JailbreakMapping, positiveLabels []string, result SequenceClassificationResult) float32 {
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

	class, confidence := deriveArgmax(result.Probabilities)
	if mapping != nil {
		if predicted, ok := mapping.GetJailbreakTypeFromIndex(class); ok &&
			isPositiveJailbreakLabel(labels, predicted) {
			return confidence
		}
	}

	return 1 - confidence
}

// isJailbreakRiskAboveThreshold reports whether result's summed positive-label
// risk score meets or exceeds threshold, returning that score alongside the
// decision. The signal-evaluation path (findBestJailbreakMatch) thresholds
// through here and ScanJailbreakRisk's callers threshold the same
// jailbreakRiskScore value, so no surface ends up thresholding a different
// quantity - see jailbreakRiskScore's doc comment for why this must stay
// independent of which class wins argmax.
func isJailbreakRiskAboveThreshold(mapping *JailbreakMapping, positiveLabels []string, result SequenceClassificationResult, threshold float32) (bool, float32) {
	riskScore := jailbreakRiskScore(mapping, positiveLabels, result)
	return riskScore >= threshold, riskScore
}

// scanJailbreakChunks classifies text one chunk at a time and returns the
// riskiest chunk's result. The model truncates at its own sequence limit, so a
// single call only ever sees the start of a long text; every jailbreak surface
// scans through here so none of them can silently answer on a prefix.
//
// A chunk that fails is skipped so a match in another chunk still counts, the
// way the routing path keeps scanning past an unresolved chunk. lastErr keeps
// the failure whether or not other chunks were scored: a clean verdict needs
// every chunk, so the callers turn a partial scan without a match into an
// error rather than a clean result. scanned is false when no chunk produced a
// result.
func (c *Classifier) scanJailbreakChunks(ctx context.Context, text string) (result SequenceClassificationResult, scanned bool, lastErr error) {
	bestRisk := float32(-1)
	for _, chunk := range jailbreakSignalChunks(text) {
		chunkResult, err := c.jailbreakInference.Classify(ctx, chunk)
		if err != nil {
			logging.Errorf("jailbreak classification failed on one chunk: %v", err)
			lastErr = err
			continue
		}
		risk := jailbreakRiskScore(c.JailbreakMapping, c.Config.PromptGuard.PositiveLabels, chunkResult)
		if risk > bestRisk {
			bestRisk = risk
			result = chunkResult
		}
	}
	return result, bestRisk >= 0, lastErr
}

// CheckForJailbreakWithRisk analyzes text for jailbreak attempts and additionally
// returns a risk score equal to P(jailbreak class), independent of which class the
// model predicts. It mirrors CheckForJailbreak but is intended for callers (such as
// the security detection API) that report a risk score, so that a confident benign
// prediction produces a low risk score rather than a misleadingly high one. ctx is
// forwarded to the configured backend so a remote (http_chat/http_classify) call
// can be cancelled with the caller instead of always running to its own timeout.
func (c *Classifier) CheckForJailbreakWithRisk(ctx context.Context, text string) (bool, string, float32, float32, error) {
	return c.CheckForJailbreakRiskWithThreshold(ctx, text, c.Config.PromptGuard.Threshold)
}

// JailbreakScan is what one scan of a text saw, before any threshold is drawn
// across it: the riskiest chunk's predicted type and confidence, its
// positive-label risk score, and, when some chunk could not be scored while
// another was, the failure that leaves the text only partly inspected.
//
// PartialErr sits next to the score rather than being folded into an error
// because whether a skipped chunk matters depends on where the line is drawn. A
// score at or above a threshold is a match whatever the skipped chunk held,
// while a score below it is clean only if every chunk was scored. A caller with
// one threshold resolves that once (CheckForJailbreakRiskWithThreshold); a
// caller that thresholds the same score once per rule has to resolve it per
// rule, or a rule whose threshold the score misses is published as clean on a
// text that was never fully read.
type JailbreakScan struct {
	Type       string
	Confidence float32
	RiskScore  float32
	PartialErr error
}

// ScanJailbreakRisk scans text in chunks and reports what it saw, without
// applying a threshold.
//
// The model truncates at its own sequence limit, so a single call only ever
// sees the start of a long text. The routing path already scans the whole text
// in chunks and keeps the riskiest one; every jailbreak surface goes through
// here so they all answer the same question. A chunk that fails does not stop
// the scan, because a genuine match in a later chunk still has to survive a
// transient failure in an earlier one; it is reported as PartialErr instead.
// An error is returned only when no chunk was scored at all.
func (c *Classifier) ScanJailbreakRisk(ctx context.Context, text string) (JailbreakScan, error) {
	if !c.IsJailbreakEnabled() {
		return JailbreakScan{}, fmt.Errorf("jailbreak detection is not enabled or properly configured")
	}

	result, scanned, lastErr := c.scanJailbreakChunks(ctx, text)
	if !scanned {
		if lastErr != nil {
			return JailbreakScan{}, fmt.Errorf("jailbreak classification failed: %w", lastErr)
		}
		return JailbreakScan{}, errNothingScored
	}

	class, confidence := deriveArgmax(result.Probabilities)
	jailbreakType, ok := c.JailbreakMapping.GetJailbreakTypeFromIndex(class)
	if !ok {
		return JailbreakScan{}, fmt.Errorf("unknown jailbreak class index: %d", class)
	}

	return JailbreakScan{
		Type:       jailbreakType,
		Confidence: confidence,
		RiskScore:  jailbreakRiskScore(c.JailbreakMapping, c.Config.PromptGuard.PositiveLabels, result),
		PartialErr: lastErr,
	}, nil
}

// CheckForJailbreakRiskWithThreshold is CheckForJailbreakWithRisk against a
// caller-supplied threshold, for surfaces that carry their own (the
// response_jailbreak plugin thresholds per decision). Both share one scan so a
// caller cannot end up thresholding a different quantity than the routing
// signal does.
func (c *Classifier) CheckForJailbreakRiskWithThreshold(ctx context.Context, text string, threshold float32) (bool, string, float32, float32, error) {
	scan, err := c.ScanJailbreakRisk(ctx, text)
	if errors.Is(err, errNothingScored) {
		return false, "", 0.0, 0.0, nil
	}
	if err != nil {
		return false, "", 0.0, 0.0, err
	}

	isJailbreak := scan.RiskScore >= threshold
	if !isJailbreak && scan.PartialErr != nil {
		// Text that was only partly scored has not been found clean.
		return false, "", 0.0, 0.0, fmt.Errorf("jailbreak classification failed on part of the text: %w", scan.PartialErr)
	}

	if isJailbreak {
		logging.Warnf("JAILBREAK DETECTED: '%s' (confidence: %.3f, risk: %.3f, threshold: %.3f)",
			scan.Type, scan.Confidence, scan.RiskScore, threshold)
	}

	return isJailbreak, scan.Type, scan.Confidence, scan.RiskScore, nil
}
