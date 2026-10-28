package classification

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// JailbreakVerdict is the policy result with only model-reported probability
// fields populated. Categorical decisions are evaluated without a threshold.
type JailbreakVerdict struct {
	Detected   bool                 `json:"detected"`
	Label      string               `json:"label"`
	Confidence *float32             `json:"confidence,omitempty"`
	RiskScore  *float32             `json:"risk_score,omitempty"`
	Decision   *tasks.LabelDecision `json:"decision,omitempty"`
}

// ScanJailbreak preserves either a full probability scan or a categorical
// verdict. It retains partial failures so an incomplete safe scan is unresolved.
func (c *Classifier) ScanJailbreak(ctx context.Context, text string) (JailbreakScan, error) {
	backend := jailbreakDecisionBackend(c.jailbreakInference)
	if backend == nil {
		return c.ScanJailbreakRisk(ctx, text)
	}
	if !c.IsJailbreakEnabled() {
		return JailbreakScan{}, fmt.Errorf("jailbreak detection is not enabled or properly configured")
	}
	var scan JailbreakScan
	for _, chunk := range jailbreakSignalChunks(text) {
		decision, err := backend.Decide(ctx, chunk)
		if err != nil {
			scan.PartialErr = err
			continue
		}
		if _, ok := c.JailbreakMapping.GetIndexForJailbreakType(decision.Label); !ok {
			scan.PartialErr = fmt.Errorf("unknown jailbreak label %q", decision.Label)
			continue
		}
		positive := isPositiveJailbreakLabel(c.Config.PromptGuard.PositiveLabels, decision.Label)
		if scan.Decision == nil || positive {
			scan.Type = decision.Label
			scan.Decision = &decision
			scan.CategoricalMatch = positive
		}
	}
	if scan.Decision == nil {
		if scan.PartialErr != nil {
			return scan, scan.PartialErr
		}
		return scan, errNothingScored
	}
	return scan, nil
}

func (s JailbreakScan) Matches(threshold float32) bool {
	if s.Decision != nil {
		return s.CategoricalMatch
	}
	return s.RiskScore >= threshold
}

func (c *Classifier) CheckForJailbreakVerdict(ctx context.Context, text string, threshold float32) (JailbreakVerdict, error) {
	scan, err := c.ScanJailbreak(ctx, text)
	if errors.Is(err, errNothingScored) {
		return JailbreakVerdict{}, nil
	}
	if err != nil {
		return JailbreakVerdict{}, err
	}
	matched := scan.Matches(threshold)
	if !matched && scan.PartialErr != nil {
		return JailbreakVerdict{}, fmt.Errorf("jailbreak classification failed on part of the text: %w", scan.PartialErr)
	}
	verdict := JailbreakVerdict{Detected: matched, Label: scan.Type, Decision: scan.Decision}
	if scan.Decision == nil {
		verdict.Confidence = &scan.Confidence
		verdict.RiskScore = &scan.RiskScore
	}
	return verdict, nil
}

// The historical content API uses winning-label confidence for sequence
// classifiers. Keep that policy while allowing score-free categorical guards.
func (c *Classifier) contentJailbreakVerdict(ctx context.Context, text string, threshold float32) (JailbreakVerdict, error) {
	if jailbreakDecisionBackend(c.jailbreakInference) != nil {
		return c.CheckForJailbreakVerdict(ctx, text, threshold)
	}
	matched, label, confidence, err := c.CheckForJailbreakWithThreshold(ctx, text, threshold)
	if err != nil {
		return JailbreakVerdict{}, err
	}
	return JailbreakVerdict{Detected: matched, Label: label, Confidence: &confidence}, nil
}
