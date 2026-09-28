package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ComplexityVerdictLabels is the label order a label_distribution.v1 backend
// must return for the complexity signal. The verdict vocabulary is fixed, so
// the order is declared here rather than configured, and it is the contract
// the remote model's output positions are read against.
var ComplexityVerdictLabels = []string{
	config.ComplexityDifficultyHard,
	config.ComplexityDifficultyEasy,
	config.ComplexityDifficultyMedium,
}

// evaluateComplexityScore turns one remote score into a per-rule verdict.
//
// The score is a property of the request, not of a rule, so a single call
// serves every rule and each rule differs only in where it draws its
// boundaries. That is also why the backend is declared on the module: one
// remote model, many interpretations.
func evaluateComplexityScore(
	ctx context.Context,
	backend ScoringBackend,
	text string,
	rules []config.ComplexityRule,
) ([]ComplexityRuleResult, error) {
	// Resolve boundaries before spending a request: a malformed pair is a
	// config error, and reporting it after the call would blame the backend.
	boundaries := make([]config.ComplexityBoundaries, len(rules))
	for i, rule := range rules {
		bounds, err := rule.EffectiveBoundaries()
		if err != nil {
			return nil, err
		}
		boundaries[i] = bounds
	}

	score, err := backend.Score(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("complexity score backend failed: %w", err)
	}

	results := make([]ComplexityRuleResult, 0, len(rules))
	for i, rule := range rules {
		results = append(results, ComplexityRuleResult{
			RuleName:   rule.Name,
			Difficulty: boundaries[i].Verdict(score),
			// Published in the model's own units. Rescaling onto the local
			// margin would imply the two are comparable when they are not.
			FusedMargin:  score,
			SignalSource: complexitySignalSourceScore,
			// score.v1 reports no confidence: a score just short of a
			// boundary is the least certain position, not a strong one, so
			// any number derived from it would be invented. The decision
			// engine already handles an absent confidence by falling back to
			// its structural default and marking the pool unscored.
			ConfidenceReported: false,
		})
	}
	return results, nil
}

// evaluateComplexityLabels reads a verdict straight off a label distribution.
//
// No boundaries are consulted on this contract: the winning label *is* the
// verdict, and its probability is a genuine confidence rather than a derived
// one. Rules still matter, because each carries its own composer, so the same
// verdict can be gated differently per rule.
func evaluateComplexityLabels(
	ctx context.Context,
	backend SequenceClassifierBackend,
	text string,
	rules []config.ComplexityRule,
) ([]ComplexityRuleResult, error) {
	result, err := backend.Classify(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("complexity label backend failed: %w", err)
	}
	if len(result.Probabilities) != len(ComplexityVerdictLabels) {
		return nil, fmt.Errorf(
			"complexity label backend returned %d scores for %d verdict labels (%v)",
			len(result.Probabilities), len(ComplexityVerdictLabels), ComplexityVerdictLabels)
	}

	index, confidence := deriveArgmax(result.Probabilities)
	if index < 0 || index >= len(ComplexityVerdictLabels) {
		return nil, fmt.Errorf("complexity label backend returned an unusable distribution")
	}
	verdict := ComplexityVerdictLabels[index]

	results := make([]ComplexityRuleResult, 0, len(rules))
	for _, rule := range rules {
		results = append(results, ComplexityRuleResult{
			RuleName:           rule.Name,
			Difficulty:         verdict,
			Confidence:         float64(confidence),
			SignalSource:       complexitySignalSourceLabels,
			ConfidenceReported: true,
		})
	}
	return results, nil
}

const (
	complexitySignalSourceScore  = "remote_score"
	complexitySignalSourceLabels = "remote_labels"
)
