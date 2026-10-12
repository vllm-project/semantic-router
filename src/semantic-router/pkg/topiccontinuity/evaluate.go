package topiccontinuity

import (
	"context"
	"math"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// HistoryLoader returns the original history and whether it is available.
// EvaluateAll calls it once, and only when it receives at least one rule.
type HistoryLoader func() ([]llmprotocol.Message, bool)

// RuleEvaluation is one rule's result with its content-free cost receipt.
type RuleEvaluation struct {
	Result Result
	// ClassifyMicros is this rule's Classify time only.
	ClassifyMicros int64
	// PrepareExtractMicros is the shared Prepare plus Extract time of the
	// rule's policy group, repeated on every rule in the group.
	PrepareExtractMicros int64
}

// Evaluation holds every rule's result in declaration order, plus one input
// size per preparation (one per distinct policy).
type Evaluation struct {
	Rules                 []RuleEvaluation
	PreparationInputBytes []int
}

type preparedGroup struct {
	prepared  preparation
	extracted extraction
	micros    int64
}

// EvaluateAll evaluates rules in declaration order over the original history.
// Rules sharing a HistoryPolicy share one preparation and extraction, so the
// number of rules never multiplies text work. A rule whose limits or
// thresholds are outside the documented ranges yields unknown_internal_error
// instead of being evaluated.
func EvaluateAll(ctx context.Context, load HistoryLoader, rules []EvalConfig) Evaluation {
	var evaluation Evaluation
	if len(rules) == 0 {
		return evaluation
	}
	messages, available := load()
	groups := make(map[HistoryPolicy]*preparedGroup, len(rules))
	evaluation.Rules = make([]RuleEvaluation, 0, len(rules))
	for _, rule := range rules {
		if !validEvalConfig(rule) {
			evaluation.Rules = append(evaluation.Rules, RuleEvaluation{
				Result: internalErrorResult(rule.Name, rule.Policy.IncludeAssistant),
			})
			continue
		}
		group, ok := groups[rule.Policy]
		if !ok {
			started := time.Now()
			prepared := prepare(ctx, messages, available, rule.Policy)
			group = &preparedGroup{prepared: prepared, extracted: extract(ctx, prepared)}
			group.micros = time.Since(started).Microseconds()
			groups[rule.Policy] = group
			evaluation.PreparationInputBytes = append(evaluation.PreparationInputBytes, prepared.InputBytes)
		}
		started := time.Now()
		result := classify(rule, group.prepared, group.extracted)
		evaluation.Rules = append(evaluation.Rules, RuleEvaluation{
			Result:               result,
			ClassifyMicros:       time.Since(started).Microseconds(),
			PrepareExtractMicros: group.micros,
		})
	}
	return evaluation
}

func validEvalConfig(rule EvalConfig) bool {
	limits := rule.Policy.Limits
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	return limits.MaxPriorTurns >= MinPriorTurns && limits.MaxPriorTurns <= MaxPriorTurns &&
		limits.MaxTurnBytes >= MinTurnBytes && limits.MaxTurnBytes <= MaxTurnBytes &&
		limits.MaxInputBytes >= MinInputBytes && limits.MaxInputBytes <= MaxInputBytes &&
		limits.MaxInputBytes >= limits.MaxTurnBytes &&
		finite(rule.Change) && finite(rule.Continuation) &&
		rule.Change >= 0 && rule.Change < rule.Continuation && rule.Continuation < 1
}
