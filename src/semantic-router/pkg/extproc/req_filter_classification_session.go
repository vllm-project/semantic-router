package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *OpenAIRouter) applyRuntimeSessionSignals(ctx *RequestContext, signals *classification.SignalResults) {
	if r == nil || ctx == nil || signals == nil || len(r.Config.SessionRules) == 0 {
		return
	}
	if signals.SignalValues == nil {
		signals.SignalValues = make(map[string]float64)
	}
	if signals.SignalConfidences == nil {
		signals.SignalConfidences = make(map[string]float64)
	}

	taskFamily := ""
	if len(signals.MatchedDomainRules) > 0 {
		taskFamily = strings.TrimSpace(signals.MatchedDomainRules[0])
	}

	for _, rule := range r.Config.SessionRules {
		value := r.resolveSessionRuleValue(rule, ctx, taskFamily)
		key := config.SignalTypeSession + ":" + rule.Name
		signals.SignalValues[key] = value
		signals.SignalConfidences[key] = 1.0
		if numericPredicateMatches(rule.Predicate, value) {
			signals.MatchedSessionRules = append(signals.MatchedSessionRules, rule.Name)
		}
	}
}

func (r *OpenAIRouter) resolveSessionRuleValue(rule config.SessionRule, ctx *RequestContext, taskFamily string) float64 {
	if ctx == nil {
		return 0
	}
	currentModel := strings.TrimSpace(ctx.PreviousModel)
	candidateModel := strings.TrimSpace(rule.CandidateModel)
	resolvedTaskFamily := strings.TrimSpace(rule.IntentOrDomain)
	if resolvedTaskFamily == "" {
		resolvedTaskFamily = strings.TrimSpace(taskFamily)
	}
	if currentModel == "" {
		currentModel = strings.TrimSpace(rule.PreviousModel)
	}

	switch config.NormalizeSessionFact(rule.Fact) {
	case config.SessionFactSessionPresent:
		return boolToFloat(ctx.SessionID != "")
	case config.SessionFactHasPreviousModel:
		if strings.TrimSpace(rule.PreviousModel) != "" {
			return boolToFloat(strings.EqualFold(strings.TrimSpace(ctx.PreviousModel), strings.TrimSpace(rule.PreviousModel)))
		}
		return boolToFloat(strings.TrimSpace(ctx.PreviousModel) != "")
	case config.SessionFactTurnIndex:
		return float64(ctx.TurnIndex)
	case config.SessionFactCacheWarmth:
		return ctx.CacheWarmthEstimate
	case config.SessionFactRemainingTurns:
		return r.lookupRemainingTurns(resolvedTaskFamily, ctx.TurnIndex)
	case config.SessionFactHandoffPenalty:
		return r.lookupHandoffPenalty(currentModel, candidateModel)
	case config.SessionFactQualityGap:
		return r.lookupQualityGap(resolvedTaskFamily, currentModel, candidateModel)
	default:
		return 0
	}
}

func numericPredicateMatches(predicate *config.NumericPredicate, value float64) bool {
	if predicate == nil {
		return false
	}
	if predicate.GT != nil && !(value > *predicate.GT) {
		return false
	}
	if predicate.GTE != nil && !(value >= *predicate.GTE) {
		return false
	}
	if predicate.LT != nil && !(value < *predicate.LT) {
		return false
	}
	if predicate.LTE != nil && !(value <= *predicate.LTE) {
		return false
	}
	return true
}

func boolToFloat(value bool) float64 {
	if value {
		return 1
	}
	return 0
}
