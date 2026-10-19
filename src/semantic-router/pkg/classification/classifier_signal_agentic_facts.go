package classification

import (
	"slices"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateAgenticFactsSignal(
	results *SignalResults,
	mu *sync.Mutex,
	facts RequestFacts,
	usedSignals map[string]bool,
) {
	start := time.Now()
	bestConfidence := 0.0
	for _, rule := range c.Config.AgenticFactsRules {
		if !signalRuleUsed(usedSignals, config.SignalTypeAgenticFacts, rule.Name) {
			continue
		}
		if !agenticFactsRuleMatches(rule, facts) {
			continue
		}
		mu.Lock()
		results.MatchedAgenticFactsRules = append(results.MatchedAgenticFactsRules, rule.Name)
		results.SignalConfidences[signalConfidenceKey(config.SignalTypeAgenticFacts, rule.Name)] = 1.0
		mu.Unlock()
		bestConfidence = 1.0
		metrics.RecordSignalMatch(config.SignalTypeAgenticFacts, rule.Name)
	}
	elapsed := time.Since(start)
	results.Metrics.AgenticFacts.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.AgenticFacts.Confidence = bestConfidence
}

func agenticFactsRuleMatches(rule config.AgenticFactsRule, facts RequestFacts) bool {
	var value string
	switch rule.Field {
	case config.AgenticFactsFieldDelegatedRole:
		value = facts.AgenticFactsDelegatedRole
	case config.AgenticFactsFieldTaskPhase:
		value = facts.AgenticFactsTaskPhase
	default:
		return false
	}
	if value == "" {
		return false
	}
	switch {
	case rule.Predicate.Equals != nil:
		return value == *rule.Predicate.Equals
	case len(rule.Predicate.In) > 0:
		return slices.Contains(rule.Predicate.In, value)
	default:
		return false
	}
}
