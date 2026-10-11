package classification

import (
	"slices"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// evaluateActionSignal publishes the phrase share as a signal value rather than
// a confidence, because a lexical match is not a model probability.
func (c *Classifier) evaluateActionSignal(results *SignalResults, mu *sync.Mutex, currentUserText string) {
	start := time.Now()
	action := ClassifyAction(currentUserText)
	elapsed := time.Since(start)
	c.recordSignalExtraction(config.SignalTypeAction, action.Action, elapsed.Seconds())

	mu.Lock()
	defer mu.Unlock()
	available := false
	results.Metrics.Action.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Action.Method = "lexical"
	results.Metrics.Action.ConfidenceAvailable = &available
	if !slices.ContainsFunc(c.Config.ActionRules, func(rule config.ActionRule) bool { return rule.Name == action.Action }) {
		return
	}
	c.recordSignalMatch(config.SignalTypeAction, action.Action)
	results.MatchedActionRules = append(results.MatchedActionRules, action.Action)
	results.SignalValues[signalConfidenceKey(config.SignalTypeAction, action.Action)] = action.Score
}
