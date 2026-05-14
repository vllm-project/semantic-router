package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateEventContextSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	matches := c.eventContextClassifier.Classify(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	results.Metrics.EventContext.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

	logging.Debugf("[Signal Computation] EventContext signal evaluation completed in %v", elapsed)

	bestConfidence := 0.0
	mu.Lock()
	for _, match := range matches {
		metrics.RecordSignalExtraction(config.SignalTypeEventContext, match.RuleName, latencySeconds)
		metrics.RecordSignalMatch(config.SignalTypeEventContext, match.RuleName)
		results.MatchedEventContextRules = append(results.MatchedEventContextRules, match.RuleName)
		results.SignalConfidences["event_context:"+match.RuleName] = match.Confidence
		if match.MatchedSeverity != "" {
			results.SignalValues["event_context:"+match.RuleName+":severity"] = 1.0
		}
		if match.TemporalMatch {
			results.SignalValues["event_context:"+match.RuleName+":temporal"] = 1.0
		}
		if match.Confidence > bestConfidence {
			bestConfidence = match.Confidence
		}
	}
	results.Metrics.EventContext.Confidence = bestConfidence
	mu.Unlock()
}
