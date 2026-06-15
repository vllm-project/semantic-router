package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateEventSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	matches := c.eventClassifier.Classify(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	results.Metrics.Event.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

	logging.Debugf("[Signal Computation] Event signal evaluation completed in %v", elapsed)

	bestConfidence := 0.0
	mu.Lock()
	for _, match := range matches {
		metrics.RecordSignalExtraction(config.SignalTypeEvent, match.RuleName, latencySeconds)
		metrics.RecordSignalMatch(config.SignalTypeEvent, match.RuleName)
		results.MatchedEventRules = append(results.MatchedEventRules, match.RuleName)
		results.SignalConfidences["event:"+match.RuleName] = match.Confidence
		if match.MatchedSeverity != "" {
			results.SignalValues["event:"+match.RuleName+":severity"] = 1.0
		}
		if match.TemporalMatch {
			results.SignalValues["event:"+match.RuleName+":temporal"] = 1.0
		}
		if match.Confidence > bestConfidence {
			bestConfidence = match.Confidence
		}
	}
	results.Metrics.Event.Confidence = bestConfidence
	mu.Unlock()
}
