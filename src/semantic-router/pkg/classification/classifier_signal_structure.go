package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func (c *Classifier) evaluateStructureSignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	matchedRules, err := c.structureClassifier.Classify(text)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	results.Metrics.Structure.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

	logging.Debugf("[Signal Computation] Structure signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("structure rule evaluation failed: %v", err)
		return
	}
	if len(matchedRules) == 0 {
		return
	}

	bestConfidence := 0.0
	mu.Lock()
	defer mu.Unlock()
	for _, match := range matchedRules {
		if match.Confidence > bestConfidence {
			bestConfidence = match.Confidence
		}
		metrics.RecordSignalExtraction(config.SignalTypeStructure, match.RuleName, latencySeconds)
		metrics.RecordSignalMatch(config.SignalTypeStructure, match.RuleName)
		results.MatchedStructureRules = append(results.MatchedStructureRules, match.RuleName)
		results.SignalConfidences[signalConfidenceKey(config.SignalTypeStructure, match.RuleName)] = match.Confidence
		results.SignalValues[signalConfidenceKey(config.SignalTypeStructure, match.RuleName)] = match.Value
	}
	results.Metrics.Structure.Confidence = bestConfidence
}
