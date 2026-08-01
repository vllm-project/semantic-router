package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *Classifier) evaluateStructureSignal(results *SignalResults, mu *sync.Mutex, text string, uncompressedText ...string) {
	start := time.Now()
	evaluatedRules, err := c.structureClassifier.EvaluateAll(text, uncompressedText...)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	results.Metrics.Structure.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0

	logging.Debugf("[Signal Computation] Structure signal evaluation completed in %v", elapsed)
	if err != nil {
		logging.Errorf("structure rule evaluation failed: %v", err)
		return
	}
	if len(evaluatedRules) == 0 {
		return
	}

	bestConfidence := 0.0
	mu.Lock()
	defer mu.Unlock()
	for _, match := range evaluatedRules {
		key := signalConfidenceKey(config.SignalTypeStructure, match.RuleName)
		results.SignalValues[key] = match.Value
		results.SignalConfidences[key] = match.Confidence
		c.recordSignalExtraction(config.SignalTypeStructure, match.RuleName, latencySeconds)
		if !match.Matched {
			continue
		}
		if match.Confidence > bestConfidence {
			bestConfidence = match.Confidence
		}
		c.recordSignalMatch(config.SignalTypeStructure, match.RuleName)
		results.MatchedStructureRules = append(results.MatchedStructureRules, match.RuleName)
	}
	results.Metrics.Structure.Confidence = bestConfidence
}
