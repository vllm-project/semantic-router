package classification

import (
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *Classifier) evaluateModalitySignal(results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	modalityResult := c.classifyModality(text, &c.Config.ModalityDetector.ModalityDetectionConfig)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	signalName := modalityResult.Modality

	// Record signal extraction metrics
	c.recordSignalExtraction(config.SignalTypeModality, signalName, latencySeconds)

	// Record metrics
	results.Metrics.Modality.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Modality.Confidence = float64(modalityResult.Confidence)

	logging.Debugf("[Signal Computation] Modality signal evaluation completed in %v: %s (confidence=%.3f, method=%s)",
		elapsed, signalName, modalityResult.Confidence, modalityResult.Method)

	// Check if this signal name is defined in modality_rules
	for _, rule := range c.Config.ModalityRules {
		if strings.EqualFold(rule.Name, signalName) {
			c.recordSignalMatch(config.SignalTypeModality, rule.Name)
			mu.Lock()
			results.MatchedModalityRules = append(results.MatchedModalityRules, rule.Name)
			mu.Unlock()
			break
		}
	}
}
