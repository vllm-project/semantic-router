package classification

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *Classifier) evaluateModalitySignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, text string) {
	start := time.Now()
	modalityResult := c.classifyModalityWithContext(ctx, text, &c.Config.ModalityDetector.ModalityDetectionConfig)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	signalName := modalityResult.Modality

	// Record signal extraction metrics
	c.recordSignalExtraction(config.SignalTypeModality, signalName, latencySeconds)

	// Record metrics
	results.Metrics.Modality.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	available := modalityResult.ConfidenceAvailable
	results.Metrics.Modality.ConfidenceAvailable = &available
	results.Metrics.Modality.Method = modalityResult.Method
	if available {
		results.Metrics.Modality.Confidence = float64(modalityResult.Confidence)
	}

	logging.Debugf("[Signal Computation] Modality signal evaluation completed in %v: %s (confidence_available=%v, method=%s)",
		elapsed, signalName, modalityResult.ConfidenceAvailable, modalityResult.Method)

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
