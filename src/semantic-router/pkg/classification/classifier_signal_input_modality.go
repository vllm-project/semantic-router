package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// evaluateInputModalitySignal matches input_modality rules against the
// structural modality-presence counts extracted from the parsed request.
// Purely deterministic: a rule matches when at least one content part of its
// declared modality is present.
func (c *Classifier) evaluateInputModalitySignal(
	results *SignalResults,
	mu *sync.Mutex,
	facts RequestFacts,
	usedSignals map[string]bool,
) {
	rules := c.Config.InputModalityRules
	if len(rules) == 0 {
		return
	}

	start := time.Now()
	matchedAny := false

	for _, rule := range rules {
		if !signalRuleUsed(usedSignals, config.SignalTypeInputModality, rule.Name) {
			continue
		}
		ruleStart := time.Now()
		value := float64(inputModalityCount(rule.Modality, facts.InputModality))
		matched := value > 0
		elapsed := time.Since(ruleStart)
		mu.Lock()
		key := signalConfidenceKey(config.SignalTypeInputModality, rule.Name)
		results.SignalValues[key] = value
		if matched {
			matchedAny = true
			results.SignalConfidences[key] = 1.0
			results.MatchedInputModalityRules = append(results.MatchedInputModalityRules, rule.Name)
		} else {
			results.SignalConfidences[key] = 0
		}
		mu.Unlock()

		c.recordSignalExtraction(config.SignalTypeInputModality, rule.Name, elapsed.Seconds())
		if matched {
			c.recordSignalMatch(config.SignalTypeInputModality, rule.Name)
		}
	}

	elapsed := time.Since(start)
	results.Metrics.InputModality.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if matchedAny {
		results.Metrics.InputModality.Confidence = 1.0
	} else {
		results.Metrics.InputModality.Confidence = 0
	}
	logging.Debugf("[Signal Computation] Input-modality signal evaluation completed in %v", elapsed)
}

func inputModalityCount(modality string, facts InputModalityFacts) int {
	switch modality {
	case config.InputModalityText:
		return facts.TextContentCount
	case config.InputModalityImage:
		return facts.ImageContentCount
	case config.InputModalityAudio:
		return facts.AudioContentCount
	case config.InputModalityVideo:
		return facts.VideoContentCount
	default:
		return 0
	}
}
