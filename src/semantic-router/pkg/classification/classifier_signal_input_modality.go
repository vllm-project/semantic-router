package classification

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// evaluateInputModalitySignal matches input_modality rules against the
// structural modality-presence counts extracted from the parsed request.
// Purely deterministic: a rule matches when at least one content part of its
// declared modality is present. Every in-scope rule publishes its count as a
// signal value, including zero, so traces show the full modality set.
func (c *Classifier) evaluateInputModalitySignal(
	results *SignalResults,
	mu *sync.Mutex,
	facts RequestFacts,
	usedSignals map[string]bool,
) {
	start := time.Now()
	bestConfidence := 0.0
	var matched []string
	values := make(map[string]float64, len(c.Config.InputModalityRules))
	for _, rule := range c.Config.InputModalityRules {
		if !signalRuleUsed(usedSignals, config.SignalTypeInputModality, rule.Name) {
			continue
		}
		value := float64(inputModalityCount(rule.Modality, facts.InputModality))
		values[signalConfidenceKey(config.SignalTypeInputModality, rule.Name)] = value
		if value > 0 {
			matched = append(matched, rule.Name)
			bestConfidence = 1.0
			c.recordSignalMatch(config.SignalTypeInputModality, rule.Name)
		}
	}
	mu.Lock()
	for key, value := range values {
		results.SignalValues[key] = value
		if value > 0 {
			results.SignalConfidences[key] = 1.0
		} else {
			results.SignalConfidences[key] = 0
		}
	}
	results.MatchedInputModalityRules = append(results.MatchedInputModalityRules, matched...)
	mu.Unlock()
	elapsed := time.Since(start)
	results.Metrics.InputModality.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.InputModality.Confidence = bestConfidence
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
