package classification

import (
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// RequestFacts carries untrusted request-envelope facts used by signal
// extraction. Trusted identity remains owned by the authz header path.
type RequestFacts struct {
	Metadata map[string]string
}

func (c *Classifier) evaluateMetadataSignal(
	results *SignalResults,
	mu *sync.Mutex,
	facts RequestFacts,
	usedSignals map[string]bool,
) {
	start := time.Now()
	bestConfidence := 0.0
	for _, rule := range c.Config.MetadataRules {
		if !signalRuleUsed(usedSignals, config.SignalTypeMetadata, rule.Name) {
			continue
		}
		if !metadataRuleMatches(rule, facts.Metadata) {
			continue
		}
		mu.Lock()
		results.MatchedMetadataRules = append(results.MatchedMetadataRules, rule.Name)
		results.SignalConfidences[signalConfidenceKey(config.SignalTypeMetadata, rule.Name)] = 1.0
		mu.Unlock()
		bestConfidence = 1.0
		metrics.RecordSignalMatch(config.SignalTypeMetadata, rule.Name)
	}
	elapsed := time.Since(start)
	results.Metrics.Metadata.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Metadata.Confidence = bestConfidence
}

func signalRuleUsed(usedSignals map[string]bool, signalType string, name string) bool {
	return usedSignals[strings.ToLower(signalConfidenceKey(signalType, name))]
}

func metadataRuleMatches(rule config.MetadataRule, metadata map[string]string) bool {
	value, exists := metadata[rule.Key]
	switch {
	case rule.Predicate.Equals != nil:
		return exists && value == *rule.Predicate.Equals
	case len(rule.Predicate.In) > 0:
		return exists && slices.Contains(rule.Predicate.In, value)
	case rule.Predicate.Exists != nil:
		return exists == *rule.Predicate.Exists
	default:
		return false
	}
}
