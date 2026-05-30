package classification

import (
	"slices"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// cachedPIIResult stores a cached PII token classification result.
type cachedPIIResult struct {
	result candle_binding.TokenClassificationResult
	err    error
}

func (c *Classifier) evaluatePIISignal(results *SignalResults, mu *sync.Mutex, piiText string, nonUserMessages []string) {
	start := time.Now()

	// Step 1: Collect the union of unique content pieces across all PII rules.
	contentSeen := make(map[string]struct{})
	var uniqueContents []string
	if piiText != "" {
		contentSeen[piiText] = struct{}{}
		uniqueContents = append(uniqueContents, piiText)
	}
	for _, rule := range c.Config.PIIRules {
		if !rule.IncludeHistory {
			continue
		}
		for _, msg := range nonUserMessages {
			if msg == "" {
				continue
			}
			if _, ok := contentSeen[msg]; !ok {
				contentSeen[msg] = struct{}{}
				uniqueContents = append(uniqueContents, msg)
			}
		}
	}

	// Step 2: Run PII token classification exactly once per unique content piece.
	// Entity types are returned as "LABEL_{class_id}" and translated by PIIMapping.
	piiCache := make(map[string]cachedPIIResult, len(uniqueContents))
	for _, content := range uniqueContents {
		tokenResult, err := c.piiInference.ClassifyTokens(content)
		piiCache[content] = cachedPIIResult{tokenResult, err}
	}

	// Step 3: Evaluate each rule concurrently using the cached token results.
	// Each goroutine applies its own threshold and allow-list without re-running the model.
	var ruleWg sync.WaitGroup
	for _, rule := range c.Config.PIIRules {
		ruleWg.Add(1)
		go func() {
			defer ruleWg.Done()
			c.evaluatePIIRule(rule, piiText, nonUserMessages, piiCache, start, results, mu)
		}()
	}
	ruleWg.Wait()

	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()
	results.Metrics.PII.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if results.PIIDetected {
		results.Metrics.PII.Confidence = 1.0 // Binary: PII found or not
	}

	metrics.RecordSignalExtraction(config.SignalTypePII, "pii_evaluated", latencySeconds)
	logging.Debugf("[Signal Computation] PII signal evaluation completed in %v", elapsed)
}

func (c *Classifier) evaluatePIIRule(rule config.PIIRule, piiText string, nonUserMessages []string, piiCache map[string]cachedPIIResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	ruleContents := collectPIIRuleContents(piiText, nonUserMessages, rule.IncludeHistory)
	if len(ruleContents) == 0 {
		return
	}

	entityTypes := c.collectPIIEntityTypes(ruleContents, rule.Name, rule.Threshold, piiCache)
	deniedEntities := findDeniedEntities(entityTypes, rule.PIITypesAllowed)

	if len(deniedEntities) > 0 {
		metrics.RecordSignalExtraction(config.SignalTypePII, rule.Name, time.Since(start).Seconds())
		metrics.RecordSignalMatch(config.SignalTypePII, rule.Name)

		logging.Debugf("[Signal Computation] PII rule %q matched: denied_entities=%v", rule.Name, deniedEntities)

		mu.Lock()
		results.MatchedPIIRules = append(results.MatchedPIIRules, rule.Name)
		results.PIIDetected = true
		for _, e := range deniedEntities {
			if !slices.Contains(results.PIIEntities, e) {
				results.PIIEntities = append(results.PIIEntities, e)
			}
		}
		mu.Unlock()
	}
}
