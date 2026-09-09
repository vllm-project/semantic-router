package classification

import (
	"slices"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// cachedPIIResult stores a cached PII token classification result.
type cachedPIIResult struct {
	result candle_binding.TokenClassificationResult
	err    error
}

const (
	piiEvaluationIncompleteCode = "pii_evaluation_incomplete"
	piiEvaluationFailedCode     = "pii_evaluation_failed"
)

func (c *Classifier) evaluatePIISignal(results *SignalResults, mu *sync.Mutex, piiText string, nonUserMessages []string, toolResultTexts []string, toolResultScanIncomplete bool) {
	start := time.Now()

	// Step 1: Collect the union of unique content pieces selected by all PII
	// rules. A source-scoped rule controls which request content enters the
	// shared cache; this keeps tool-result scanning opt-in.
	contentSeen := make(map[string]struct{})
	var uniqueContents []string
	for _, rule := range c.Config.PIIRules {
		for _, content := range collectPIIRuleContentsForSource(rule, piiText, nonUserMessages, toolResultTexts) {
			if _, ok := contentSeen[content]; ok {
				continue
			}
			contentSeen[content] = struct{}{}
			uniqueContents = append(uniqueContents, content)
		}
	}

	// Step 2: Run PII token classification exactly once per unique content piece.
	// Entity types are returned as "LABEL_{class_id}" and translated by PIIMapping.
	piiCache := make(map[string][]cachedPIIResult, len(uniqueContents))
	for _, content := range uniqueContents {
		chunks := piiSignalChunks(content)
		cached := make([]cachedPIIResult, 0, len(chunks))
		for _, chunk := range chunks {
			tokenResult, err := c.piiInference.ClassifyTokens(chunk)
			cached = append(cached, cachedPIIResult{tokenResult, err})
		}
		piiCache[content] = cached
	}

	// Step 3: Evaluate each rule concurrently using the cached token results.
	// Each goroutine applies its own threshold and allow-list without re-running the model.
	var ruleWg sync.WaitGroup
	for _, rule := range c.Config.PIIRules {
		ruleWg.Add(1)
		go func() {
			defer ruleWg.Done()
			c.evaluatePIIRule(rule, piiText, nonUserMessages, toolResultTexts, toolResultScanIncomplete, piiCache, start, results, mu)
		}()
	}
	ruleWg.Wait()

	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()
	results.Metrics.PII.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if results.PIIDetected {
		results.Metrics.PII.Confidence = 1.0 // Binary: PII found or not
	}

	c.recordSignalExtraction(config.SignalTypePII, "pii_evaluated", latencySeconds)
	logging.Debugf("[Signal Computation] PII signal evaluation completed in %v", elapsed)
}

func (c *Classifier) evaluatePIIRule(rule config.PIIRule, piiText string, nonUserMessages []string, toolResultTexts []string, toolResultScanIncomplete bool, piiCache map[string][]cachedPIIResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	ruleContents := collectPIIRuleContentsForSource(rule, piiText, nonUserMessages, toolResultTexts)
	if len(ruleContents) == 0 {
		if rule.Source == config.PIISourceToolResult && toolResultScanIncomplete {
			c.recordPIIRuleError(rule, piiScanIncomplete, results, mu)
		}
		return
	}

	entityTypes, status := c.collectPIIEntityTypes(ruleContents, rule.Name, rule.Threshold, piiCache)
	if rule.Source == config.PIISourceToolResult && toolResultScanIncomplete && status == piiScanClean {
		status = piiScanIncomplete
	}
	if status != piiScanClean {
		c.recordPIIRuleError(rule, status, results, mu)
	}
	deniedEntities := findDeniedEntities(entityTypes, rule.PIITypesAllowed)

	if len(deniedEntities) > 0 {
		c.recordSignalExtraction(config.SignalTypePII, rule.Name, time.Since(start).Seconds())
		c.recordSignalMatch(config.SignalTypePII, rule.Name)

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

type piiScanStatus string

const (
	piiScanClean      piiScanStatus = "clean"
	piiScanIncomplete piiScanStatus = "incomplete"
	piiScanFailed     piiScanStatus = "error"
)

func (c *Classifier) recordPIIRuleError(rule config.PIIRule, status piiScanStatus, results *SignalResults, mu *sync.Mutex) {
	code := piiEvaluationFailedCode
	if status == piiScanIncomplete {
		code = piiEvaluationIncompleteCode
	}

	mu.Lock()
	defer mu.Unlock()
	if results.SignalErrors == nil {
		results.SignalErrors = make(map[string]string)
	}
	results.SignalErrors[signalConfidenceKey(config.SignalTypePII, rule.Name)] = code
}
