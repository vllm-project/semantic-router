package classification

import (
	"context"
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

// cachedPIIContent stores all inference results that were available for one
// content item. incomplete is distinct from an inference error: it means the
// request-scoped scan budget stopped before the content was fully inspected.
type cachedPIIContent struct {
	results    []cachedPIIResult
	incomplete bool
}

const (
	piiEvaluationIncompleteCode = "pii_evaluation_incomplete"
	// A tool-result request can contain many independently chunked content
	// items. Bound the expensive detector fanout per request while preserving
	// the already-scanned results for conservative on_unknown handling.
	maxPIIToolResultInferenceCalls = 1024
)

type piiToolResultScanBudget struct {
	remainingInferenceCalls int
}

func (b *piiToolResultScanBudget) consumeInferenceCall() bool {
	if b.remainingInferenceCalls <= 0 {
		return false
	}
	b.remainingInferenceCalls--
	return true
}

func (c *Classifier) evaluatePIISignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, piiText string, nonUserMessages []string, toolResultTexts []string, toolResultScanIncomplete bool) {
	start := time.Now()

	// Step 1: Collect the union of unique content pieces selected by all PII
	// rules. A source-scoped rule controls which request content enters the
	// shared cache; this keeps tool-result scanning opt-in.
	contentSeen := make(map[string]struct{})
	toolResultContentSeen := make(map[string]struct{})
	var uniqueContents []string
	for _, rule := range c.Config.PIIRules {
		for _, content := range collectPIIRuleContentsForSource(rule, piiText, nonUserMessages, toolResultTexts) {
			if rule.Source == config.PIISourceToolResult {
				toolResultContentSeen[content] = struct{}{}
			}
			if _, ok := contentSeen[content]; ok {
				continue
			}
			contentSeen[content] = struct{}{}
			uniqueContents = append(uniqueContents, content)
		}
	}

	// Step 2: Run PII token classification exactly once per unique content piece.
	// Entity types are returned as "LABEL_{class_id}" and translated by
	// PIIMapping. Tool-result content uses a request-scoped inference budget;
	// legacy prompt/history content retains its existing behavior.
	piiCache := make(map[string]cachedPIIContent, len(uniqueContents))
	toolResultBudget := piiToolResultScanBudget{remainingInferenceCalls: maxPIIToolResultInferenceCalls}
	for _, content := range uniqueContents {
		cached := cachedPIIContent{}
		_, isToolResult := toolResultContentSeen[content]
		if isToolResult {
			// Tool results can be much larger than the request text and can
			// contain thousands of blocks. Stream chunks directly into the
			// bounded inference loop so the request does not first materialize
			// every chunk in memory.
			fullyScanned := forEachUniquePIISignalChunk(content, func(chunk string) bool {
				if !toolResultBudget.consumeInferenceCall() {
					return false
				}
				tokenResult, err := c.piiInference.ClassifyTokens(ctx, chunk)
				cached.results = append(cached.results, cachedPIIResult{tokenResult, err})
				return true
			})
			cached.incomplete = !fullyScanned
		} else {
			chunks := piiSignalChunks(content)
			cached.results = make([]cachedPIIResult, 0, len(chunks))
			for _, chunk := range chunks {
				tokenResult, err := c.piiInference.ClassifyTokens(ctx, chunk)
				cached.results = append(cached.results, cachedPIIResult{tokenResult, err})
			}
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

func (c *Classifier) evaluatePIIRule(rule config.PIIRule, piiText string, nonUserMessages []string, toolResultTexts []string, toolResultScanIncomplete bool, piiCache map[string]cachedPIIContent, start time.Time, results *SignalResults, mu *sync.Mutex) {
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
