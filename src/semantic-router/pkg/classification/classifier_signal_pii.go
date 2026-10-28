package classification

import (
	"context"
	"slices"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// PIIClassificationErrorType is the entity type a PII rule reports when its
// content could not be fully classified and on_error is block. It mirrors the
// fail-closed behavior of the jailbreak classifier.
const PIIClassificationErrorType = "classification_error"

// PIIUnscannedType is the entity type a PII rule reports when its model did
// not read all of the content (JailbreakUnscannedType): unless on_unscanned is
// allow, such content matches, so a long request routes as private.
const PIIUnscannedType = "unscanned"

// cachedPIIResult stores a cached PII token classification result.
type cachedPIIResult struct {
	result tasks.TokenClassificationResult
	err    error
}

// cachedPIIContent stores all inference results that were available for one
// content item. incomplete is distinct from an inference error: it means the
// request-scoped scan budget stopped before the content was fully inspected.
type cachedPIIContent struct {
	results    []cachedPIIResult
	incomplete bool
}

// piiCacheKey keeps detector results isolated by the rule source. The same
// bytes can be selected by a tool-result rule and a legacy prompt/history rule,
// but those scans have different completeness guarantees: tool results are
// request-budgeted while legacy content is not.
type piiCacheKey struct {
	source  string
	content string
}

func piiCacheSource(source string) string {
	if source == config.PIISourceToolResult {
		return config.PIISourceToolResult
	}
	return "legacy"
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

func (c *Classifier) evaluatePIISignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, piiText string, nonUserMessages []string) {
	c.evaluatePIISignalWithToolResults(ctx, results, mu, piiText, nonUserMessages, nil, false)
}

func (c *Classifier) evaluatePIISignalWithToolResults(ctx context.Context, results *SignalResults, mu *sync.Mutex, piiText string, nonUserMessages []string, toolResultTexts []string, toolResultScanIncomplete bool) {
	if backend := decisionPIIInference(c.piiInference); backend != nil {
		c.evaluateDecisionPIISignalWithToolResults(ctx, results, mu, piiText, nonUserMessages, toolResultTexts, toolResultScanIncomplete, backend)
		return
	}
	start := time.Now()

	// Step 1: Collect the union of unique content pieces selected by all PII
	// rules. A source-scoped rule controls which request content enters the
	// shared cache; this keeps tool-result scanning opt-in.
	contentSeen := make(map[piiCacheKey]struct{})
	var uniqueContents []piiCacheKey
	for _, rule := range c.Config.PIIRules {
		for _, content := range collectPIIRuleContentsForSource(rule, piiText, nonUserMessages, toolResultTexts) {
			key := piiCacheKey{source: piiCacheSource(rule.Source), content: content}
			if _, ok := contentSeen[key]; ok {
				continue
			}
			contentSeen[key] = struct{}{}
			uniqueContents = append(uniqueContents, key)
		}
	}

	// Bound tool-result work before dispatching it through the model-service bundle.
	piiCache := make(map[piiCacheKey]cachedPIIContent, len(uniqueContents))
	type piece struct {
		key   piiCacheKey
		chunk string
	}
	var pieces []piece
	budget := piiToolResultScanBudget{remainingInferenceCalls: maxPIIToolResultInferenceCalls}
	for _, key := range uniqueContents {
		cached := cachedPIIContent{}
		add := func(chunk string) bool {
			if key.source == config.PIISourceToolResult && !budget.consumeInferenceCall() {
				return false
			}
			pieces = append(pieces, piece{key, chunk})
			return true
		}
		if key.source == config.PIISourceToolResult && c.Config.PIIModel.Window == nil && !c.hasLongContextClassifier(config.SignalTypePII) {
			cached.incomplete = !forEachUniquePIISignalChunk(key.content, add)
		} else {
			for _, chunk := range c.piiInputs(key.content) {
				if !add(chunk) {
					cached.incomplete = true
					break
				}
			}
		}
		piiCache[key] = cached
	}
	classified := make([]cachedPIIResult, len(pieces))
	modelservice.Fan(ctx, len(pieces), func(i int) {
		classified[i].result, classified[i].err = c.classifyPIITokens(ctx, pieces[i].chunk)
		classified[i].err = signalDeadline(ctx, classified[i].err)
	})
	for i, piece := range pieces {
		cached := piiCache[piece.key]
		cached.results = append(cached.results, classified[i])
		piiCache[piece.key] = cached
	}
	rules := c.Config.PIIRules
	modelservice.Fan(ctx, len(rules), func(i int) {
		c.evaluatePIIRule(rules[i], piiText, nonUserMessages, toolResultTexts, toolResultScanIncomplete, piiCache, start, results, mu)
	})

	// Replay evidence must cover complete, clean content regardless of routing allow-lists.
	verified := len(rules) > 0 && len(pieces) > 0 && !toolResultScanIncomplete
	for _, content := range append(append([]string{piiText}, nonUserMessages...), toolResultTexts...) {
		if content != "" {
			_, legacy := contentSeen[piiCacheKey{source: "legacy", content: content}]
			_, tool := contentSeen[piiCacheKey{source: config.PIISourceToolResult, content: content}]
			if !legacy && !tool {
				verified = false
			}
		}
	}
	for _, key := range uniqueContents {
		complete, clean := !piiCache[key].incomplete, true
		if key.source == config.PIISourceToolResult && toolResultScanIncomplete {
			complete = false
		}
		for _, cached := range piiCache[key].results {
			if cached.err != nil {
				complete = false
			}
		}
		for _, rule := range rules {
			if piiCacheSource(rule.Source) != key.source {
				continue
			}
			entities, status, _ := c.collectPIIEntityTypes([]string{key.content}, rule.Name, rule.Source, rule.Threshold, piiCache)
			complete = complete && status == piiScanClean
			clean = clean && len(entities) == 0
		}
		verified = verified && complete && clean
		results.PIIEvidence = append(results.PIIEvidence, NewPrivacyEvidence("request", key.content, complete, clean))
	}
	results.PIIContentVerified = verified

	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()
	results.Metrics.PII.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	if results.PIIDetected {
		results.Metrics.PII.Confidence = 1.0 // Binary: PII found or not
	}

	c.recordSignalExtraction(config.SignalTypePII, "pii_evaluated", latencySeconds)
	logging.Debugf("[Signal Computation] PII signal evaluation completed in %v", elapsed)
}

func (c *Classifier) evaluatePIIRule(rule config.PIIRule, piiText string, nonUserMessages []string, toolResultTexts []string, toolResultScanIncomplete bool, piiCache map[piiCacheKey]cachedPIIContent, start time.Time, results *SignalResults, mu *sync.Mutex) {
	ruleContents := collectPIIRuleContentsForSource(rule, piiText, nonUserMessages, toolResultTexts)
	if len(ruleContents) == 0 {
		if rule.Source == config.PIISourceToolResult && toolResultScanIncomplete {
			c.recordPIIRuleError(rule, piiScanIncomplete, results, mu)
		}
		return
	}

	entityTypes, status, errorCode := c.collectPIIEntityTypes(ruleContents, rule.Name, rule.Source, rule.Threshold, piiCache)
	if rule.Source == config.PIISourceToolResult && toolResultScanIncomplete && status == piiScanClean {
		status = piiScanIncomplete
	}
	unscanned := piiRuleUnscanned(ruleContents, rule.Source, piiCache) && c.Config.PIIModel.UnscannedBlocks()
	if unscanned && errorCode == "" {
		errorCode = signalInputLimitCode
	}
	if status != piiScanClean {
		recordedStatus := status
		// Legacy scans retain the historical failed code for incomplete
		// coverage, while tool-result scans expose the more precise incomplete
		// code. A bounded provider error such as input_limit is always preserved.
		if rule.Source != config.PIISourceToolResult {
			recordedStatus = piiScanFailed
		}
		c.recordPIIRuleErrorCode(rule, recordedStatus, errorCode, results, mu)
	}
	deniedEntities := findDeniedEntities(entityTypes, rule.PIITypesAllowed)
	errorDrivenMatch := false
	if status != piiScanClean && (unscanned || c.Config.PIIModel.IsBlock()) {
		logging.Errorf("[Signal Computation] PII rule %q: content not fully classified; failing closed", rule.Name)
		errorDrivenMatch = len(deniedEntities) == 0
		sentinel := PIIClassificationErrorType
		if unscanned {
			sentinel = PIIUnscannedType
		}
		deniedEntities = append(deniedEntities, sentinel)
	}

	if len(deniedEntities) > 0 {
		c.recordSignalExtraction(config.SignalTypePII, rule.Name, time.Since(start).Seconds())
		c.recordSignalMatch(config.SignalTypePII, rule.Name)

		logging.Debugf("[Signal Computation] PII rule %q matched: denied_entities=%v", rule.Name, deniedEntities)

		mu.Lock()
		results.MatchedPIIRules = append(results.MatchedPIIRules, rule.Name)
		results.PIIDetected = true
		if errorDrivenMatch {
			if results.SignalErrorMatches == nil {
				results.SignalErrorMatches = make(map[string]bool)
			}
			results.SignalErrorMatches[signalConfidenceKey(config.SignalTypePII, rule.Name)] = true
		}
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
	c.recordPIIRuleErrorCode(rule, status, "", results, mu)
}

func (c *Classifier) recordPIIRuleErrorCode(rule config.PIIRule, status piiScanStatus, errorCode string, results *SignalResults, mu *sync.Mutex) {
	code := piiEvaluationFailedCode
	if status == piiScanIncomplete {
		code = piiEvaluationIncompleteCode
	}
	if errorCode != "" && errorCode != piiEvaluationFailedCode {
		code = errorCode
	}

	mu.Lock()
	defer mu.Unlock()
	if results.SignalErrors == nil {
		results.SignalErrors = make(map[string]string)
	}
	results.SignalErrors[signalConfidenceKey(config.SignalTypePII, rule.Name)] = code
}

// piiRuleUnscanned distinguishes unread model input from a backend failure.
func piiRuleUnscanned(contents []string, source string, cache map[piiCacheKey]cachedPIIContent) bool {
	for _, content := range contents {
		for _, result := range cache[piiCacheKey{source: piiCacheSource(source), content: content}].results {
			if UnscannedInput(result.err) {
				return true
			}
		}
	}
	return false
}
