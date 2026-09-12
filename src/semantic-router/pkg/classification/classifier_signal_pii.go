package classification

import (
	"context"
	"errors"
	"slices"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// PIIClassificationErrorType is the entity type a PII rule reports when its
// content could not be fully classified and on_error is block: the request is
// unverified, which under fail-closed reads as a match, mirroring
// JailbreakClassificationErrorType.
const PIIClassificationErrorType = "classification_error"

// cachedPIIResult stores a cached PII token classification result.
type cachedPIIResult struct {
	result candle_binding.TokenClassificationResult
	err    error
}

func (c *Classifier) evaluatePIISignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, piiText string, nonUserMessages []string) {
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
	piiCache := make(map[string][]cachedPIIResult, len(uniqueContents))
	for _, content := range uniqueContents {
		chunks := piiSignalChunks(content)
		cached := make([]cachedPIIResult, 0, len(chunks))
		for _, chunk := range chunks {
			tokenResult, err := c.piiInference.ClassifyTokens(ctx, chunk)
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

	c.recordSignalExtraction(config.SignalTypePII, "pii_evaluated", latencySeconds)
	logging.Debugf("[Signal Computation] PII signal evaluation completed in %v", elapsed)
}

// piiRuleHasInferenceError reports whether any chunk the rule reads failed to
// classify. A declared truncation (ErrTokenSpansTruncated) is not a failure
// here: the call succeeded and its spans are valid for the part the provider
// saw, so on_error decides what the unseen remainder means.
func piiRuleHasInferenceError(ruleContents []string, piiCache map[string][]cachedPIIResult) bool {
	for _, content := range ruleContents {
		for _, cached := range piiCache[content] {
			if cached.err != nil && !errors.Is(cached.err, ErrTokenSpansTruncated) {
				return true
			}
		}
	}
	return false
}

func (c *Classifier) evaluatePIIRule(rule config.PIIRule, piiText string, nonUserMessages []string, piiCache map[string][]cachedPIIResult, start time.Time, results *SignalResults, mu *sync.Mutex) {
	ruleContents := collectPIIRuleContents(piiText, nonUserMessages, rule.IncludeHistory)
	if len(ruleContents) == 0 {
		return
	}

	inferenceFailed := piiRuleHasInferenceError(ruleContents, piiCache)
	if inferenceFailed {
		// The failure is recorded visibly either way; on_error below decides
		// whether the rule also fails closed for the content never scored.
		recordSignalRuleErrors(results, mu, config.SignalTypePII, []string{rule.Name}, piiEvaluationFailedCode)
	}

	entityTypes, failed := c.collectPIIEntityTypes(ruleContents, rule.Name, rule.Threshold, piiCache)
	deniedEntities := findDeniedEntities(entityTypes, rule.PIITypesAllowed)
	errorDrivenMatch := false
	if failed && c.Config.PIIModel.IsBlock() {
		// Part of the content was never scored (backend error or a declared
		// truncation). Under on_error: block that is not a clean result.
		logging.Errorf("[Signal Computation] PII rule %q: content not fully classified; failing closed", rule.Name)
		// A denied entity already makes this rule true. Only a match created
		// by the failure itself is unknown to the decision engine.
		errorDrivenMatch = len(deniedEntities) == 0
		deniedEntities = append(deniedEntities, PIIClassificationErrorType)
		if !inferenceFailed {
			// A declared truncation is not an inference error, but under block
			// it still leaves the rule not fully evaluated, and the decision
			// engine reads unknown from the pair (error, error-driven match).
			recordSignalRuleErrors(results, mu, config.SignalTypePII, []string{rule.Name}, piiEvaluationFailedCode)
		}
	}

	if len(deniedEntities) > 0 {
		c.recordSignalExtraction(config.SignalTypePII, rule.Name, time.Since(start).Seconds())
		c.recordSignalMatch(config.SignalTypePII, rule.Name)

		logging.Debugf("[Signal Computation] PII rule %q matched: denied_entities=%v", rule.Name, deniedEntities)

		mu.Lock()
		results.MatchedPIIRules = append(results.MatchedPIIRules, rule.Name)
		results.PIIDetected = true
		if errorDrivenMatch {
			// Same signal the jailbreak path raises: a match that exists only
			// because classification failed must not read as a real detection.
			// decision.evalLeaf turns error plus error-driven match into
			// unknown, so unknown_policy decides instead of the match.
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
