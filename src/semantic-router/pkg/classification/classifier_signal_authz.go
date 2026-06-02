package classification

import (
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// EvaluateAllSignalsWithHeaders evaluates all signal types including the authz signal.
// The authz signal reads user identity and groups from request headers (x-authz-user-id,
// x-authz-user-groups) and evaluates role_bindings. Other signals are evaluated via
// EvaluateAllSignalsWithContext as before.
//
// Returns an error if authz evaluation fails (e.g., missing user identity header when
// role_bindings are configured). Errors are NOT swallowed — the caller must handle them.
// This prevents silent bypass of authz policies.
//
// headers: request headers from ext_proc (includes Authorino-injected authz headers)
//
// Optional trailing arguments (positional after imageURL):
//   - uncompressedText (string): original text before prompt compression
//   - skipCompressionSignals (map[string]bool): signal types that must use uncompressedText
//   - convFacts (ConversationFacts): optional conversation facts for conversation signals
func (c *Classifier) EvaluateAllSignalsWithHeaders(text string, contextText string, currentUserText string, priorUserMessages []string, nonUserMessages []string, hasPriorAssistantReply bool, headers map[string]string, forceEvaluateAll bool, imageURL string, extra ...interface{}) (*SignalResults, error) {
	opts := parseEvaluateSignalsExtra(extra)
	var img []string
	if strings.TrimSpace(imageURL) != "" {
		img = []string{imageURL}
	}
	results := c.EvaluateAllSignalsWithContext(text, contextText, currentUserText, priorUserMessages, nonUserMessages, hasPriorAssistantReply, forceEvaluateAll, opts.uncompressedText, opts.skipCompressionSignals, opts.convFacts, img...)
	if err := c.appendAuthzFromHeaders(results, headers, forceEvaluateAll); err != nil {
		return nil, err
	}
	return results, nil
}

// evaluateSignalsExtra mirrors optional trailing args after imageURL for EvaluateAllSignalsWithHeaders.
type evaluateSignalsExtra struct {
	uncompressedText       string
	skipCompressionSignals map[string]bool
	convFacts              ConversationFacts
}

func parseEvaluateSignalsExtra(extra []interface{}) evaluateSignalsExtra {
	var e evaluateSignalsExtra
	if len(extra) < 2 {
		return e
	}
	if s, ok := extra[0].(string); ok {
		e.uncompressedText = s
	}
	if m, ok := extra[1].(map[string]bool); ok {
		e.skipCompressionSignals = m
	}
	if len(extra) >= 3 {
		if cf, ok := extra[2].(ConversationFacts); ok {
			e.convFacts = cf
		}
	}
	return e
}

func (c *Classifier) appendAuthzFromHeaders(results *SignalResults, headers map[string]string, forceEvaluateAll bool) error {
	usedSignals := c.getUsedSignals()
	if forceEvaluateAll {
		usedSignals = c.getAllSignalTypes()
	}
	if !isSignalTypeUsed(usedSignals, config.SignalTypeAuthz) {
		logging.Debugf("[Signal Computation] Authz signal not used in any decision, skipping evaluation")
		return nil
	}
	if c.authzClassifier == nil {
		return nil
	}

	start := time.Now()
	userID := headers[c.authzUserIDHeader]
	userGroups := ParseUserGroups(headers[c.authzUserGroupsHeader])

	authzResult, err := c.authzClassifier.Classify(userID, userGroups)
	authzResult, err = applyAuthzFailOpenOnClassifyError(c.authzFailOpen, userID, authzResult, err)
	elapsed := time.Since(start)
	latencySeconds := elapsed.Seconds()

	results.Metrics.Authz.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
	results.Metrics.Authz.Confidence = 1.0

	if err != nil {
		logging.Errorf("[Authz Signal] classification failed: %v", err)
		metrics.RecordSignalExtraction(config.SignalTypeAuthz, "error", latencySeconds)
		return fmt.Errorf("authz signal evaluation failed: %w", err)
	}

	for _, ruleName := range authzResult.MatchedRules {
		metrics.RecordSignalExtraction(config.SignalTypeAuthz, ruleName, latencySeconds)
		metrics.RecordSignalMatch(config.SignalTypeAuthz, ruleName)
	}
	results.MatchedAuthzRules = authzResult.MatchedRules
	logging.Debugf("[Signal Computation] Authz signal evaluation completed in %v", elapsed)
	return nil
}
