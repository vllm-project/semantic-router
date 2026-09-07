package classification

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SignalEvaluationInput is the complete, typed input to request-time signal
// evaluation. Keeping optional values named avoids positional interface{}
// arguments and makes new signal context additive without changing call order.
type SignalEvaluationInput struct {
	Text                   string
	ContextText            string
	CurrentUserText        string
	PriorUserMessages      []string
	NonUserMessages        []string
	HasPriorAssistantReply bool
	Headers                map[string]string
	ForceEvaluateAll       bool
	ImageURL               string
	UncompressedText       string
	SkipCompressionSignals map[string]bool
	ConversationFacts      ConversationFacts
	RequestFacts           RequestFacts
}

// EvaluateAllSignalsWithHeaders evaluates the selected recipe's signals,
// including authz role bindings. Authz errors are returned to the caller so a
// missing identity cannot silently bypass policy.
func (c *Classifier) EvaluateAllSignalsWithHeaders(input SignalEvaluationInput) (*SignalResults, error) {
	results := c.evaluateAllSignalsWithContext(
		input.Text,
		input.ContextText,
		input.CurrentUserText,
		input.PriorUserMessages,
		input.NonUserMessages,
		input.HasPriorAssistantReply,
		input.ForceEvaluateAll,
		input.UncompressedText,
		input.SkipCompressionSignals,
		input.ConversationFacts,
		input.ImageURL,
		input.RequestFacts,
		nil,
		false,
	)
	if err := c.appendAuthzFromHeaders(results, input.Headers, input.ForceEvaluateAll); err != nil {
		return nil, err
	}
	return results, nil
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
		c.recordSignalExtraction(config.SignalTypeAuthz, "error", latencySeconds)
		return fmt.Errorf("authz signal evaluation failed: %w", err)
	}

	for _, ruleName := range authzResult.MatchedRules {
		c.recordSignalExtraction(config.SignalTypeAuthz, ruleName, latencySeconds)
		c.recordSignalMatch(config.SignalTypeAuthz, ruleName)
	}
	results.MatchedAuthzRules = authzResult.MatchedRules
	logging.Debugf("[Signal Computation] Authz signal evaluation completed in %v", elapsed)
	return nil
}
