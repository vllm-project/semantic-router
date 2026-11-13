package classification

import (
	"context"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// signalReadiness returns a map indicating whether each signal type's infrastructure is ready.
// Separated from EvaluateAllSignalsWithContext to keep cyclomatic complexity under the linter limit.
func (c *Classifier) signalReadiness() map[string]bool {
	ready := map[string]bool{
		config.SignalTypeKeyword:       c.keywordClassifier != nil,
		config.SignalTypeLanguage:      len(c.Config.LanguageRules) > 0 && c.IsLanguageEnabled(),
		config.SignalTypeContext:       c.contextClassifier != nil,
		config.SignalTypeConversation:  len(c.Config.ConversationRules) > 0,
		config.SignalTypeMetadata:      len(c.Config.MetadataRules) > 0,
		config.SignalTypeInputModality: len(c.Config.InputModalityRules) > 0,
		config.SignalTypeSystemOne:     len(c.Config.SystemOneRules) > 0,
	}
	for _, rule := range c.applicationRules {
		ready[rule.Type] = true
	}
	return ready
}

// textForSignalFunc returns a function that resolves the correct text for a given signal type,
// using uncompressed text for signals that must not receive compressed input.
func textForSignalFunc(text, uncompressedText string, skipCompressionSignals map[string]bool) func(string) string {
	return func(signalType string) string {
		resolved := text
		if uncompressedText != "" && skipCompressionSignals[signalType] {
			resolved = uncompressedText
		}
		return textForRoutingSignal(signalType, resolved)
	}
}

// EvaluateAllSignalsWithContext evaluates all signal types with separate text for context counting.
//
// text: (possibly compressed) text for signal evaluation
// contextText: text for context token counting (usually all messages combined)
// nonUserMessages: conversation history for jailbreak/PII with include_history
// forceEvaluateAll: if true, evaluates all configured signals regardless of decision usage
// uncompressedText: original text before prompt compression (empty = no compression happened)
// skipCompressionSignals: signal types that must use uncompressedText instead of text
// imageURL: image URL for multimodal signals ("" when the request carries no image)
func (c *Classifier) EvaluateAllSignalsWithContext(text string, contextText string, currentUserText string, priorUserMessages []string, nonUserMessages []string, hasPriorAssistantReply bool, forceEvaluateAll bool, uncompressedText string, skipCompressionSignals map[string]bool, convFacts ConversationFacts, imageURL string) *SignalResults {
	return c.EvaluateAllSignalsWithRequestFacts(
		text,
		contextText,
		currentUserText,
		priorUserMessages,
		nonUserMessages,
		hasPriorAssistantReply,
		forceEvaluateAll,
		uncompressedText,
		skipCompressionSignals,
		convFacts,
		imageURL,
		RequestFacts{},
	)
}

// EvaluateAllSignalsWithRequestFacts extends context-aware evaluation with
// bounded request-envelope facts used by metadata and conversation signals.
func (c *Classifier) EvaluateAllSignalsWithRequestFacts(
	text string,
	contextText string,
	currentUserText string,
	priorUserMessages []string,
	nonUserMessages []string,
	hasPriorAssistantReply bool,
	forceEvaluateAll bool,
	uncompressedText string,
	skipCompressionSignals map[string]bool,
	convFacts ConversationFacts,
	imageURL string,
	requestFacts RequestFacts,
) *SignalResults {
	return c.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: text, ContextText: contextText, CurrentUserText: currentUserText,
		PriorUserMessages: priorUserMessages, NonUserMessages: nonUserMessages,
		HasPriorAssistantReply: hasPriorAssistantReply, ForceEvaluateAll: forceEvaluateAll,
		UncompressedText: uncompressedText, SkipCompressionSignals: skipCompressionSignals,
		ConversationFacts: convFacts, ImageURL: imageURL, RequestFacts: requestFacts,
	}, nil, false)
}

// EvaluateAllSignalsWithRequestFactsForDecisions scopes signal usage to one
// routing profile while carrying request-envelope facts.
func (c *Classifier) EvaluateAllSignalsWithRequestFactsForDecisions(
	text string,
	contextText string,
	currentUserText string,
	priorUserMessages []string,
	nonUserMessages []string,
	hasPriorAssistantReply bool,
	forceEvaluateAll bool,
	uncompressedText string,
	skipCompressionSignals map[string]bool,
	convFacts ConversationFacts,
	imageURL string,
	requestFacts RequestFacts,
	decisions []config.Decision,
) *SignalResults {
	return c.evaluateAllSignalsWithContext(SignalEvaluationInput{
		Text: text, ContextText: contextText, CurrentUserText: currentUserText,
		PriorUserMessages: priorUserMessages, NonUserMessages: nonUserMessages,
		HasPriorAssistantReply: hasPriorAssistantReply, ForceEvaluateAll: forceEvaluateAll,
		UncompressedText: uncompressedText, SkipCompressionSignals: skipCompressionSignals,
		ConversationFacts: convFacts, ImageURL: imageURL, RequestFacts: requestFacts,
	}, decisions, true)
}

func (c *Classifier) evaluateAllSignalsWithContext(input SignalEvaluationInput, signalScope []config.Decision, signalScopeSet bool) *SignalResults {
	return c.applyProjections(c.applySignalGroups(c.evaluateBaseSignals(input, signalScope, signalScopeSet)))
}

// evaluateBaseSignals evaluates the request's signals, without the partitions
// and projections derived from them.
func (c *Classifier) evaluateBaseSignals(input SignalEvaluationInput, signalScope []config.Decision, signalScopeSet bool) *SignalResults {
	// Determine which signals (type:name) should be evaluated
	var usedSignals map[string]bool
	switch {
	case input.ForceEvaluateAll:
		usedSignals = c.getAllSignalTypes()
		logging.Debugf("[Signal Computation] Force evaluate all signals mode enabled")
	case signalScopeSet:
		usedSignals = c.getUsedSignalsForDecisions(signalScope)
	default:
		usedSignals = c.getUsedSignals()
	}

	boundedText := textForSignalFunc(input.Text, input.UncompressedText, input.SkipCompressionSignals)
	textForSignal := func(signalType string) string {
		// The systemone and application signals ask the decision model,
		// which reads the request as it came.
		if signalType == systemOneQuestionText {
			if input.UncompressedText != "" {
				return input.UncompressedText
			}
			return input.Text
		}
		return boundedText(signalType)
	}
	ready := c.signalReadiness()

	results := &SignalResults{
		Metrics:            &SignalMetricsCollection{},
		SignalConfidences:  make(map[string]float64),
		SignalValues:       make(map[string]float64),
		SignalErrors:       make(map[string]string),
		SignalErrorMatches: make(map[string]bool),
	}
	if input.RequestFacts.Context == nil {
		// The legacy, context-free classifier APIs do not have a caller context.
		// Keep those APIs working while ensuring every request-aware path passes
		// its supplied context all the way to remote category HTTP calls.
		input.RequestFacts.Context = context.Background()
	}

	// One request stage, one bundle: the model calls of every signal reach each
	// runtime process as a single /v1/bundle call, before the signals' deadline,
	// and every question the stage asks one decision model goes in one call.
	stage, bundle := modelservice.WithBundle(input.RequestFacts.Context, 0)
	stage, cancel := withSignalDeadline(stage, c.Config.SignalTimeout(), time.Now())
	defer cancel()
	stage, endScans := withScanStop(stage)
	defer endScans()
	input.RequestFacts.Context = stage
	var wg sync.WaitGroup
	var mu sync.Mutex
	dispatchers := c.buildSignalDispatchers(input, results, &mu, textForSignal, usedSignals)

	asks := func(signalType string) []string { return c.signalQuestionDeployments(signalType, usedSignals) }
	runSignalDispatchers(stage, dispatchers, usedSignals, ready, bundle, asks, &wg)

	wg.Wait()
	request := textForSignal(systemOneQuestionText)
	for _, evidence := range results.PIIEvidence {
		results.PIIContentVerified = results.PIIContentVerified || evidence.CoversClean("request", request)
	}
	if input.ImageURL != "" || input.Audio != "" {
		results.PIIContentVerified = false
	}
	return results
}
