package classification

import (
	"context"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// signalReadiness returns a map indicating whether each signal type's infrastructure is ready.
// Separated from EvaluateAllSignalsWithContext to keep cyclomatic complexity under the linter limit.
func (c *Classifier) signalReadiness() map[string]bool {
	return map[string]bool{
		config.SignalTypeKeyword:       c.keywordClassifier != nil,
		config.SignalTypeEmbedding:     c.keywordEmbeddingClassifier != nil,
		config.SignalTypeDomain:        c.IsCategoryEnabled() && c.categoryInference != nil && c.CategoryMapping != nil,
		config.SignalTypeFactCheck:     len(c.Config.FactCheckRules) > 0 && c.factCheckClassifier != nil && c.factCheckClassifier.IsInitialized(),
		config.SignalTypeUserFeedback:  len(c.Config.UserFeedbackRules) > 0 && c.feedbackDetector != nil && c.feedbackDetector.IsInitialized(),
		config.SignalTypeReask:         c.reaskClassifier != nil,
		config.SignalTypePreference:    len(c.Config.PreferenceRules) > 0 && c.IsPreferenceClassifierEnabled(),
		config.SignalTypeLanguage:      len(c.Config.LanguageRules) > 0 && c.IsLanguageEnabled(),
		config.SignalTypeContext:       c.contextClassifier != nil,
		config.SignalTypeStructure:     c.structureClassifier != nil,
		config.SignalTypeComplexity:    c.complexityClassifier != nil,
		config.SignalTypeModality:      len(c.Config.ModalityRules) > 0 && c.Config.ModalityDetector.Enabled,
		config.SignalTypeJailbreak:     c.isJailbreakSignalReady(),
		config.SignalTypePII:           len(c.Config.PIIRules) > 0 && c.IsPIIEnabled(),
		config.SignalTypeKB:            len(c.kbClassifiers) > 0,
		config.SignalTypeConversation:  len(c.Config.ConversationRules) > 0,
		config.SignalTypeEvent:         c.eventClassifier != nil,
		config.SignalTypeMetadata:      len(c.Config.MetadataRules) > 0,
		config.SignalTypeClassifier:    len(c.genericClassifiers) > 0,
		config.SignalTypeInputModality: len(c.Config.InputModalityRules) > 0,
	}
}

// isJailbreakSignalReady keeps the two jailbreak backends independent. The
// BERT classifier requires Prompt Guard and its model assets; contrastive rules
// require only their preloaded embedding classifiers. Coupling both paths to
// IsJailbreakEnabled silently skipped otherwise healthy contrastive rules when
// the optional Prompt Guard model was disabled.
func (c *Classifier) isJailbreakSignalReady() bool {
	// Response-direction rules are scored from the model's output, so they do
	// not make the request-stage signal ready on their own.
	if len(c.Config.RequestJailbreakRules()) == 0 {
		return false
	}

	requiresBERT := false
	for _, rule := range c.Config.JailbreakRules {
		if rule.Method == "contrastive" {
			if _, ready := c.contrastiveJailbreakClassifiers[rule.Name]; !ready {
				return false
			}
			continue
		}
		requiresBERT = true
	}
	return !requiresBERT || (c.IsJailbreakEnabled() && c.jailbreakInference != nil)
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
	return c.evaluateAllSignalsWithContext(
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
		requestFacts,
		nil,
		false,
	)
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
	return c.evaluateAllSignalsWithContext(
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
		requestFacts,
		decisions,
		true,
	)
}

func (c *Classifier) evaluateAllSignalsWithContext(
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
	signalScope []config.Decision,
	signalScopeSet bool,
) *SignalResults {
	defer c.enterSignalEvaluationLoadGate()()
	// Determine which signals (type:name) should be evaluated
	var usedSignals map[string]bool
	switch {
	case forceEvaluateAll:
		usedSignals = c.getAllSignalTypes()
		logging.Debugf("[Signal Computation] Force evaluate all signals mode enabled")
	case signalScopeSet:
		usedSignals = c.getUsedSignalsForDecisions(signalScope)
	default:
		usedSignals = c.getUsedSignals()
	}

	textForSignal := textForSignalFunc(text, uncompressedText, skipCompressionSignals)
	ready := c.signalReadiness()

	results := &SignalResults{
		Metrics:            &SignalMetricsCollection{},
		SignalConfidences:  make(map[string]float64),
		SignalValues:       make(map[string]float64),
		SignalErrors:       make(map[string]string),
		SignalErrorMatches: make(map[string]bool),
	}
	if requestFacts.Context == nil {
		// The legacy, context-free classifier APIs do not have a caller context.
		// Keep those APIs working while ensuring every request-aware path passes
		// its supplied context all the way to remote category HTTP calls.
		requestFacts.Context = context.Background()
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	imgArg := imageURL

	// Allocate a request-scoped image embedding cache only when an image is
	// actually attached. Two signals - complexity (image rules) and embedding
	// (image-modality rules) - independently pull image embeddings via FFI;
	// the cache lets whichever runs first donate its result to the other,
	// turning two SigLIP forward passes into one. With no image attached,
	// neither signal touches the cache, so leaving it nil is correct.
	var imgCache *requestImageEmbeddingCache
	if imgArg != "" {
		imgCache = newRequestImageEmbeddingCache()
	}

	dispatchers := c.buildSignalDispatchers(
		results,
		&mu,
		textForSignal,
		contextText,
		currentUserText,
		priorUserMessages,
		nonUserMessages,
		hasPriorAssistantReply,
		imgArg,
		imgCache,
		convFacts,
		requestFacts.Context,
		requestFacts,
		usedSignals,
	)
	runSignalDispatchers(dispatchers, usedSignals, ready, &wg)

	wg.Wait()
	results = c.applySignalGroups(results)
	results = c.applySignalComposers(results)
	results = c.applySignalOutputPolicies(results)
	results = c.applyProjections(results)
	return results
}
