package classification

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// signalReadiness returns a map indicating whether each signal type's infrastructure is ready.
// Separated from EvaluateAllSignalsWithContext to keep cyclomatic complexity under the linter limit.
func (c *Classifier) signalReadiness() map[string]bool {
	return map[string]bool{
		config.SignalTypeKeyword:      c.keywordClassifier != nil,
		config.SignalTypeEmbedding:    c.keywordEmbeddingClassifier != nil,
		config.SignalTypeDomain:       c.IsCategoryEnabled() && c.categoryInference != nil && c.CategoryMapping != nil,
		config.SignalTypeFactCheck:    len(c.Config.FactCheckRules) > 0 && c.IsFactCheckEnabled(),
		config.SignalTypeUserFeedback: len(c.Config.UserFeedbackRules) > 0 && c.IsFeedbackDetectorEnabled(),
		config.SignalTypeReask:        c.reaskClassifier != nil,
		config.SignalTypePreference:   len(c.Config.PreferenceRules) > 0 && c.IsPreferenceClassifierEnabled(),
		config.SignalTypeLanguage:     len(c.Config.LanguageRules) > 0 && c.IsLanguageEnabled(),
		config.SignalTypeContext:      c.contextClassifier != nil,
		config.SignalTypeStructure:    c.structureClassifier != nil,
		config.SignalTypeComplexity:   c.complexityClassifier != nil,
		config.SignalTypeModality:     len(c.Config.ModalityRules) > 0 && c.Config.ModalityDetector.Enabled,
		config.SignalTypeJailbreak:    len(c.Config.JailbreakRules) > 0 && c.IsJailbreakEnabled(),
		config.SignalTypePII:          len(c.Config.PIIRules) > 0 && c.IsPIIEnabled(),
		config.SignalTypeKB:           len(c.kbClassifiers) > 0,
		config.SignalTypeConversation: len(c.Config.ConversationRules) > 0,
		config.SignalTypeEvent:        c.eventClassifier != nil,
	}
}

// textForSignalFunc returns a function that resolves the correct text for a given signal type,
// using uncompressed text for signals that must not receive compressed input.
func textForSignalFunc(text, uncompressedText string, skipCompressionSignals map[string]bool) func(string) string {
	return func(signalType string) string {
		if uncompressedText != "" && skipCompressionSignals[signalType] {
			return uncompressedText
		}
		return text
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
// imageURL: optional image URL for multimodal signals
func (c *Classifier) EvaluateAllSignalsWithContext(text string, contextText string, currentUserText string, priorUserMessages []string, nonUserMessages []string, hasPriorAssistantReply bool, forceEvaluateAll bool, uncompressedText string, skipCompressionSignals map[string]bool, convFacts ConversationFacts, imageURL ...string) *SignalResults {
	defer c.enterSignalEvaluationLoadGate()()
	// Determine which signals (type:name) should be evaluated
	var usedSignals map[string]bool
	if forceEvaluateAll {
		usedSignals = c.getAllSignalTypes()
		logging.Debugf("[Signal Computation] Force evaluate all signals mode enabled")
	} else {
		usedSignals = c.getUsedSignals()
	}

	textForSignal := textForSignalFunc(text, uncompressedText, skipCompressionSignals)
	ready := c.signalReadiness()

	results := &SignalResults{
		Metrics:           &SignalMetricsCollection{},
		SignalConfidences: make(map[string]float64),
		SignalValues:      make(map[string]float64),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	imgArg := ""
	if len(imageURL) > 0 {
		imgArg = imageURL[0]
	}

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

	dispatchers := c.buildSignalDispatchers(results, &mu, textForSignal, contextText, currentUserText, priorUserMessages, nonUserMessages, hasPriorAssistantReply, imgArg, imgCache, convFacts)
	runSignalDispatchers(dispatchers, usedSignals, ready, &wg)

	wg.Wait()
	results = c.applySignalGroups(results)
	results = c.applySignalComposers(results)
	results = c.applySignalOutputPolicies(results)
	results = c.applyProjections(results)
	return results
}
