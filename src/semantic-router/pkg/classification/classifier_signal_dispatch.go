package classification

import (
	"context"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type signalDispatch struct {
	signalType string
	name       string
	evaluate   func()
}

func (c *Classifier) buildSignalDispatchers(input SignalEvaluationInput, results *SignalResults, mu *sync.Mutex, textForSignal func(string) string, mediaCache *requestMediaEmbeddingCache, usedSignals map[string]bool) []signalDispatch {
	dispatchers := c.buildPrimarySignalDispatchers(input, results, mu, textForSignal, mediaCache)
	dispatchers = append(dispatchers, c.buildRequestFactSignalDispatchers(
		results, mu, textForSignal, input.ContextText, input.CurrentUserText,
		input.ImageURL, mediaCache, input.RequestFacts, input.RequestFacts.Context,
	)...)
	return append(dispatchers, c.buildPolicySignalDispatchers(
		results, mu, textForSignal, input.PriorUserMessages, input.NonUserMessages,
		input.ConversationFacts, input.RequestFacts, usedSignals,
	)...)
}

func (c *Classifier) buildPrimarySignalDispatchers(input SignalEvaluationInput, results *SignalResults, mu *sync.Mutex, textForSignal func(string) string, mediaCache *requestMediaEmbeddingCache) []signalDispatch {
	return []signalDispatch{
		{
			config.SignalTypeKeyword, "Keyword",
			func() { c.evaluateKeywordSignal(results, mu, textForSignal(config.SignalTypeKeyword)) },
		},
		{
			config.SignalTypeEmbedding, "Embedding",
			func() {
				c.evaluateEmbeddingSignal(input.RequestFacts.Context, results, mu, embeddingSignalInput{Text: textForSignal(config.SignalTypeEmbedding), Image: input.ImageURL, Audio: input.Audio}, mediaCache)
			},
		},
		{
			config.SignalTypeDomain, "Domain",
			func() {
				c.evaluateDomainSignal(input.RequestFacts.Context, results, mu, textForSignal(config.SignalTypeDomain))
			},
		},
		{
			config.SignalTypeFactCheck, "Fact-check",
			func() {
				c.evaluateFactCheckSignal(input.RequestFacts.Context, results, mu, textForSignal(config.SignalTypeFactCheck))
			},
		},
		{
			config.SignalTypeUserFeedback, "User feedback",
			func() {
				c.evaluateUserFeedbackSignal(
					input.RequestFacts.Context,
					results,
					mu,
					textForSignal(config.SignalTypeUserFeedback),
					input.HasPriorAssistantReply,
				)
			},
		},
		{
			config.SignalTypeReask, "Reask",
			func() { c.evaluateBoundedReaskSignal(results, mu, input.CurrentUserText, input.PriorUserMessages) },
		},
		{
			config.SignalTypePreference, "Preference",
			func() { c.evaluatePreferenceSignal(results, mu, textForSignal(config.SignalTypePreference)) },
		},
		{
			config.SignalTypeLanguage, "Language",
			func() { c.evaluateLanguageSignal(results, mu, textForSignal(config.SignalTypeLanguage)) },
		},
		{
			config.SignalTypeAction, "Action",
			func() { c.evaluateActionSignal(results, mu, input.CurrentUserText) },
		},
	}
}

func (c *Classifier) buildRequestFactSignalDispatchers(
	results *SignalResults,
	mu *sync.Mutex,
	textForSignal func(string) string,
	contextText string,
	currentUserText string,
	imgArg string,
	imgCache *requestMediaEmbeddingCache,
	requestFacts RequestFacts,
	requestCtx context.Context,
) []signalDispatch {
	return []signalDispatch{
		{
			config.SignalTypeContext, "Context",
			func() {
				c.evaluateContextSignal(
					results,
					mu,
					contextText,
					requestFacts.ContextTokenFloor,
				)
			},
		},
		{
			config.SignalTypeStructure, "Structure",
			func() {
				c.evaluateStructureSignal(
					results,
					mu,
					textForSignal(config.SignalTypeStructure),
					currentUserText,
				)
			},
		},
		{
			config.SignalTypeComplexity, "Complexity",
			func() {
				c.evaluateComplexitySignal(requestCtx, results, mu, textForSignal(config.SignalTypeComplexity), imgArg, imgCache)
			},
		},
		{
			config.SignalTypeModality, "Modality",
			func() { c.evaluateModalitySignal(requestCtx, results, mu, textForSignal(config.SignalTypeModality)) },
		},
	}
}

func (c *Classifier) evaluateBoundedReaskSignal(
	results *SignalResults,
	mu *sync.Mutex,
	currentUserText string,
	priorUserMessages []string,
) {
	c.evaluateReaskSignal(
		results,
		mu,
		textForRoutingSignal(config.SignalTypeReask, currentUserText),
		boundedReaskMessages(priorUserMessages),
	)
}

func boundedReaskMessages(messages []string) []string {
	bounded := make([]string, len(messages))
	for index, message := range messages {
		bounded[index] = textForRoutingSignal(config.SignalTypeReask, message)
	}
	return bounded
}

func runSignalDispatchers(dispatchers []signalDispatch, usedSignals map[string]bool, ready map[string]bool, wg *sync.WaitGroup) {
	for _, d := range dispatchers {
		if isSignalTypeUsed(usedSignals, d.signalType) && ready[d.signalType] {
			wg.Add(1)
			go func(dispatch signalDispatch) {
				defer wg.Done()
				dispatch.evaluate()
			}(d)
			continue
		}

		if !isSignalTypeUsed(usedSignals, d.signalType) {
			logging.Debugf("[Signal Computation] %s signal not used in any decision, skipping evaluation", d.name)
		}
	}
}
