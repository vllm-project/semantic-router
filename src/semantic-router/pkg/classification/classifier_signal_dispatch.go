package classification

import (
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type signalDispatch struct {
	signalType string
	name       string
	evaluate   func()
}

func (c *Classifier) buildSignalDispatchers(
	results *SignalResults,
	mu *sync.Mutex,
	textForSignal func(string) string,
	contextText string,
	currentUserText string,
	priorUserMessages []string,
	nonUserMessages []string,
	hasPriorAssistantReply bool,
	imgArg string,
	imgCache *requestImageEmbeddingCache, // may be nil; both image-consuming evaluators handle nil via cache.resolve's nil-receiver fallthrough
	convFacts ConversationFacts,
	requestFacts RequestFacts,
	usedSignals map[string]bool,
) []signalDispatch {
	dispatchers := []signalDispatch{
		c.keywordSignalDispatcher(results, mu, textForSignal, priorUserMessages),
		{
			config.SignalTypeEmbedding, "Embedding",
			func() {
				c.evaluateEmbeddingSignal(results, mu, textForSignal(config.SignalTypeEmbedding), imgArg, imgCache)
			},
		},
		{
			config.SignalTypeDomain, "Domain",
			func() { c.evaluateDomainSignal(results, mu, textForSignal(config.SignalTypeDomain)) },
		},
		{
			config.SignalTypeFactCheck, "Fact-check",
			func() { c.evaluateFactCheckSignal(results, mu, textForSignal(config.SignalTypeFactCheck)) },
		},
		{
			config.SignalTypeUserFeedback, "User feedback",
			func() {
				c.evaluateUserFeedbackSignal(
					results,
					mu,
					textForSignal(config.SignalTypeUserFeedback),
					hasPriorAssistantReply,
				)
			},
		},
		{
			config.SignalTypeReask, "Reask",
			func() { c.evaluateBoundedReaskSignal(results, mu, currentUserText, priorUserMessages) },
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
			config.SignalTypeContext, "Context",
			func() { c.evaluateContextSignal(results, mu, contextText) },
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
				c.evaluateComplexitySignal(results, mu, textForSignal(config.SignalTypeComplexity), imgArg, imgCache)
			},
		},
		{
			config.SignalTypeModality, "Modality",
			func() { c.evaluateModalitySignal(results, mu, textForSignal(config.SignalTypeModality)) },
		},
	}
	return append(
		dispatchers,
		c.buildPolicySignalDispatchers(
			results,
			mu,
			textForSignal,
			priorUserMessages,
			nonUserMessages,
			convFacts,
			requestFacts,
			usedSignals,
		)...,
	)
}

func (c *Classifier) keywordSignalDispatcher(
	results *SignalResults,
	mu *sync.Mutex,
	textForSignal func(string) string,
	priorUserMessages []string,
) signalDispatch {
	return signalDispatch{
		config.SignalTypeKeyword, "Keyword",
		func() {
			c.evaluateKeywordSignal(
				results,
				mu,
				keywordSignalText(textForSignal(config.SignalTypeKeyword), priorUserMessages),
			)
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

// keywordSignalHistoryLimit bounds how many prior user messages are joined
// into the keyword evaluation text. Keyword rules are cheap exact-match scans,
// but rescanning the full conversation on every request grows linearly with the
// chat length, so only the most recent window is considered.
const keywordSignalHistoryLimit = 20

// keywordSignalText returns the text evaluated by keyword rules: the current
// user message plus the most recent prior user messages in conversation order.
// Keyword rules are exact-match markers (privacy, local-only, internal-doc, ...),
// so a marker that appeared in an earlier turn must still match on the latest
// turn; the previous behavior evaluated only the current user message and
// silently dropped multi-turn privacy requests. Entries are joined with a
// newline so a rule that matches a phrase spanning the space between two turns
// cannot fire across message boundaries, and empty entries are skipped so stray
// whitespace-only history cannot turn into a false-positive separator.
func keywordSignalText(currentText string, priorUserMessages []string) string {
	if len(priorUserMessages) == 0 {
		return currentText
	}
	start := 0
	if len(priorUserMessages) > keywordSignalHistoryLimit {
		start = len(priorUserMessages) - keywordSignalHistoryLimit
	}
	parts := make([]string, 0, keywordSignalHistoryLimit+1)
	for _, msg := range priorUserMessages[start:] {
		if trimmed := strings.TrimSpace(msg); trimmed != "" {
			parts = append(parts, trimmed)
		}
	}
	if trimmed := strings.TrimSpace(currentText); trimmed != "" {
		parts = append(parts, trimmed)
	}
	return strings.Join(parts, "\n")
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
