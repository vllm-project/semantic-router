package classification

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (c *Classifier) buildPolicySignalDispatchers(
	results *SignalResults,
	mu *sync.Mutex,
	textForSignal func(string) string,
	priorUserMessages []string,
	nonUserMessages []string,
	convFacts ConversationFacts,
	requestFacts RequestFacts,
	usedSignals map[string]bool,
) []signalDispatch {
	var (
		history     []string
		historyOnce sync.Once
	)
	historyForSignals := func() []string {
		historyOnce.Do(func() {
			history = historyForHistoryAwareSignals(
				priorUserMessages,
				nonUserMessages,
			)
		})
		return history
	}
	return []signalDispatch{
		{
			config.SignalTypeJailbreak, "Jailbreak",
			func() {
				c.evaluateJailbreakSignal(
					requestFacts.Context,
					results,
					mu,
					textForSignal(config.SignalTypeJailbreak),
					historyForSignals(),
				)
			},
		},
		{
			config.SignalTypePII, "PII",
			func() {
				c.evaluatePIISignal(
					results,
					mu,
					textForSignal(config.SignalTypePII),
					historyForSignals(),
				)
			},
		},
		{
			config.SignalTypeKB, "KB",
			func() { c.evaluateKBSignals(results, mu, textForSignal(config.SignalTypeKB)) },
		},
		{
			config.SignalTypeConversation, "Conversation",
			func() {
				c.evaluateConversationSignal(
					results,
					mu,
					convFacts,
					usedSignals,
				)
			},
		},
		{
			config.SignalTypeEvent, "Event",
			func() { c.evaluateEventSignal(results, mu, textForSignal(config.SignalTypeEvent)) },
		},
		{
			config.SignalTypeMetadata, "Metadata",
			func() { c.evaluateMetadataSignal(results, mu, requestFacts, usedSignals) },
		},
		{
			config.SignalTypeClassifier, "Classifier",
			func() {
				c.evaluateGenericClassifierSignals(
					results,
					mu,
					textForSignal(config.SignalTypeClassifier),
					usedSignals,
					requestFacts.Context,
				)
			},
		},
	}
}
