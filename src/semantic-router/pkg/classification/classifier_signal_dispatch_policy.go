package classification

import (
	"context"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (c *Classifier) buildPolicySignalDispatchers(
	results *SignalResults,
	mu *sync.Mutex,
	textForSignal func(string) string,
	priorUserMessages []string,
	toolResultTexts []string,
	convFacts ConversationFacts,
	requestFacts RequestFacts,
	usedSignals map[string]bool,
) []signalDispatch {
	return []signalDispatch{
		{
			config.SignalTypeConversation, "Conversation",
			func(context.Context) {
				c.evaluateConversationSignal(
					results,
					mu,
					convFacts,
					usedSignals,
				)
			},
		},
		{
			config.SignalTypeMetadata, "Metadata",
			func(context.Context) { c.evaluateMetadataSignal(results, mu, requestFacts, usedSignals) },
		},
		{
			config.SignalTypeInputModality, "InputModality",
			func(context.Context) { c.evaluateInputModalitySignal(results, mu, requestFacts, usedSignals) },
		},
		{
			config.SignalTypeSystemOne, "SystemOne",
			func(ctx context.Context) {
				c.evaluateSystemOneSignals(ctx, results, mu, systemOneReading{
					request: textForSignal(systemOneQuestionText), history: priorUserMessages, turns: requestFacts.Turns,
					toolResults: toolResultTexts, images: requestFacts.Images, videos: requestFacts.Videos,
				}, usedSignals)
			},
		},
	}
}
