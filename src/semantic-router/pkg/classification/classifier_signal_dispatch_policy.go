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
	nonUserMessages []string,
	toolResultTexts []string,
	toolResultScanIncomplete bool,
	convFacts ConversationFacts,
	requestFacts RequestFacts,
	usedSignals map[string]bool,
) []signalDispatch {
	var (
		history     []string
		historyOnce sync.Once
	)
	// Only the safety signals below read this history, so it keeps the user
	// messages routing_input left out.
	routing := requestFacts.RoutingInput
	historyForSignals := func() []string {
		historyOnce.Do(func() {
			history = historyForHistoryAwareSignals(
				safetyPriorUserMessages(priorUserMessages, routing),
				nonUserMessages,
			)
		})
		return history
	}
	return []signalDispatch{
		{
			config.SignalTypeSafety, "Safety",
			func(ctx context.Context) {
				c.evaluateSafetySignals(ctx, results, mu, safetyText(textForSignal(config.SignalTypeSafety), routing), usedSignals)
			},
		},
		{
			config.SignalTypeJailbreak, "Jailbreak",
			func(ctx context.Context) {
				if input := requestFacts.JailbreakInput; input != nil {
					c.evaluateJailbreakSignalPieces(ctx, results, mu,
						jailbreakInputTexts(input.Current), jailbreakInputTexts(input.History))
					return
				}
				c.evaluateJailbreakSignalPieces(
					ctx,
					results,
					mu,
					safetyPieces(textForSignal(config.SignalTypeJailbreak), routing),
					historyForSignals(),
				)
			},
		},
		{
			config.SignalTypePII, "PII",
			func(ctx context.Context) {
				c.evaluatePIISignalPieces(
					ctx,
					results,
					mu,
					safetyPieces(textForSignal(config.SignalTypePII), routing),
					historyForSignals(),
					toolResultTexts,
					toolResultScanIncomplete,
				)
			},
		},
		{
			config.SignalTypeKB, "KB",
			func(context.Context) { c.evaluateKBSignals(results, mu, textForSignal(config.SignalTypeKB)) },
		},
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
			config.SignalTypeEvent, "Event",
			func(context.Context) { c.evaluateEventSignal(results, mu, textForSignal(config.SignalTypeEvent)) },
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
		{
			config.SignalTypeClassifier, "Classifier",
			func(ctx context.Context) {
				c.evaluateGenericClassifierSignals(
					results,
					mu,
					textForSignal(config.SignalTypeClassifier),
					usedSignals,
					ctx,
				)
			},
		},
	}
}
