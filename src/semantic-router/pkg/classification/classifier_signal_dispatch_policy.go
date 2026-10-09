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
			config.SignalTypeSafety, "Safety",
			func(ctx context.Context) {
				c.evaluateSafetySignals(ctx, results, mu, textForSignal(config.SignalTypeSafety), usedSignals)
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
				c.evaluateJailbreakSignal(
					ctx,
					results,
					mu,
					textForSignal(config.SignalTypeJailbreak),
					historyForSignals(),
				)
			},
		},
		{
			config.SignalTypePII, "PII",
			func(ctx context.Context) {
				c.evaluatePIISignal(
					ctx,
					results,
					mu,
					textForSignal(config.SignalTypePII),
					historyForSignals(),
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
			config.SignalTypeDecision, "Decision",
			func(ctx context.Context) {
				c.evaluateDecisionModelSignals(
					ctx,
					results,
					mu,
					textForSignal(config.SignalTypeDecision),
					textForSignal(decisionModelQuestionText),
					priorUserMessages,
					usedSignals,
				)
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
