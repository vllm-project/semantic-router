package classification

import (
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
	nonUserMessages []string,
	imgArg string,
) []signalDispatch {
	return []signalDispatch{
		{
			config.SignalTypeKeyword, "Keyword",
			func() { c.evaluateKeywordSignal(results, mu, textForSignal(config.SignalTypeKeyword)) },
		},
		{
			config.SignalTypeEmbedding, "Embedding",
			func() { c.evaluateEmbeddingSignal(results, mu, textForSignal(config.SignalTypeEmbedding)) },
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
			func() { c.evaluateUserFeedbackSignal(results, mu, textForSignal(config.SignalTypeUserFeedback)) },
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
			func() { c.evaluateStructureSignal(results, mu, textForSignal(config.SignalTypeStructure)) },
		},
		{
			config.SignalTypeComplexity, "Complexity",
			func() { c.evaluateComplexitySignal(results, mu, textForSignal(config.SignalTypeComplexity), imgArg) },
		},
		{
			config.SignalTypeModality, "Modality",
			func() { c.evaluateModalitySignal(results, mu, textForSignal(config.SignalTypeModality)) },
		},
		{
			config.SignalTypeJailbreak, "Jailbreak",
			func() {
				c.evaluateJailbreakSignal(results, mu, textForSignal(config.SignalTypeJailbreak), nonUserMessages)
			},
		},
		{
			config.SignalTypePII, "PII",
			func() { c.evaluatePIISignal(results, mu, textForSignal(config.SignalTypePII), nonUserMessages) },
		},
		{
			config.SignalTypeKB, "KB",
			func() { c.evaluateKBSignals(results, mu, textForSignal(config.SignalTypeKB)) },
		},
	}
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
