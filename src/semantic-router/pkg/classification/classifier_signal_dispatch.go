package classification

import (
	"context"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// signalDispatch is one signal's evaluation; evaluate runs with the
// context of the signal's participant in the stage's bundle.
type signalDispatch struct {
	signalType string
	name       string
	evaluate   func(ctx context.Context)
}

func (c *Classifier) buildSignalDispatchers(input SignalEvaluationInput, results *SignalResults, mu *sync.Mutex, textForSignal func(string) string, mediaCache *requestMediaEmbeddingCache, usedSignals map[string]bool) []signalDispatch {
	dispatchers := c.buildPrimarySignalDispatchers(input, results, mu, textForSignal, mediaCache)
	dispatchers = append(dispatchers, c.buildRequestFactSignalDispatchers(
		results, mu, textForSignal, input.ContextText, input.CurrentUserText,
		input.ImageURL, mediaCache, input.RequestFacts,
	)...)
	return append(dispatchers, c.buildPolicySignalDispatchers(
		results, mu, textForSignal, input.PriorUserMessages, input.NonUserMessages,
		input.ToolResultTexts, input.ToolResultScanIncomplete,
		input.ConversationFacts, input.RequestFacts, usedSignals,
	)...)
}

func (c *Classifier) buildPrimarySignalDispatchers(input SignalEvaluationInput, results *SignalResults, mu *sync.Mutex, textForSignal func(string) string, mediaCache *requestMediaEmbeddingCache) []signalDispatch {
	return []signalDispatch{
		{
			config.SignalTypeKeyword, "Keyword",
			func(context.Context) { c.evaluateKeywordSignal(results, mu, textForSignal(config.SignalTypeKeyword)) },
		},
		{
			config.SignalTypeEmbedding, "Embedding",
			func(ctx context.Context) {
				c.evaluateEmbeddingSignal(ctx, results, mu, embeddingSignalInput{Text: textForSignal(config.SignalTypeEmbedding), Image: input.ImageURL, Audio: input.Audio}, mediaCache)
			},
		},
		{
			config.SignalTypeDomain, "Domain",
			func(ctx context.Context) {
				c.evaluateDomainSignal(ctx, results, mu, textForSignal(config.SignalTypeDomain))
			},
		},
		{
			config.SignalTypeFactCheck, "Fact-check",
			func(ctx context.Context) {
				c.evaluateFactCheckSignal(ctx, results, mu, textForSignal(config.SignalTypeFactCheck))
			},
		},
		{
			config.SignalTypeUserFeedback, "User feedback",
			func(ctx context.Context) {
				c.evaluateUserFeedbackSignal(
					ctx,
					results,
					mu,
					textForSignal(config.SignalTypeUserFeedback),
					input.HasPriorAssistantReply,
				)
			},
		},
		{
			config.SignalTypeReask, "Reask",
			func(ctx context.Context) {
				c.evaluateReaskSignalContext(ctx, results, mu, input.CurrentUserText, input.PriorUserMessages)
			},
		},
		{
			config.SignalTypePreference, "Preference",
			func(ctx context.Context) {
				c.evaluatePreferenceSignal(ctx, results, mu, textForSignal(config.SignalTypePreference))
			},
		},
		{
			config.SignalTypeLanguage, "Language",
			func(context.Context) { c.evaluateLanguageSignal(results, mu, textForSignal(config.SignalTypeLanguage)) },
		},
		{
			config.SignalTypeAction, "Action",
			func(context.Context) { c.evaluateActionSignal(results, mu, input.CurrentUserText) },
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
) []signalDispatch {
	return []signalDispatch{
		{
			config.SignalTypeContext, "Context",
			func(context.Context) {
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
			func(context.Context) {
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
			func(ctx context.Context) {
				c.evaluateComplexitySignal(ctx, results, mu, textForSignal(config.SignalTypeComplexity), imgArg, imgCache)
			},
		},
		{
			config.SignalTypeModality, "Modality",
			func(ctx context.Context) {
				c.evaluateModalitySignal(ctx, results, mu, textForSignal(config.SignalTypeModality))
			},
		},
	}
}

// modelBackedSignalTypes call model deployments; their goroutines join the
// stage's request bundle, each declaring the deployments it asks questions,
// so a deployment's questions are sent once all of its askers have asked.
// Heuristic signals never delay a send.
var modelBackedSignalTypes = map[string]bool{
	config.SignalTypeDomain:       true,
	config.SignalTypeFactCheck:    true,
	config.SignalTypeUserFeedback: true,
	config.SignalTypeModality:     true,
	config.SignalTypeSafety:       true,
	config.SignalTypeJailbreak:    true,
	config.SignalTypePII:          true,
	config.SignalTypeClassifier:   true,
	config.SignalTypeDecision:     true,
	config.SignalTypeEmbedding:    true,
	config.SignalTypeComplexity:   true,
	config.SignalTypeKB:           true,
	config.SignalTypeReask:        true,
	config.SignalTypePreference:   true,
}

// stageBundle is the part of a request bundle the dispatchers use.
type stageBundle interface {
	JoinAsking(ctx context.Context, deployments ...string) (context.Context, func())
}

// runSignalDispatchers joins every model-backed participant, with the
// deployments asks says its signal asks, before it starts any: a participant
// that parks its call at once must not find the others not yet joined, which
// would send the bundle's calls without theirs. Every signal runs with stage
// or, when it joined, its participant's context.
func runSignalDispatchers(stage context.Context, dispatchers []signalDispatch, usedSignals map[string]bool, ready map[string]bool, bundle stageBundle, asks func(signalType string) []string, wg *sync.WaitGroup) {
	type run struct {
		dispatch signalDispatch
		ctx      context.Context
		leave    func()
	}
	runs := make([]run, 0, len(dispatchers))
	for _, d := range dispatchers {
		if isSignalTypeUsed(usedSignals, d.signalType) && ready[d.signalType] {
			ctx, leave := stage, func() {}
			if modelBackedSignalTypes[d.signalType] {
				ctx, leave = bundle.JoinAsking(stage, asks(d.signalType)...)
			}
			runs = append(runs, run{dispatch: d, ctx: ctx, leave: leave})
			continue
		}

		if !isSignalTypeUsed(usedSignals, d.signalType) {
			logging.Debugf("[Signal Computation] %s signal not used in any decision, skipping evaluation", d.name)
		}
	}
	for _, r := range runs {
		wg.Add(1)
		go func(r run) {
			defer wg.Done()
			defer r.leave()
			r.dispatch.evaluate(r.ctx)
		}(r)
	}
}
