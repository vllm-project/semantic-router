package classification

import (
	"context"
	"strings"
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

func (c *Classifier) buildSignalDispatchers(input SignalEvaluationInput, results *SignalResults, mu *sync.Mutex, textForSignal func(string) string, usedSignals map[string]bool) []signalDispatch {
	dispatchers := c.buildPrimarySignalDispatchers(input, results, mu, textForSignal)
	dispatchers = append(dispatchers, c.buildRequestFactSignalDispatchers(
		results, mu, input.ContextText, input.RequestFacts,
	)...)
	dispatchers = append(dispatchers, c.buildPolicySignalDispatchers(
		results, mu, textForSignal, input.PriorUserMessages, input.ToolResultTexts, input.ToolResultScanIncomplete,
		input.ConversationFacts, input.RequestFacts, usedSignals,
	)...)
	return append(dispatchers, c.buildApplicationSignalDispatchers(input, results, mu, textForSignal, usedSignals)...)
}

func (c *Classifier) buildPrimarySignalDispatchers(input SignalEvaluationInput, results *SignalResults, mu *sync.Mutex, textForSignal func(string) string) []signalDispatch {
	return []signalDispatch{
		{
			config.SignalTypeKeyword, "Keyword",
			func(context.Context) {
				c.evaluateKeywordSignalWithSources(results, mu, KeywordMatchInput{
					CurrentText:           textForSignal(config.SignalTypeKeyword),
					ToolResultTexts:       input.ToolResultTexts,
					ToolResultsIncomplete: input.ToolResultScanIncomplete,
				}, input.PriorUserMessages)
			},
		},
		{
			config.SignalTypeLanguage, "Language",
			func(context.Context) { c.evaluateLanguageSignal(results, mu, textForSignal(config.SignalTypeLanguage)) },
		},
	}
}

func (c *Classifier) buildRequestFactSignalDispatchers(
	results *SignalResults,
	mu *sync.Mutex,
	contextText string,
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
	}
}

// buildApplicationSignalDispatchers asks the request-stage application
// rules, one dispatcher per type. Every type reads the request as it came;
// jailbreak, safety and pii also read what routing_input left out of it, and
// the earlier user turns as the client sent them, so a client cannot hide
// content from them; jailbreak reads the role-scoped guard content when the
// request carries it.
func (c *Classifier) buildApplicationSignalDispatchers(input SignalEvaluationInput, results *SignalResults, mu *sync.Mutex, textForSignal func(string) string, usedSignals map[string]bool) []signalDispatch {
	facts := input.RequestFacts
	reading := systemOneReading{
		request: textForSignal(systemOneQuestionText), history: input.PriorUserMessages, turns: facts.Turns,
		toolResults: input.ToolResultTexts, toolResultsIncomplete: input.ToolResultScanIncomplete,
		images: facts.Images, videos: facts.Videos,
	}
	guard := reading
	guard.request = safetyText(reading.request, facts.RoutingInput)
	guard.history = safetyPriorUserMessages(input.PriorUserMessages, facts.RoutingInput)
	jailbreak := guard
	if guardInput := facts.JailbreakInput; guardInput != nil {
		jailbreak.request = strings.Join(jailbreakInputTexts(guardInput.Current), "\n\n")
		jailbreak.history = jailbreakInputTexts(guardInput.History)
	}
	dispatch := func(signalType, name string, reading systemOneReading) signalDispatch {
		return signalDispatch{signalType, name, func(ctx context.Context) {
			c.evaluateApplicationSignals(ctx, results, mu, signalType, reading, usedSignals)
		}}
	}
	dispatchers := []signalDispatch{
		dispatch(config.SignalTypeDomain, "Domain", reading),
		dispatch(config.SignalTypeFactCheck, "Fact-check", reading),
		dispatch(config.SignalTypeComplexity, "Complexity", reading),
		dispatch(config.SignalTypeModality, "Modality", reading),
		dispatch(config.SignalTypeJailbreak, "Jailbreak", jailbreak),
		dispatch(config.SignalTypeSafety, "Safety", guard),
		dispatch(config.SignalTypePII, "PII", guard),
	}
	// The turn relation question relates the user turn to the turns before
	// it, so a first turn asks none.
	if len(facts.Turns) > 0 {
		dispatchers = append(dispatchers, dispatch(config.SignalTypeTurnRelation, "Turn relation", reading))
	}
	// The feedback question judges the answer the user turn replies to, so a
	// request without one asks none.
	if input.HasPriorAssistantReply {
		dispatchers = append(dispatchers, dispatch(config.SignalTypeUserFeedback, "User feedback", reading))
	}
	return dispatchers
}

// modelBackedSignalTypes call model deployments; their goroutines join the
// stage's request bundle, each declaring the deployments it asks questions,
// so a deployment's questions are sent once all of its askers have asked.
// Heuristic signals never delay a send.
var modelBackedSignalTypes = map[string]bool{
	config.SignalTypeSystemOne:    true,
	config.SignalTypeDomain:       true,
	config.SignalTypeFactCheck:    true,
	config.SignalTypeUserFeedback: true,
	config.SignalTypeTurnRelation: true,
	config.SignalTypeComplexity:   true,
	config.SignalTypeModality:     true,
	config.SignalTypeJailbreak:    true,
	config.SignalTypeSafety:       true,
	config.SignalTypePII:          true,
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
