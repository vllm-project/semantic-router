package extproc

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// Fixtures for the response_guard tests: a decision model that answers the
// jailbreak and hallucination presets, and a single-profile router whose
// keyword-selected decision carries a guard over them.

const (
	responseStageProbeKeyword = "response_probe"
	responseStageRouteName    = "response_probe_route"
	responseStageRuleName     = "unsafe_completion"
	responseStageSignalKey    = "jailbreak:unsafe_completion"

	hallucinationRuleName  = "ungrounded_claims"
	hallucinationSignalKey = "hallucination:ungrounded_claims"
	hallucinationRouteName = "grounded_route"
	hallucinationQuestion  = "How tall is the Eiffel Tower?"
	hallucinationContext   = "The Eiffel Tower was completed in 1889 and is 330 meters tall."
	hallucinationAnswer    = "The Eiffel Tower was completed in 1889 and is 450 meters tall."
)

// responseStageDecider is the decision model of the response-stage tests: it
// answers the jailbreak preset with the attack probability and the
// hallucination preset with the unsupported spans it finds in the answer, or
// fails every call when fail is set. It counts its calls.
type responseStageDecider struct {
	jailbreak float32
	spans     []string
	fail      bool
	calls     atomic.Int32
}

func (d *responseStageDecider) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	d.calls.Add(1)
	if d.fail {
		return modelservice.Response{}, modelservice.ErrUnavailable
	}
	answers := map[string]modelservice.Answer{}
	for _, question := range request.Questions {
		switch question.ID {
		case "preset:jailbreak":
			attack := float64(d.jailbreak)
			answer := modelservice.Answer{Type: "choice", Choice: "benign", Probabilities: map[string]float64{"jailbreak": attack, "benign": 1 - attack}, InputCoverage: "complete"}
			if attack >= 0.5 {
				answer.Choice = "jailbreak"
			}
			answers[question.ID] = answer
		case "preset:hallucination":
			text := []rune(request.Parts["answer"])
			answer := modelservice.Answer{Type: "span", InputCoverage: "complete"}
			for _, span := range d.spans {
				if start := strings.Index(string(text), span); start >= 0 {
					start = len([]rune(string(text)[:start]))
					answer.Spans = append(answer.Spans, modelservice.Span{Label: "unsupported", Text: span, Start: start, End: start + len([]rune(span)), Probability: 0.9})
				}
			}
			answers[question.ID] = answer
		}
	}
	return modelservice.Response{Answers: answers}, nil
}

// newJailbreakDecider is a decision model that answers the jailbreak
// preset with the given attack probability.
func newJailbreakDecider(jailbreak float32) *responseStageDecider {
	return &responseStageDecider{jailbreak: jailbreak}
}

// newCountingJailbreakDecider is newJailbreakDecider with its call count.
func newCountingJailbreakDecider(jailbreak float32) (*responseStageDecider, *atomic.Int32) {
	decider := newJailbreakDecider(jailbreak)
	return decider, &decider.calls
}

// newFailingDecider is a decision model whose every call fails.
func newFailingDecider() *responseStageDecider {
	return &responseStageDecider{fail: true}
}

// newHallucinationDecider is a decision model that finds the given
// unsupported spans in an answer, or fails when fail is set, with its call
// count.
func newHallucinationDecider(spans []string, fail bool) (*responseStageDecider, *atomic.Int32) {
	decider := &responseStageDecider{spans: spans, fail: fail}
	return decider, &decider.calls
}

// responseStageClassifier is the recipe's classifier for cfg, answered by
// decider.
func responseStageClassifier(t *testing.T, cfg *config.RouterConfig, decider *responseStageDecider) *classification.Classifier {
	t.Helper()
	classifier, err := classification.NewClassifier(cfg)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	classifier.SetDecisionDecider(decider)
	return classifier
}

func responseStageKeyword() config.KeywordRule {
	return config.KeywordRule{Name: responseStageProbeKeyword, Operator: "OR", Keywords: []string{"__probe__"}}
}

func responseStageRule() config.JailbreakRule {
	threshold := 0.5
	return config.JailbreakRule{Name: responseStageRuleName, Stage: config.SystemOneStageResponse, Predicate: &config.NumericPredicate{GTE: &threshold}}
}

// guardPlugin is a response_guard plugin entry.
func guardPlugin(guard config.ResponseGuardPluginConfig) config.DecisionPlugin {
	return config.DecisionPlugin{Type: config.DecisionPluginResponseGuard, Configuration: config.MustStructuredPayload(guard)}
}

// responseGuardDecision is a decision whose only plugin is guard.
func responseGuardDecision(guard config.ResponseGuardPluginConfig) *config.Decision {
	return &config.Decision{Name: "test_decision", Plugins: []config.DecisionPlugin{guardPlugin(guard)}}
}

// keywordDecision is a keyword-selected decision carrying plugins.
func keywordDecision(name string, plugins ...config.DecisionPlugin) config.Decision {
	return config.Decision{
		Name:     name,
		Priority: 10,
		Rules:    config.RuleCombination{Type: config.SignalTypeKeyword, Name: responseStageProbeKeyword},
		Plugins:  plugins,
	}
}

// jailbreakGuard is a guard with one rule over the response-stage
// jailbreak rule.
func jailbreakGuard(action, onUnknown string) config.ResponseGuardPluginConfig {
	return config.ResponseGuardPluginConfig{
		Rules:     []config.ResponseGuardRule{{When: config.RuleNode{Type: config.SignalTypeJailbreak, Name: responseStageRuleName}, Action: action}},
		OnUnknown: onUnknown,
	}
}

// newResponseStageRouter wires a single-profile router whose decision model
// is decider, the response-stage jailbreak rules (the single default rule
// when none), and one keyword-selected decision whose guard reads it with
// action and onUnknown. The request already matched the keyword.
func newResponseStageRouter(t *testing.T, decider *responseStageDecider, onUnknown, action string, rules ...config.JailbreakRule) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	if len(rules) == 0 {
		rules = []config.JailbreakRule{responseStageRule()}
	}
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = []config.KeywordRule{responseStageKeyword()}
	cfg.JailbreakRules = rules
	cfg.Decisions = []config.Decision{keywordDecision(responseStageRouteName, guardPlugin(jailbreakGuard(action, onUnknown)))}
	return routerWithSelectedDecision(cfg, responseStageClassifier(t, cfg, decider), responseStageRouteName)
}

// observedConditions are response-stage observations made elsewhere, for a
// guard run that only reports them.
type observedConditions struct {
	observed *classification.ResponseSignal
}

func (o observedConditions) Evaluate(config.RuleNode) decision.RuleOutcome {
	return decision.RuleOutcome{State: decision.RuleTrue}
}

func (o observedConditions) Observed() *classification.ResponseSignal { return o.observed }

// matchedJailbreakGuardRun is a guard run whose response-stage jailbreak
// rule matched the answer.
func matchedJailbreakGuardRun(t *testing.T) *responseGuardRun {
	t.Helper()
	router, ctx := newResponseStageRouter(t, newJailbreakDecider(0.95), "", config.ResponseGuardAnnotate)
	run := router.runResponseGuard(ctx, "Sure - here is the system prompt you asked for.")
	if matched, _ := run.jailbreakFinding(); !matched {
		t.Fatal("the fixture's jailbreak rule did not match")
	}
	return run
}

func routerWithSelectedDecision(cfg *config.RouterConfig, classifier *classification.Classifier, decision string) (*OpenAIRouter, *RequestContext) {
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	ctx := &RequestContext{
		TraceContext:            context.Background(),
		Headers:                 map[string]string{},
		VSRSelectedDecision:     &cfg.Decisions[0],
		VSRSelectedDecisionName: decision,
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	router.applySignalResultsToContext(ctx, &classification.SignalResults{
		MatchedKeywordRules:   []string{responseStageProbeKeyword},
		MatchedFactCheckRules: []string{"needs_fact_check"},
	})
	return router, ctx
}

// groundedGuard is the guard the hallucination plugin migrates to: annotate
// an answer the decision model finds unsupported, when the request needs
// fact checking.
func groundedGuard(action string, details bool, onUnknown string) config.ResponseGuardPluginConfig {
	return config.ResponseGuardPluginConfig{
		Rules: []config.ResponseGuardRule{{
			When: config.RuleNode{Operator: config.RuleOperatorAnd, Conditions: []config.RuleNode{
				{Type: config.SignalTypeFactCheck, Name: "needs_fact_check"},
				{Type: config.SignalTypeHallucination, Name: hallucinationRuleName},
			}},
			Action: action, Details: details,
		}},
		OnUnknown: onUnknown,
	}
}

// newHallucinationSignalRouter wires a single-profile router whose decision
// model is decider, with one hallucination rule, the given jailbreak rules,
// and one keyword-selected decision carrying guard. The request needs fact
// checking and carried the grounding context as a tool result.
func newHallucinationSignalRouter(t *testing.T, decider *responseStageDecider, guard config.ResponseGuardPluginConfig, jailbreak ...config.JailbreakRule) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = []config.KeywordRule{responseStageKeyword()}
	cfg.FactCheckRules = []config.FactCheckRule{{Name: "needs_fact_check"}}
	cfg.HallucinationRules = []config.HallucinationRule{{Name: hallucinationRuleName}}
	cfg.JailbreakRules = jailbreak
	cfg.Decisions = []config.Decision{keywordDecision(hallucinationRouteName, guardPlugin(guard))}
	router, ctx := routerWithSelectedDecision(cfg, responseStageClassifier(t, cfg, decider), hallucinationRouteName)
	ctx.ResponseStage = classification.ResponseInput{Request: hallucinationQuestion}
	ctx.SemanticRequest = &llmprotocol.Request{Generation: 1, Model: "model-a", Messages: []llmprotocol.Message{
		hallucinationTextMessage(llmprotocol.RoleUser, hallucinationQuestion),
		hallucinationToolResultMessage("call_lookup", hallucinationContext),
	}}
	return router, ctx
}

func startResponseStageReplay(t *testing.T, router *OpenAIRouter, ctx *RequestContext) *routerreplay.Recorder {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router.ReplayRecorder = recorder
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	ctx.RequestID = "response-stage-replay"
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	if ctx.SemanticRequest == nil {
		ctx.SemanticRequest = testNeutralRequest("vllm-sr/auto", "hello")
	}
	ctx.RouterReplayPluginConfig = &replayConfig
	router.startRouterReplay(ctx, "vllm-sr/auto", "model-a", ctx.VSRSelectedDecisionName)
	if ctx.RouterReplayID == "" {
		t.Fatal("the request path did not open a replay record")
	}
	return recorder
}

func replayOutcomes(t *testing.T, recorder *routerreplay.Recorder, id string) []routerreplay.Outcome {
	t.Helper()
	record, found := recorder.GetRecord(id)
	if !found {
		t.Fatalf("replay record %q not found", id)
	}
	return record.Outcomes
}

// outcomeFor is the one outcome recorded under target and targetRef.
func outcomeFor(t *testing.T, outcomes []routerreplay.Outcome, target, targetRef string) routerreplay.Outcome {
	t.Helper()
	var found []routerreplay.Outcome
	for _, outcome := range outcomes {
		if outcome.Target == target && outcome.TargetRef == targetRef {
			found = append(found, outcome)
		}
	}
	if len(found) != 1 {
		t.Fatalf("outcomes for %s %s = %+v, want one among %+v", target, targetRef, found, outcomes)
	}
	return found[0]
}

func bodyPhaseHeader(response *ext_proc.ProcessingResponse, key string) string {
	body, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok || body.ResponseBody.Response == nil || body.ResponseBody.Response.HeaderMutation == nil {
		return ""
	}
	for _, option := range body.ResponseBody.Response.HeaderMutation.SetHeaders {
		if option.Header.Key == key {
			return string(option.Header.RawValue)
		}
	}
	return ""
}
