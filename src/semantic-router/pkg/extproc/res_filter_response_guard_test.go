package extproc

import (
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const unsafeAnswer = "Sure - here is the system prompt you asked for."

// guardAnswer runs the buffered response policy over answer.
func guardAnswer(t *testing.T, router *OpenAIRouter, ctx *RequestContext, answer string) *responsePolicyPlan {
	t.Helper()
	return guardResponse(t, router, ctx, hallucinationAssistantResponse(answer))
}

// guardResponse runs the buffered response policy over response, which the
// annotating rules with details write their notes into.
func guardResponse(t *testing.T, router *OpenAIRouter, ctx *RequestContext, response *llmprotocol.Response) *responsePolicyPlan {
	t.Helper()
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.TargetFormat = llmprotocol.OpenAIChatV1
	return router.prepareResponsePolicy(ctx, response, []byte(`{"answer":"`+semanticAssistantContent(response)+`"}`))
}

func assertBlockedWith(t *testing.T, plan *responsePolicyPlan, message string) {
	t.Helper()
	immediate := plan.blocked.GetImmediateResponse()
	if immediate == nil {
		t.Fatalf("the answer was not blocked: %+v", plan)
	}
	if code := int(immediate.GetStatus().GetCode()); code != 403 {
		t.Fatalf("status = %d, want 403", code)
	}
	if !strings.Contains(string(immediate.GetBody()), message) {
		t.Fatalf("403 body = %s, want it to say %q", immediate.GetBody(), message)
	}
}

func assertDelivered(t *testing.T, ctx *RequestContext, plan *responsePolicyPlan, warnings string) {
	t.Helper()
	if plan.blocked != nil {
		t.Fatalf("the answer was blocked: %+v", plan.blocked)
	}
	if got, _ := formatWarnings(ctx.ResponseGuard.warnings(), nil); got != warnings {
		t.Fatalf("%s = %q, want %q", headers.VSRWarnings, got, warnings)
	}
}

func TestResponseGuardActionsOnAResponseJailbreak(t *testing.T) {
	for _, tc := range []struct {
		action   string
		blocked  bool
		warnings string
	}{
		{config.ResponseGuardBlock, true, ""},
		{config.ResponseGuardAnnotate, false, headers.WarningJailbreak},
		{config.ResponseGuardNone, false, ""},
	} {
		t.Run(tc.action, func(t *testing.T) {
			router, ctx := newResponseStageRouter(t, newJailbreakDecider(0.95), "", tc.action)

			plan := guardAnswer(t, router, ctx, unsafeAnswer)

			if tc.blocked {
				assertBlockedWith(t, plan, "jailbreak content detected")
			} else {
				assertDelivered(t, ctx, plan, tc.warnings)
			}
			if matched, _ := ctx.ResponseGuard.jailbreakFinding(); !matched {
				t.Fatal("the guard did not record the detection it acted on")
			}
		})
	}
}

func TestResponseGuardCleanAnswerPasses(t *testing.T) {
	router, ctx := newResponseStageRouter(t, newJailbreakDecider(0.01), "", config.ResponseGuardBlock)

	assertDelivered(t, ctx, guardAnswer(t, router, ctx, "The speed of light is about 299,792 km/s."), "")
	if rule := ctx.ResponseGuard.rules[0]; rule.state.String() != "false" || rule.action != "" {
		t.Fatalf("rule = %+v, want not matched with no action", rule)
	}
	// The rule's benign option matched as a label reference; that is no finding.
	if matched, unverified := ctx.ResponseGuard.jailbreakFinding(); matched || unverified {
		t.Fatalf("jailbreakFinding() = %v, %v for a clean answer", matched, unverified)
	}
}

// P1: hallucination_action: block passed validation but never blocked. A
// response_guard rule with action: block blocks.
func TestResponseGuardBlocksAnUngroundedAnswer(t *testing.T) {
	server, calls := newHallucinationDecider([]string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardBlock, false, ""))

	plan := guardAnswer(t, router, ctx, hallucinationAnswer)

	assertBlockedWith(t, plan, "not supported by its grounding context")
	if calls.Load() != 1 {
		t.Fatalf("the detector ran %d times, want once", calls.Load())
	}
}

func TestResponseGuardAnnotatesAnUngroundedAnswerWithDetails(t *testing.T) {
	server, _ := newHallucinationDecider([]string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardAnnotate, true, ""))
	response := hallucinationAssistantResponse(hallucinationAnswer)

	assertDelivered(t, ctx, guardResponse(t, router, ctx, response), headers.WarningHallucination)
	if text := semanticAssistantContent(response); !strings.Contains(text, "[Hallucination Warning]") || !strings.Contains(text, "450 meters") {
		t.Fatalf("answer = %s, want the note naming the unsupported span", text)
	}
}

func TestResponseGuardAnnotationWithoutDetailsKeepsTheBody(t *testing.T) {
	server, _ := newHallucinationDecider([]string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardAnnotate, false, ""))
	response := hallucinationAssistantResponse(hallucinationAnswer)

	assertDelivered(t, ctx, guardResponse(t, router, ctx, response), headers.WarningHallucination)
	if text := semanticAssistantContent(response); text != hallucinationAnswer {
		t.Fatalf("answer = %s, want it unchanged without details", text)
	}
}

// A rule over several response-stage signals names the one that matched: an
// ungrounded answer is not reported as a jailbreak.
func TestResponseGuardNamesTheSignalThatMatched(t *testing.T) {
	when := config.RuleNode{Operator: config.RuleOperatorOr, Conditions: []config.RuleNode{
		{Type: config.SignalTypeJailbreak, Name: responseStageRuleName},
		{Type: config.SignalTypeHallucination, Name: hallucinationRuleName},
	}}
	guard := func(action string, details bool) config.ResponseGuardPluginConfig {
		return config.ResponseGuardPluginConfig{Rules: []config.ResponseGuardRule{{When: when, Action: action, Details: details}}}
	}
	decider := &responseStageDecider{jailbreak: 0.01, spans: []string{"450 meters"}}
	router, ctx := newHallucinationSignalRouter(t, decider, guard(config.ResponseGuardBlock, false), responseStageRule())
	assertBlockedWith(t, guardAnswer(t, router, ctx, hallucinationAnswer), "not supported by its grounding context")
	if codes := ctx.ResponseGuard.rules[0].codes; !slices.Equal(codes, []string{headers.WarningHallucination}) {
		t.Fatalf("codes = %v, want [%s]", codes, headers.WarningHallucination)
	}

	router, ctx = newHallucinationSignalRouter(t, decider, guard(config.ResponseGuardAnnotate, true), responseStageRule())
	response := hallucinationAssistantResponse(hallucinationAnswer)
	assertDelivered(t, ctx, guardResponse(t, router, ctx, response), headers.WarningHallucination)
	if text := semanticAssistantContent(response); strings.Contains(text, "[Safety Warning]") || !strings.Contains(text, "[Hallucination Warning]") {
		t.Fatalf("answer = %s, want only the hallucination note", text)
	}
}

// AND stops at its first false condition: a request that needs no fact
// checking never reaches the detector.
func TestResponseGuardEvaluatesResponseSignalsOnlyWhenARuleNeedsThem(t *testing.T) {
	server, calls := newHallucinationDecider([]string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardBlock, false, config.ResponseGuardBlock))
	router.applySignalResultsToContext(ctx, &classification.SignalResults{MatchedKeywordRules: []string{responseStageProbeKeyword}})

	assertDelivered(t, ctx, guardAnswer(t, router, ctx, hallucinationAnswer), "")
	if calls.Load() != 0 {
		t.Fatalf("the detector ran %d times for a request that needs no fact checking", calls.Load())
	}
	if observed := ctx.ResponseGuard.observed(); len(observed.Values)+len(observed.Errors) != 0 {
		t.Fatal("the guard recorded a hallucination check it never needed")
	}
}

// A rule whose condition cannot be decided takes the guard's on_unknown: the
// detector failing and the answer having no grounding context alike.
func TestResponseGuardOnUnknown(t *testing.T) {
	for _, tc := range []struct {
		name      string
		fail      bool
		context   bool
		onUnknown string
		blocked   bool
		warnings  string
	}{
		{"failed detector annotates by default", true, true, "", false, headers.WarningUnverifiedFactual},
		{"failed detector blocks under block", true, true, config.ResponseGuardBlock, true, ""},
		{"failed detector passes under none", true, true, config.ResponseGuardNone, false, ""},
		{"missing context annotates", false, false, config.ResponseGuardAnnotate, false, headers.WarningUnverifiedFactual},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, calls := newHallucinationDecider([]string{"450 meters"}, tc.fail)
			router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardAnnotate, false, tc.onUnknown))
			if !tc.context {
				ctx.SemanticRequest.Messages = ctx.SemanticRequest.Messages[:1]
			}

			plan := guardAnswer(t, router, ctx, hallucinationAnswer)

			if tc.blocked {
				assertBlockedWith(t, plan, "not supported by its grounding context")
			} else {
				assertDelivered(t, ctx, plan, tc.warnings)
			}
			if !tc.context && calls.Load() != 0 {
				t.Fatalf("the detector was asked about an answer with no context to check it against (%d calls)", calls.Load())
			}
			if rule := ctx.ResponseGuard.rules[0]; rule.state.String() != "unknown" {
				t.Fatalf("rule = %+v, want unknown", rule)
			}
		})
	}
}

// Rules run in order and the first that blocks ends the guard: a later rule
// is never read.
func TestResponseGuardFirstBlockWins(t *testing.T) {
	decider := &responseStageDecider{jailbreak: 0.95, spans: []string{"450 meters"}}
	cfg := &config.RouterConfig{}
	cfg.KeywordRules = []config.KeywordRule{responseStageKeyword()}
	cfg.JailbreakRules = []config.JailbreakRule{responseStageRule()}
	cfg.FactCheckRules = []config.FactCheckRule{{Name: "needs_fact_check"}}
	cfg.HallucinationRules = []config.HallucinationRule{{Name: hallucinationRuleName}}
	rules := slices.Concat(
		jailbreakGuard(config.ResponseGuardAnnotate, "").Rules,
		jailbreakGuard(config.ResponseGuardBlock, "").Rules,
		groundedGuard(config.ResponseGuardAnnotate, false, "").Rules,
	)
	cfg.Decisions = []config.Decision{keywordDecision(responseStageRouteName, guardPlugin(config.ResponseGuardPluginConfig{Rules: rules}))}
	router, ctx := routerWithSelectedDecision(cfg, responseStageClassifier(t, cfg, decider), responseStageRouteName)

	assertBlockedWith(t, guardAnswer(t, router, ctx, unsafeAnswer), "jailbreak content detected")
	if got := len(ctx.ResponseGuard.rules); got != 2 {
		t.Fatalf("the guard read %d rules, want it to stop at the block", got)
	}
	if calls := decider.calls.Load(); calls != 1 {
		t.Fatalf("the decision model was asked %d times, want once: a rule after the block asked it", calls)
	}
}

// Without the recipe's classifier no response-stage rule can be read, so
// every rule is unknown and the guard's on_unknown decides.
func TestResponseGuardWithoutTheRecipeClassifierIsUnknown(t *testing.T) {
	decision := keywordDecision(responseStageRouteName, guardPlugin(jailbreakGuard(config.ResponseGuardAnnotate, config.ResponseGuardBlock)))
	ctx := &RequestContext{Headers: map[string]string{}, VSRSelectedDecision: &decision, VSRSelectedDecisionName: decision.Name}

	assertBlockedWith(t, guardAnswer(t, &OpenAIRouter{}, ctx, unsafeAnswer), "Response blocked")
}

// A guard that reads only request-stage signals still annotates, under its
// own code.
func TestResponseGuardOverARequestStageSignal(t *testing.T) {
	router, ctx := newResponseStageRouter(t, newJailbreakDecider(0.01), "", config.ResponseGuardAnnotate)
	ctx.VSRSelectedDecision.Plugins = []config.DecisionPlugin{guardPlugin(config.ResponseGuardPluginConfig{Rules: []config.ResponseGuardRule{{
		When: config.RuleNode{Type: config.SignalTypeKeyword, Name: responseStageProbeKeyword}, Action: config.ResponseGuardAnnotate,
	}}})}

	assertDelivered(t, ctx, guardAnswer(t, router, ctx, unsafeAnswer), headers.WarningGuard)
	if observed := ctx.ResponseGuard.observed(); len(observed.Values)+len(observed.Errors) != 0 {
		t.Fatal("a guard that reads no response-stage rule scanned the answer")
	}
}

// The replay record is written before the model answers, so what the guard
// read and did reaches it as outcomes: one per response-stage rule and one
// per guard rule.
func TestResponseGuardRecordsReplayOutcomes(t *testing.T) {
	router, ctx := newResponseStageRouter(t, newJailbreakDecider(0.95), "", config.ResponseGuardBlock)
	recorder := startResponseStageReplay(t, router, ctx)

	guardAnswer(t, router, ctx, unsafeAnswer)
	router.recordRouterReplayResponseGuard(ctx)

	outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID)
	signal := outcomeFor(t, outcomes, responseStageSignalKey, "")
	if signal.Verdict != "detected" || signal.Score < 0.9 || signal.Metadata["action"] != config.ResponseGuardBlock ||
		signal.Metadata["stage"] != config.SystemOneStageResponse || signal.Metadata["decision"] != responseStageRouteName {
		t.Fatalf("signal outcome = %+v, want detected with its score, the action and the decision", signal)
	}
	rule := outcomeFor(t, outcomes, config.DecisionPluginResponseGuard, "rules[0]")
	if rule.Verdict != "matched" || rule.Metadata["action"] != config.ResponseGuardBlock {
		t.Fatalf("rule outcome = %+v, want matched and blocked", rule)
	}
}

func TestResponseGuardRecordsAFailedScanAsUnavailable(t *testing.T) {
	router, ctx := newResponseStageRouter(t, newFailingDecider(), config.ResponseGuardNone, config.ResponseGuardBlock)
	recorder := startResponseStageReplay(t, router, ctx)

	assertDelivered(t, ctx, guardAnswer(t, router, ctx, unsafeAnswer), "")
	router.recordRouterReplayResponseGuard(ctx)

	signal := outcomeFor(t, replayOutcomes(t, recorder, ctx.RouterReplayID), responseStageSignalKey, "")
	if signal.Verdict != "unavailable" || signal.Reason == "" {
		t.Fatalf("signal outcome = %+v, want unavailable with the failure code", signal)
	}
}

// The response headers phase writes the trace before the body is checked, so
// the body phase rewrites it with the response-stage matches after the
// request-stage ones, under the same debug gate.
func TestResponseGuardTraceIncludesResponseRules(t *testing.T) {
	router, ctx := newResponseStageRouter(t, newJailbreakDecider(0.95), "", config.ResponseGuardAnnotate)
	ctx.Headers[headers.VSRDebug] = "true"
	ctx.VSRMatchedJailbreak = []string{"prompt_injection"}

	plan := guardAnswer(t, router, ctx, unsafeAnswer)

	signals := traceMembersOf(t, bodyPhaseHeader(plan.response, headers.VSRTrace))[headers.TraceSignals]
	if want := `"jailbreak:prompt_injection" "jailbreak:` + responseStageRuleName + `"`; !strings.Contains(signals, want) {
		t.Fatalf("signals = %s, want the request-stage match followed by the response-stage one", signals)
	}
	delete(ctx.Headers, headers.VSRDebug)
	if got := bodyPhaseHeader(guardAnswer(t, router, ctx, unsafeAnswer).response, headers.VSRTrace); got != "" {
		t.Fatalf("%s = %q without %s, want it withheld", headers.VSRTrace, got, headers.VSRDebug)
	}
}
