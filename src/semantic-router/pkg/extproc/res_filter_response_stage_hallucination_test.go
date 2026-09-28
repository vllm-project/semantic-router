package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const (
	hallucinationRuleName      = "ungrounded_claims"
	hallucinationSignalKey     = "hallucination:ungrounded_claims"
	hallucinationRouteName     = "grounded_route"
	hallucinationGroundedModel = "vllm-sr/grounded"
	hallucinationContext       = "The Eiffel Tower was completed in 1889 and is 330 meters tall."
	hallucinationAnswer        = "The Eiffel Tower was completed in 1889 and is 450 meters tall."
)

// newHallucinationEndpointServer stands in for the endpoint detector: it
// answers the OpenAI chat contract with the given unsupported spans, or a
// server error when fail is set, and counts how often it was asked.
func newHallucinationEndpointServer(t *testing.T, spans []string, fail bool) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		if fail {
			http.Error(w, "detector unavailable", http.StatusInternalServerError)
			return
		}
		rawSpans := make([]map[string]string, 0, len(spans))
		for _, span := range spans {
			rawSpans = append(rawSpans, map[string]string{
				"text": span, "category": "contradiction", "subcategory": "numerical",
				"explanation": "the context gives a different value",
			})
		}
		content, _ := json.Marshal(map[string]interface{}{"hallucinated_spans": rawSpans})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"message": map[string]string{"content": string(content)}}},
		})
	}))
	t.Cleanup(server.Close)
	return server, &calls
}

// hallucinationSignalConfig points the hallucination detector at server over
// the endpoint backend.
func hallucinationSignalConfig(server *httptest.Server) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.HallucinationMitigation.HallucinationModel = config.HallucinationModelConfig{
		Backend:  config.HallucinationBackendEndpoint,
		Endpoint: server.URL + "/v1",
		ModelID:  "stub-detector",
	}
	return cfg
}

func hallucinationRule() config.HallucinationRule {
	return config.HallucinationRule{Name: hallucinationRuleName}
}

// hallucinationDecision is a keyword-selected decision carrying a
// hallucination plugin with action, or no plugin when action is empty.
func hallucinationDecision(name, action string) config.Decision {
	decision := config.Decision{
		Name:     name,
		Priority: 10,
		Rules:    config.RuleCombination{Type: config.SignalTypeKeyword, Name: responseStageProbeKeyword},
	}
	if action != "" {
		decision.Plugins = []config.DecisionPlugin{{
			Type: config.DecisionPluginHallucination,
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":                   true,
				"hallucination_action":      action,
				"unverified_factual_action": action,
			}),
		}}
	}
	return decision
}

// newHallucinationSignalRouter wires a single-profile router whose
// hallucination detector is server, with one hallucination rule and one
// decision selected at request time by a keyword, whose hallucination plugin
// carries action when it is set. The request carries tool context and the
// fact-check signal said the prompt needs grounding, which is the state the
// detector runs in.
func newHallucinationSignalRouter(t *testing.T, server *httptest.Server, action string) (*OpenAIRouter, *RequestContext) {
	t.Helper()

	cfg := hallucinationSignalConfig(server)
	cfg.KeywordRules = []config.KeywordRule{responseStageKeyword()}
	cfg.HallucinationRules = []config.HallucinationRule{hallucinationRule()}
	cfg.Decisions = []config.Decision{hallucinationDecision(hallucinationRouteName, action)}

	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	if !classifier.IsHallucinationDetectionEnabled() {
		t.Fatal("the declared rule did not provision the endpoint detector")
	}

	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	ctx := &RequestContext{
		TraceContext:            context.Background(),
		Headers:                 map[string]string{},
		VSRSelectedDecision:     &cfg.Decisions[0],
		VSRSelectedDecisionName: hallucinationRouteName,
		FactCheckNeeded:         true,
		HasToolsForFactCheck:    true,
		ToolResultsContext:      hallucinationContext,
		UserContent:             "How tall is the Eiffel Tower?",
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	router.applySignalResultsToContext(ctx, &classification.SignalResults{MatchedKeywordRules: []string{responseStageProbeKeyword}})
	return router, ctx
}

// runHallucinationStage runs the response pipeline's hallucination steps in
// order: the signal, the plugin's consumption, the unverified-factual mark,
// and the warning the plugin's action produces.
func runHallucinationStage(t *testing.T, router *OpenAIRouter, ctx *RequestContext, answer string) (warning string) {
	t.Helper()
	router.evaluateHallucinationSignal(ctx, answer)
	if response := router.performHallucinationDetectionText(ctx, answer); response != nil {
		t.Fatalf("the hallucination plugin does not block, got %+v", response)
	}
	router.markUnverifiedFactualResponse(ctx)
	semantic := hallucinationAssistantResponse(answer)
	if _, code := router.applySemanticHallucinationWarning(ctx, semantic); code != "" {
		return code
	}
	_, code := router.applySemanticUnverifiedFactualWarning(ctx, semantic)
	return code
}

// The rule checks the answer and the plugin of the decision selected for the
// request enforces on that observation. The detector runs once: the plugin
// reads the signal instead of classifying again.
func TestHallucinationSignalDrivesTheSelectedDecisionPlugin(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "header")

	warning := runHallucinationStage(t, router, ctx, hallucinationAnswer)

	if calls.Load() != 1 {
		t.Fatalf("the detector must run once per response, ran %d times", calls.Load())
	}
	if len(ctx.VSRMatchedHallucination) != 1 || ctx.VSRMatchedHallucination[0] != hallucinationRuleName {
		t.Fatalf("matched hallucination rules = %v, want [%s]", ctx.VSRMatchedHallucination, hallucinationRuleName)
	}
	if _, ok := ctx.VSRSignalConfidences[hallucinationSignalKey]; ok {
		t.Fatal("categorical endpoint must not invent confidence")
	}
	if !ctx.HallucinationDetected || len(ctx.HallucinationSpans) != 1 || ctx.HallucinationSpans[0] != "450 meters" {
		t.Fatalf("the plugin did not carry the evidence: detected=%v spans=%v", ctx.HallucinationDetected, ctx.HallucinationSpans)
	}
	if warning != headers.ResponseWarningHallucination {
		t.Fatalf("warning code = %q, want %q", warning, headers.ResponseWarningHallucination)
	}
}

func TestHallucinationSignalCleanAnswerTakesNoAction(t *testing.T) {
	server, _ := newHallucinationEndpointServer(t, nil, false)
	router, ctx := newHallucinationSignalRouter(t, server, "header")

	warning := runHallucinationStage(t, router, ctx, hallucinationAnswer)

	if len(ctx.VSRMatchedHallucination) != 0 || len(ctx.VSRSignalErrors) != 0 {
		t.Fatalf("a clean answer must match nothing and fail nothing, got matched=%v errors=%v", ctx.VSRMatchedHallucination, ctx.VSRSignalErrors)
	}
	if ctx.VSRHallucinationEvidence == nil || ctx.VSRHallucinationEvidence.ScoreAvailable {
		t.Fatal("a clean endpoint verdict must retain evidence without scores")
	}
	if ctx.HallucinationDetected || warning != "" {
		t.Fatalf("a clean answer is not a detection, got detected=%v warning=%q", ctx.HallucinationDetected, warning)
	}
}

// A detector that fails must not look like a clean answer: the failure lands
// in SignalErrors under the rule's key and the plugin takes no action on it.
func TestHallucinationSignalBackendFailureIsNotHidden(t *testing.T) {
	server, _ := newHallucinationEndpointServer(t, nil, true)
	router, ctx := newHallucinationSignalRouter(t, server, "header")

	warning := runHallucinationStage(t, router, ctx, hallucinationAnswer)

	if got := ctx.VSRSignalErrors[hallucinationSignalKey]; got != classification.HallucinationSignalFailedCode {
		t.Fatalf("signal error = %q, want %q", got, classification.HallucinationSignalFailedCode)
	}
	if _, ok := ctx.VSRSignalConfidences[hallucinationSignalKey]; ok {
		t.Fatal("a failed check must not report a confidence that reads as clean")
	}
	if ctx.HallucinationDetected || warning != "" {
		t.Fatalf("a detector failure is not a detection, got detected=%v warning=%q", ctx.HallucinationDetected, warning)
	}
}

// An answer with nothing to ground it against cannot be checked. The rule is
// unavailable with its own code, the detector is never asked, and the plugin's
// unverified_factual_action applies.
func TestHallucinationSignalWithoutContextIsUnavailable(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "header")
	ctx.HasToolsForFactCheck = false
	ctx.ToolResultsContext = ""

	warning := runHallucinationStage(t, router, ctx, hallucinationAnswer)

	if calls.Load() != 0 {
		t.Fatalf("nothing to check against, yet the detector was asked %d time(s)", calls.Load())
	}
	if got := ctx.VSRSignalErrors[hallucinationSignalKey]; got != classification.HallucinationSignalContextUnavailable {
		t.Fatalf("signal error = %q, want %q", got, classification.HallucinationSignalContextUnavailable)
	}
	if !ctx.UnverifiedFactualResponse || warning != headers.ResponseWarningUnverifiedFactual {
		t.Fatalf("an answer that could not be checked is unverified: marked=%v warning=%q", ctx.UnverifiedFactualResponse, warning)
	}
}

// When the request-stage fact-check signal said the prompt makes no claims
// worth grounding, the rule is not applicable: nothing is published and the
// detector is never asked.
func TestHallucinationSignalNotApplicableWithoutFactCheck(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "header")
	ctx.FactCheckNeeded = false

	warning := runHallucinationStage(t, router, ctx, hallucinationAnswer)

	if calls.Load() != 0 {
		t.Fatalf("fact-check not needed, yet the detector was asked %d time(s)", calls.Load())
	}
	if len(ctx.VSRSignalErrors) != 0 || len(ctx.VSRSignalConfidences) != 0 || len(ctx.VSRMatchedHallucination) != 0 {
		t.Fatalf("a rule that does not apply must stay unpublished, got errors=%v confidences=%v matched=%v",
			ctx.VSRSignalErrors, ctx.VSRSignalConfidences, ctx.VSRMatchedHallucination)
	}
	if ctx.HallucinationDetected || ctx.UnverifiedFactualResponse || warning != "" {
		t.Fatalf("nothing applies, got detected=%v unverified=%v warning=%q", ctx.HallucinationDetected, ctx.UnverifiedFactualResponse, warning)
	}
}

// The observation exists whether or not the selected decision carries a
// plugin: a decision without one leaves the evidence on record and acts on
// nothing.
func TestHallucinationSignalIsPublishedWithoutAPlugin(t *testing.T) {
	server, calls := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "")

	warning := runHallucinationStage(t, router, ctx, hallucinationAnswer)

	if calls.Load() != 1 || len(ctx.VSRMatchedHallucination) != 1 {
		t.Fatalf("the rule must be checked without a plugin: calls=%d matched=%v", calls.Load(), ctx.VSRMatchedHallucination)
	}
	if ctx.HallucinationDetected || warning != "" {
		t.Fatalf("no plugin, no action, got detected=%v warning=%q", ctx.HallucinationDetected, warning)
	}
}

// A named entrypoint's recipe declares the hallucination rule and the default
// recipe does not. The rule has to come from the recipe the request resolved
// to, the way the response-direction jailbreak rule does.
func TestHallucinationSignalReadsTheSelectedRecipeRules(t *testing.T) {
	server, _ := newHallucinationEndpointServer(t, []string{"450 meters"}, false)

	cfg := hallucinationSignalConfig(server)
	cfg.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{
			Signals:   config.Signals{KeywordRules: []config.KeywordRule{responseStageKeyword()}},
			Decisions: []config.Decision{hallucinationDecision("default_route", "")},
		}},
		{Name: "grounded", Profile: config.RoutingProfile{
			Signals: config.Signals{
				KeywordRules:       []config.KeywordRule{responseStageKeyword()},
				HallucinationRules: []config.HallucinationRule{hallucinationRule()},
			},
			Decisions: []config.Decision{hallucinationDecision("grounded_route", "header")},
		}},
	}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{hallucinationGroundedModel}, Recipe: "grounded"}}

	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildRecipeClassifiers() error = %v", err)
	}
	if err := classifiers.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}

	grounded := &RequestContext{
		TraceContext: context.Background(), Headers: map[string]string{},
		FactCheckNeeded: true, HasToolsForFactCheck: true, ToolResultsContext: hallucinationContext,
	}
	router.resolveEntrypointForRequest(hallucinationGroundedModel, grounded)
	recipe := grounded.Routing.SelectedRecipe()
	if recipe == nil || recipe.Name != "grounded" {
		t.Fatalf("named entrypoint resolved %+v, want the grounded recipe", recipe)
	}
	grounded.VSRSelectedDecision = &recipe.Profile.Decisions[0]
	grounded.VSRSelectedDecisionName = recipe.Profile.Decisions[0].Name

	if warning := runHallucinationStage(t, router, grounded, hallucinationAnswer); warning != headers.ResponseWarningHallucination {
		t.Fatalf("the grounded recipe's rule was not enforced: warning=%q matched=%v errors=%v",
			warning, grounded.VSRMatchedHallucination, grounded.VSRSignalErrors)
	}

	plain := &RequestContext{
		TraceContext: context.Background(), Headers: map[string]string{},
		FactCheckNeeded: true, HasToolsForFactCheck: true, ToolResultsContext: hallucinationContext,
	}
	router.resolveEntrypointForRequest(config.DefaultVSRAutoModelName, plain)
	if recipe := plain.Routing.SelectedRecipe(); recipe == nil || recipe.Name != config.DefaultRecipeName {
		t.Fatalf("auto model resolved %+v, want the default recipe", recipe)
	}
	plain.VSRSelectedDecision = &cfg.Recipes[0].Profile.Decisions[0]

	router.evaluateHallucinationSignal(plain, hallucinationAnswer)
	if len(plain.VSRMatchedHallucination) != 0 || len(plain.VSRSignalConfidences) != 0 || len(plain.VSRSignalErrors) != 0 {
		t.Fatalf("the default recipe declares no hallucination rule, yet its answer was checked: matched=%v confidences=%v errors=%v",
			plain.VSRMatchedHallucination, plain.VSRSignalConfidences, plain.VSRSignalErrors)
	}
}

// assertHallucinationOutcome checks one replay outcome's verdict and reason
// under the hallucination rule's key.
func assertHallucinationOutcome(t *testing.T, outcome routerreplay.Outcome, verdict, reason string) {
	t.Helper()
	if outcome.Target != hallucinationSignalKey || outcome.Verdict != verdict || outcome.Reason != reason {
		t.Fatalf("outcome = %+v, want verdict %q reason %q under %s", outcome, verdict, reason, hallucinationSignalKey)
	}
}

// Router Replay gets one outcome per hallucination rule: the verdict, the
// confidence, the span count and the plugin action; an answer that could not
// be checked carries its failure code, and a rule that did not apply says so.
func TestHallucinationSignalRecordsReplayOutcome(t *testing.T) {
	t.Run("detected", func(t *testing.T) {
		server, _ := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
		router, ctx := newHallucinationSignalRouter(t, server, "body")
		recorder := startResponseStageReplay(t, router, ctx)

		router.evaluateHallucinationSignal(ctx, hallucinationAnswer)
		router.recordRouterReplayHallucination(ctx)

		outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID)
		if len(outcomes) != 1 {
			t.Fatalf("outcomes = %+v, want one per hallucination rule", outcomes)
		}
		assertHallucinationOutcome(t, outcomes[0], "detected", "")
		metadata := outcomes[0].Metadata
		if outcomes[0].Score != 0 || metadata["score_available"] != "false" || metadata["spans"] != "1" || metadata["action"] != "body" || metadata["direction"] != config.SignalDirectionResponse {
			t.Fatalf("outcome = %+v, want unavailable score metadata, the span count, the plugin action and the response direction", outcomes[0])
		}
	})

	t.Run("unavailable and not applicable", func(t *testing.T) {
		server, _ := newHallucinationEndpointServer(t, nil, false)
		router, ctx := newHallucinationSignalRouter(t, server, "header")
		recorder := startResponseStageReplay(t, router, ctx)

		ctx.ToolResultsContext = ""
		router.evaluateHallucinationSignal(ctx, hallucinationAnswer)
		router.recordRouterReplayHallucination(ctx)

		other := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}, VSRSelectedDecision: ctx.VSRSelectedDecision}
		other.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
		other.RouterReplayID = ctx.RouterReplayID
		router.evaluateHallucinationSignal(other, hallucinationAnswer)
		router.recordRouterReplayHallucination(other)

		outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID)
		if len(outcomes) != 2 {
			t.Fatalf("outcomes = %+v, want one unavailable and one not applicable", outcomes)
		}
		assertHallucinationOutcome(t, outcomes[0], "unavailable", classification.HallucinationSignalContextUnavailable)
		assertHallucinationOutcome(t, outcomes[1], "not_applicable", "fact_check_not_needed")
	})
}

// With x-vsr-debug the body phase names the matched hallucination rules, the
// way it names the matched response-direction jailbreak rules.
func TestHallucinationMatchedHeaderIsWrittenInTheBodyPhase(t *testing.T) {
	server, _ := newHallucinationEndpointServer(t, []string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, "header")
	ctx.Headers[headers.VSRDebug] = "true"

	router.evaluateHallucinationSignal(ctx, hallucinationAnswer)
	response := buildResponseBodyContinueResponse(nil, nil)
	addResponseStageSignalHeaders(ctx, response)

	if got := bodyPhaseHeader(response, headers.VSRMatchedHallucination); got != hallucinationRuleName {
		t.Fatalf("%s = %q, want %q", headers.VSRMatchedHallucination, got, hallucinationRuleName)
	}
}

var _ = llmprotocol.OpenAIChatV1
