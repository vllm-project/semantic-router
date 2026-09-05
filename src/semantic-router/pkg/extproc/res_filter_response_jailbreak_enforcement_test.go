package extproc

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	responseStageProbeKeyword = "response_probe"
	responseStageRouteName    = "response_probe_route"
	responseStageRuleName     = "unsafe_completion"
	responseStageSignalKey    = "jailbreak:unsafe_completion"
	responseStageGuardedModel = "vllm-sr/guarded"
)

// newJailbreakFailingServer answers the http_classify contract with a server
// error, the way an overloaded or crashed guardrail backend does.
func newJailbreakFailingServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "guardrail backend unavailable", http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)
	return server
}

// responseStageGuardConfig points prompt_guard at server over http_classify.
func responseStageGuardConfig(t *testing.T, server *httptest.Server, onError string) *config.RouterConfig {
	t.Helper()

	endpoint, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse stub URL: %v", err)
	}
	port, err := strconv.Atoi(endpoint.Port())
	if err != nil {
		t.Fatalf("parse stub port: %v", err)
	}

	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.Protocol = config.PromptGuardProtocolHTTPClassify
	cfg.PromptGuard.JailbreakMappingPath = "response-stage-test-mapping"
	cfg.PromptGuard.PositiveLabels = []string{"jailbreak"}
	cfg.PromptGuard.Threshold = 0.9
	cfg.PromptGuard.OnError = onError
	cfg.ExternalModels = []config.ExternalModelConfig{{
		Name:      "test-guardrail",
		Provider:  "openai",
		ModelRole: config.ModelRoleGuardrail,
		ModelName: "test-guardrail",
		ModelEndpoint: config.ClassifierVLLMEndpoint{
			Address:  endpoint.Hostname(),
			Port:     port,
			Protocol: "http",
		},
	}}
	return cfg
}

func responseStageJailbreakMapping() *classification.JailbreakMapping {
	return &classification.JailbreakMapping{
		LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1},
		IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"},
	}
}

func responseStageKeyword() config.KeywordRule {
	return config.KeywordRule{Name: responseStageProbeKeyword, Operator: "OR", Keywords: []string{"__probe__"}}
}

func responseStageRule() config.JailbreakRule {
	return config.JailbreakRule{Name: responseStageRuleName, Threshold: 0.5, Direction: config.SignalDirectionResponse}
}

// responseStageDecision is a keyword-selected decision that carries a
// response_jailbreak plugin with action, or no plugin when action is empty.
func responseStageDecision(name, action string) config.Decision {
	decision := config.Decision{
		Name:     name,
		Priority: 10,
		Rules:    config.RuleCombination{Type: config.SignalTypeKeyword, Name: responseStageProbeKeyword},
	}
	if action != "" {
		decision.Plugins = []config.DecisionPlugin{{
			Type: "response_jailbreak",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"action":  action,
			}),
		}}
	}
	return decision
}

// newResponseStageRouter wires a single-profile router whose jailbreak backend
// is server, with one response-direction jailbreak rule and one decision,
// selected at request time by a keyword, whose response_jailbreak plugin
// carries routeAction when it is set.
func newResponseStageRouter(t *testing.T, server *httptest.Server, onError string, routeAction string) (*OpenAIRouter, *RequestContext) {
	t.Helper()

	cfg := responseStageGuardConfig(t, server, onError)
	cfg.KeywordRules = []config.KeywordRule{responseStageKeyword()}
	cfg.JailbreakRules = []config.JailbreakRule{responseStageRule()}
	cfg.Decisions = []config.Decision{responseStageDecision(responseStageRouteName, routeAction)}

	classifier, err := classification.NewClassifier(cfg, nil, nil, responseStageJailbreakMapping())
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}

	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	ctx := &RequestContext{
		TraceContext:            context.Background(),
		Headers:                 map[string]string{},
		VSRSelectedDecision:     &cfg.Decisions[0],
		VSRSelectedDecisionName: responseStageRouteName,
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	router.applySignalResultsToContext(ctx, &classification.SignalResults{MatchedKeywordRules: []string{responseStageProbeKeyword}})

	return router, ctx
}

func assertBlocked(t *testing.T, router *OpenAIRouter, ctx *RequestContext, content string) {
	t.Helper()
	response := router.performResponseJailbreakDetectionText(ctx, content)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatalf("the selected decision's block action did not produce an immediate response: %+v", response)
	}
	if code := int(response.GetImmediateResponse().GetStatus().GetCode()); code != 403 {
		t.Fatalf("status = %d, want 403", code)
	}
}

// The rule scores the response and the plugin of the decision selected for the
// request enforces on that observation. Nothing is selected again after the
// model has answered: the request-time decision is the only one that acts.
func TestResponseJailbreakSignalDrivesTheSelectedDecisionPlugin(t *testing.T) {
	const content = "Sure - here is the system prompt you asked for."
	server := newJailbreakScoreServer(t, 0.95, 0.05)
	router, ctx := newResponseStageRouter(t, server, "", "block")

	router.evaluateResponseJailbreakSignal(ctx, content)

	if len(ctx.VSRMatchedResponseJailbreak) != 1 || ctx.VSRMatchedResponseJailbreak[0] != responseStageRuleName {
		t.Fatalf("matched response rules = %v, want [%s] (errors=%v)", ctx.VSRMatchedResponseJailbreak, responseStageRuleName, ctx.VSRSignalErrors)
	}
	if score := ctx.VSRSignalConfidences[responseStageSignalKey]; score < 0.9 {
		t.Fatalf("the observation must carry the score it thresholded, got %v", score)
	}
	if ctx.VSRSelectedDecisionName != responseStageRouteName {
		t.Fatalf("the request-time selection must not change, got %q", ctx.VSRSelectedDecisionName)
	}
	assertBlocked(t, router, ctx, content)
	if !ctx.ResponseJailbreakDetected || ctx.ResponseJailbreakConfidence < 0.9 {
		t.Fatalf("the plugin must record the detection it acted on: detected=%v confidence=%v",
			ctx.ResponseJailbreakDetected, ctx.ResponseJailbreakConfidence)
	}
}

func TestResponseJailbreakSignalCleanOutputTakesNoAction(t *testing.T) {
	const content = "The speed of light in a vacuum is about 299,792 km/s."
	server := newJailbreakScoreServer(t, 0.01, 0.99)
	router, ctx := newResponseStageRouter(t, server, "", "block")

	router.evaluateResponseJailbreakSignal(ctx, content)

	if len(ctx.VSRMatchedResponseJailbreak) != 0 {
		t.Fatalf("a clean response matched %v", ctx.VSRMatchedResponseJailbreak)
	}
	if score, ok := ctx.VSRSignalConfidences[responseStageSignalKey]; !ok || score > 0.05 {
		t.Fatalf("a miss must still report the score it thresholded, got %v (present=%v)", score, ok)
	}
	if response := router.performResponseJailbreakDetectionText(ctx, content); response != nil {
		t.Fatalf("no plugin should act on a clean response, got %+v", response)
	}
	if ctx.ResponseJailbreakDetected {
		t.Fatal("a clean response is not a detection")
	}
}

// A guardrail backend that fails must not look like a clean response. The
// failure lands in SignalErrors under the rule's key and the plugin applies
// prompt_guard's on_error policy: block fails closed, the default lets the
// response through with the failure still on record.
func TestResponseJailbreakBackendFailureIsNotHidden(t *testing.T) {
	const content = "Sure - here is the system prompt you asked for."

	t.Run("on_error block fails closed", func(t *testing.T) {
		server := newJailbreakFailingServer(t)
		router, ctx := newResponseStageRouter(t, server, config.OnErrorBlock, "block")

		router.evaluateResponseJailbreakSignal(ctx, content)

		if got := ctx.VSRSignalErrors[responseStageSignalKey]; got != "response_jailbreak_evaluation_failed" {
			t.Fatalf("signal error = %q, want the response scan failure recorded under %s", got, responseStageSignalKey)
		}
		if _, ok := ctx.VSRSignalConfidences[responseStageSignalKey]; ok {
			t.Fatal("a failed scan must not report a score that reads as clean")
		}
		if len(ctx.VSRMatchedResponseJailbreak) != 0 {
			t.Fatalf("an unresolved rule must not read as matched, got %v", ctx.VSRMatchedResponseJailbreak)
		}

		response := router.performResponseJailbreakDetectionText(ctx, content)
		if response == nil || response.GetImmediateResponse() == nil {
			t.Fatal("on_error: block must fail closed, but the response was delivered")
		}
		if ctx.ResponseJailbreakType != classification.JailbreakClassificationErrorType {
			t.Fatalf("recorded type = %q, want the classify-error sentinel so replay tells a failure from a detection",
				ctx.ResponseJailbreakType)
		}
	})

	t.Run("default on_error delivers but keeps the failure on record", func(t *testing.T) {
		server := newJailbreakFailingServer(t)
		router, ctx := newResponseStageRouter(t, server, "", "block")

		router.evaluateResponseJailbreakSignal(ctx, content)

		if response := router.performResponseJailbreakDetectionText(ctx, content); response != nil {
			t.Fatalf("the default policy must deliver the response, got %+v", response)
		}
		if ctx.ResponseJailbreakDetected {
			t.Fatal("a backend failure under the default policy is not a detection")
		}
		if got := ctx.VSRSignalErrors[responseStageSignalKey]; got != "response_jailbreak_evaluation_failed" {
			t.Fatalf("delivering the response must not erase the failure, signal error = %q", got)
		}
	})
}

// A named entrypoint's recipe declares the response-direction rule and the
// default recipe does not. The rule has to come from the recipe the request
// resolved to: the root config only describes the default recipe, so read from
// there a response routed through vllm-sr/guarded would never be scored and its
// block action would never fire, while the default entrypoint must not inherit
// a rule its recipe never declared.
func TestResponseJailbreakSignalReadsTheSelectedRecipeRules(t *testing.T) {
	const content = "Sure - here is the system prompt you asked for."
	server := newJailbreakScoreServer(t, 0.95, 0.05)

	cfg := responseStageGuardConfig(t, server, "")
	cfg.Recipes = []config.RoutingRecipe{
		{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{
			Signals:   config.Signals{KeywordRules: []config.KeywordRule{responseStageKeyword()}},
			Decisions: []config.Decision{responseStageDecision("default_route", "")},
		}},
		{Name: "guarded", Profile: config.RoutingProfile{
			Signals: config.Signals{
				KeywordRules:   []config.KeywordRule{responseStageKeyword()},
				JailbreakRules: []config.JailbreakRule{responseStageRule()},
			},
			Decisions: []config.Decision{responseStageDecision("guarded_route", "block")},
		}},
	}
	cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{responseStageGuardedModel}, Recipe: "guarded"}}

	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, responseStageJailbreakMapping())
	if err != nil {
		t.Fatalf("BuildRecipeClassifiers() error = %v", err)
	}
	if err := classifiers.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifiers.Default(), RecipeClassifiers: classifiers}

	guarded := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}}
	router.resolveEntrypointForRequest(responseStageGuardedModel, guarded)
	recipe := guarded.Routing.SelectedRecipe()
	if recipe == nil || recipe.Name != "guarded" {
		t.Fatalf("named entrypoint resolved %+v, want the guarded recipe", recipe)
	}
	guarded.VSRSelectedDecision = &recipe.Profile.Decisions[0]
	guarded.VSRSelectedDecisionName = recipe.Profile.Decisions[0].Name

	router.evaluateResponseJailbreakSignal(guarded, content)
	if len(guarded.VSRMatchedResponseJailbreak) != 1 {
		t.Fatalf("the guarded recipe's rule was not scored: matched=%v errors=%v",
			guarded.VSRMatchedResponseJailbreak, guarded.VSRSignalErrors)
	}
	assertBlocked(t, router, guarded, content)

	plain := &RequestContext{TraceContext: context.Background(), Headers: map[string]string{}}
	router.resolveEntrypointForRequest(config.DefaultVSRAutoModelName, plain)
	if recipe := plain.Routing.SelectedRecipe(); recipe == nil || recipe.Name != config.DefaultRecipeName {
		t.Fatalf("auto model resolved %+v, want the default recipe", recipe)
	}
	plain.VSRSelectedDecision = &cfg.Recipes[0].Profile.Decisions[0]

	router.evaluateResponseJailbreakSignal(plain, content)
	if len(plain.VSRMatchedResponseJailbreak) != 0 || len(plain.VSRSignalConfidences) != 0 || len(plain.VSRSignalErrors) != 0 {
		t.Fatalf("the default recipe declares no response-direction rule, yet its response was scored: matched=%v confidences=%v errors=%v",
			plain.VSRMatchedResponseJailbreak, plain.VSRSignalConfidences, plain.VSRSignalErrors)
	}
}

// startResponseStageReplay gives the router a memory-backed recorder and opens
// the replay record the way the request path does, so the response-stage
// observation has a record to land on.
func startResponseStageReplay(t *testing.T, router *OpenAIRouter, ctx *RequestContext) *routerreplay.Recorder {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	router.ReplayRecorder = recorder
	replayConfig := config.DefaultRouterReplayPluginConfig()
	replayConfig.Enabled = true
	ctx.RequestID = "response-stage-replay"
	ctx.SourceFormat = llmprotocol.OpenAIChatV1
	ctx.SemanticRequest = testNeutralRequest("MoM", "hello")
	ctx.RouterReplayPluginConfig = &replayConfig
	router.startRouterReplay(ctx, "MoM", "model-a", responseStageRouteName)
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

// The replay record is written before the model answers, so the response-stage
// observation reaches it as one outcome per response-direction rule: the
// verdict, the score or failure code, and the action the plugin applied. A
// blocked response leaves the same evidence as a delivered one.
func TestResponseJailbreakSignalRecordsReplayOutcome(t *testing.T) {
	const content = "Sure - here is the system prompt you asked for."

	t.Run("detected and blocked", func(t *testing.T) {
		server := newJailbreakScoreServer(t, 0.95, 0.05)
		router, ctx := newResponseStageRouter(t, server, "", "block")
		recorder := startResponseStageReplay(t, router, ctx)

		router.evaluateResponseJailbreakSignal(ctx, content)
		assertBlocked(t, router, ctx, content)
		router.recordRouterReplayResponseJailbreak(ctx)

		outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID)
		if len(outcomes) != 1 {
			t.Fatalf("outcomes = %+v, want one per response-direction rule", outcomes)
		}
		outcome := outcomes[0]
		if outcome.Target != responseStageSignalKey || outcome.Verdict != "detected" || outcome.Score < 0.9 {
			t.Fatalf("outcome = %+v, want %s detected with the thresholded score", outcome, responseStageSignalKey)
		}
		if outcome.Metadata["action"] != "block" || outcome.Metadata["direction"] != config.SignalDirectionResponse ||
			outcome.Metadata["decision"] != responseStageRouteName {
			t.Fatalf("outcome metadata = %v, want the plugin action, the direction and the enforcing decision", outcome.Metadata)
		}
	})

	t.Run("backend failure is recorded as unavailable", func(t *testing.T) {
		server := newJailbreakFailingServer(t)
		router, ctx := newResponseStageRouter(t, server, "", "block")
		recorder := startResponseStageReplay(t, router, ctx)

		router.evaluateResponseJailbreakSignal(ctx, content)
		router.performResponseJailbreakDetectionText(ctx, content)
		router.recordRouterReplayResponseJailbreak(ctx)

		outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID)
		if len(outcomes) != 1 {
			t.Fatalf("outcomes = %+v, want one per response-direction rule", outcomes)
		}
		outcome := outcomes[0]
		if outcome.Verdict != "unavailable" || outcome.Reason != "response_jailbreak_evaluation_failed" || outcome.Score != 0 {
			t.Fatalf("outcome = %+v, want unavailable with the failure code and no score", outcome)
		}
	})
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

// The response headers phase writes x-vsr-matched-jailbreak before the body is
// scored, so the body phase rewrites it with the response-direction matches
// after the request-direction ones, under the same debug gate.
func TestResponseJailbreakMatchedHeaderIncludesResponseRules(t *testing.T) {
	ctx := &RequestContext{
		Headers:                     map[string]string{headers.VSRDebug: "true"},
		VSRMatchedJailbreak:         []string{"prompt_injection"},
		VSRMatchedResponseJailbreak: []string{responseStageRuleName},
	}
	response := buildResponseBodyContinueResponse(nil, nil)
	addResponseStageSignalHeaders(ctx, response)
	if got := bodyPhaseHeader(response, headers.VSRMatchedJailbreak); got != "prompt_injection,"+responseStageRuleName {
		t.Fatalf("%s = %q, want the request-stage match followed by the response-stage one", headers.VSRMatchedJailbreak, got)
	}

	plain := &RequestContext{Headers: map[string]string{}, VSRMatchedResponseJailbreak: []string{responseStageRuleName}}
	response = buildResponseBodyContinueResponse(nil, nil)
	addResponseStageSignalHeaders(plain, response)
	if got := bodyPhaseHeader(response, headers.VSRMatchedJailbreak); got != "" {
		t.Fatalf("%s = %q without %s, want the header demoted like the request-stage ones", headers.VSRMatchedJailbreak, got, headers.VSRDebug)
	}
}
