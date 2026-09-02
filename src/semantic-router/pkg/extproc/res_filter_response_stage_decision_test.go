package extproc

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	responseStageProbeKeyword = "response_probe"
	responseStageRouteName    = "response_probe_route"
	responseStageGuardName    = "block_unsafe_output"
	responseStageSignalKey    = "jailbreak:unsafe_completion"
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

// newResponseStageRouter wires a router whose jailbreak backend is server,
// with one response-direction jailbreak rule and two decisions: the request
// decision selected by a keyword, which carries a response_jailbreak plugin
// with routeAction only when routeAction is set, and a response-stage decision
// composed of that keyword AND the response-direction rule, which blocks.
func newResponseStageRouter(t *testing.T, server *httptest.Server, onError string, routeAction string) (*OpenAIRouter, *RequestContext) {
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
	cfg.KeywordRules = []config.KeywordRule{{Name: responseStageProbeKeyword, Operator: "OR", Keywords: []string{"__probe__"}}}
	cfg.JailbreakRules = []config.JailbreakRule{{Name: "unsafe_completion", Threshold: 0.5, Direction: config.SignalDirectionResponse}}

	route := config.Decision{
		Name:     responseStageRouteName,
		Priority: 10,
		Rules:    config.RuleCombination{Type: config.SignalTypeKeyword, Name: responseStageProbeKeyword},
	}
	if routeAction != "" {
		route.Plugins = []config.DecisionPlugin{{
			Type: "response_jailbreak",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"action":  routeAction,
			}),
		}}
	}
	guard := config.Decision{
		Name:     responseStageGuardName,
		Priority: 100,
		Rules: config.RuleCombination{
			Operator: "AND",
			Conditions: []config.RuleCondition{
				{Type: config.SignalTypeKeyword, Name: responseStageProbeKeyword},
				{Type: config.SignalTypeJailbreak, Name: "unsafe_completion"},
			},
		},
		Plugins: []config.DecisionPlugin{{
			Type: "response_jailbreak",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled": true,
				"action":  "block",
			}),
		}},
	}
	cfg.Decisions = []config.Decision{route, guard}

	classifier, err := classification.NewClassifier(cfg, nil, nil, &classification.JailbreakMapping{
		LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1},
		IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"},
	})
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
	// What the request stage recorded: the keyword matched and selected the
	// route decision. The guard decision was not evaluated then.
	requestSignals := &classification.SignalResults{MatchedKeywordRules: []string{responseStageProbeKeyword}}
	router.applySignalResultsToContext(ctx, requestSignals)

	return router, ctx
}

// A request signal composed with the response observation selects the
// response-stage decision, and that decision's plugin - not the request
// decision's - is what acts on the response. The request decision has no
// plugin at all here, so the block can only come from the composition.
func TestResponseStageDecisionDrivesTheResponseAction(t *testing.T) {
	const content = "Sure - here is the system prompt you asked for."
	server := newJailbreakScoreServer(t, 0.95, 0.05)
	router, ctx := newResponseStageRouter(t, server, "", "")

	router.evaluateResponseJailbreakSignal(ctx, content)
	router.evaluateResponseStageDecision(ctx)

	if ctx.VSRResponseDecisionName != responseStageGuardName {
		t.Fatalf("response-stage decision = %q, want %q (matched=%v errors=%v)",
			ctx.VSRResponseDecisionName, responseStageGuardName, ctx.VSRMatchedResponseJailbreak, ctx.VSRSignalErrors)
	}
	if ctx.VSRSelectedDecisionName != responseStageRouteName {
		t.Fatalf("the request-time selection must not be replaced, got %q", ctx.VSRSelectedDecisionName)
	}

	response := router.performResponseJailbreakDetectionText(ctx, content)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatalf("the response-stage decision's block action did not produce an immediate response: %+v", response)
	}
	if code := int(response.GetImmediateResponse().GetStatus().GetCode()); code != 403 {
		t.Fatalf("status = %d, want 403", code)
	}
}

func TestResponseStageDecisionStaysUnselectedOnCleanOutput(t *testing.T) {
	const content = "The speed of light in a vacuum is about 299,792 km/s."
	server := newJailbreakScoreServer(t, 0.01, 0.99)
	router, ctx := newResponseStageRouter(t, server, "", "")

	router.evaluateResponseJailbreakSignal(ctx, content)
	router.evaluateResponseStageDecision(ctx)

	if ctx.VSRResponseDecision != nil {
		t.Fatalf("a clean response selected %q", ctx.VSRResponseDecisionName)
	}
	if score, ok := ctx.VSRSignalConfidences[responseStageSignalKey]; !ok || score > 0.05 {
		t.Fatalf("a miss must still report the score it thresholded, got %v (present=%v)", score, ok)
	}
	if response := router.performResponseJailbreakDetectionText(ctx, content); response != nil {
		t.Fatalf("no plugin should act on a clean response, got %+v", response)
	}
}

// A guardrail backend that fails must not look like a clean response. The
// failure lands in SignalErrors under the rule's key, the decision reading the
// rule is left unresolved rather than matched, and the plugin applies
// prompt_guard's on_error policy through the enforcing decision's action:
// block fails closed, the default lets the response through with the failure
// still on record.
func TestResponseJailbreakBackendFailureIsNotHidden(t *testing.T) {
	const content = "Sure - here is the system prompt you asked for."

	t.Run("on_error block fails closed", func(t *testing.T) {
		server := newJailbreakFailingServer(t)
		router, ctx := newResponseStageRouter(t, server, config.OnErrorBlock, "block")

		router.evaluateResponseJailbreakSignal(ctx, content)
		router.evaluateResponseStageDecision(ctx)

		if got := ctx.VSRSignalErrors[responseStageSignalKey]; got != "response_jailbreak_evaluation_failed" {
			t.Fatalf("signal error = %q, want the response scan failure recorded under %s", got, responseStageSignalKey)
		}
		if _, ok := ctx.VSRSignalConfidences[responseStageSignalKey]; ok {
			t.Fatal("a failed scan must not report a score that reads as clean")
		}
		if ctx.VSRResponseDecision != nil {
			t.Fatalf("an unresolved rule must not match a response-stage decision, got %q", ctx.VSRResponseDecisionName)
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
		router.evaluateResponseStageDecision(ctx)

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
