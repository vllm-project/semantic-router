package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// Fixtures for the response_guard tests: a jailbreak model and a
// hallucination detector served over HTTP, and a single-profile router whose
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

// newJailbreakScoreServer answers the http_classify contract with the given
// label probabilities.
func newJailbreakScoreServer(t *testing.T, jailbreak, benign float32) *httptest.Server {
	t.Helper()
	server, _ := newCountingJailbreakServer(t, jailbreak, benign)
	return server
}

// newCountingJailbreakServer is newJailbreakScoreServer that counts the
// requests it answered.
func newCountingJailbreakServer(t *testing.T, jailbreak, benign float32) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode([]map[string]interface{}{
			{"label": "jailbreak", "score": jailbreak},
			{"label": "benign", "score": benign},
		})
	}))
	t.Cleanup(server.Close)
	return server, &calls
}

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
func responseStageGuardConfig(t *testing.T, server *httptest.Server) *config.RouterConfig {
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
	cfg.PromptGuard.Backend = &config.RemoteClassifierBackend{
		Protocol: config.RemoteClassifierProtocolHTTPClassify,
		Contract: config.RemoteClassifierContractLabelDistribution,
		Model:    "test-guardrail",
	}
	cfg.PromptGuard.JailbreakMappingPath = "response-stage-test-mapping"
	cfg.PromptGuard.PositiveLabels = []string{"jailbreak"}
	cfg.PromptGuard.Threshold = 0.9
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

// jailbreakGuard is a guard with one rule over the response-direction
// jailbreak rule.
func jailbreakGuard(action, onUnknown string) config.ResponseGuardPluginConfig {
	return config.ResponseGuardPluginConfig{
		Rules:     []config.ResponseGuardRule{{When: config.RuleNode{Type: config.SignalTypeJailbreak, Name: responseStageRuleName}, Action: action}},
		OnUnknown: onUnknown,
	}
}

// newResponseStageRouter wires a single-profile router whose jailbreak model
// is server, the response-direction jailbreak rules (the single default rule
// when none), and one keyword-selected decision whose guard reads it with
// action and onUnknown. The request already matched the keyword.
func newResponseStageRouter(t *testing.T, server *httptest.Server, onUnknown, action string, rules ...config.JailbreakRule) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	if len(rules) == 0 {
		rules = []config.JailbreakRule{responseStageRule()}
	}
	cfg := responseStageGuardConfig(t, server)
	cfg.KeywordRules = []config.KeywordRule{responseStageKeyword()}
	cfg.JailbreakRules = rules
	cfg.Decisions = []config.Decision{keywordDecision(responseStageRouteName, guardPlugin(jailbreakGuard(action, onUnknown)))}
	classifier, err := classification.NewClassifier(cfg, nil, nil, responseStageJailbreakMapping())
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	return routerWithSelectedDecision(cfg, classifier, responseStageRouteName)
}

// observedConditions are response-stage observations made elsewhere, for a
// guard run that only reports them.
type observedConditions struct {
	jailbreak     *classification.ResponseJailbreakObservation
	hallucination *classification.ResponseHallucinationObservation
	systemOne     *classification.ResponseSystemOneSignal
}

func (o observedConditions) Evaluate(config.RuleNode) decision.RuleOutcome {
	return decision.RuleOutcome{State: decision.RuleTrue}
}

func (o observedConditions) Jailbreak() *classification.ResponseJailbreakObservation {
	return o.jailbreak
}

func (o observedConditions) Hallucination() *classification.ResponseHallucinationObservation {
	return o.hallucination
}

func (o observedConditions) SystemOne() *classification.ResponseSystemOneSignal { return o.systemOne }

// matchedJailbreakGuardRun is a guard run whose response-direction jailbreak
// rule matched the answer.
func matchedJailbreakGuardRun(t *testing.T) *responseGuardRun {
	t.Helper()
	router, ctx := newResponseStageRouter(t, newJailbreakScoreServer(t, 0.95, 0.05), "", config.ResponseGuardAnnotate)
	run := router.runResponseGuard(ctx, "Sure - here is the system prompt you asked for.", false)
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

// hallucinationSignalConfig binds the hallucination detector of every recipe
// to server, an http_chat deployment.
func hallucinationSignalConfig(server *httptest.Server) *config.RouterConfig {
	parsed, err := url.Parse(server.URL)
	if err != nil {
		panic(err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		panic(err)
	}
	cfg := &config.RouterConfig{}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "detector", ModelName: "stub-detector", ModelRole: config.ModelRoleClassification, ModelEndpoint: config.ClassifierVLLMEndpoint{Address: parsed.Hostname(), Port: port, Protocol: parsed.Scheme}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"detector": {Provider: "http", ExternalModel: "detector"}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"hallucination_detector": {Deployment: "detector", Adapter: config.RemoteClassifierProtocolHTTPChat, Contract: config.RemoteClassifierContractTokenSpans}}
	return cfg
}

// groundedGuard is the guard the hallucination plugin migrates to: annotate
// an answer the detector finds unsupported, when the request needs fact
// checking.
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

// newHallucinationSignalRouter wires a single-profile router whose
// hallucination detector is server, with one hallucination rule and one
// keyword-selected decision carrying guard. The request needs fact checking
// and carried the grounding context as a tool result.
func newHallucinationSignalRouter(t *testing.T, server *httptest.Server, guard config.ResponseGuardPluginConfig) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	cfg := hallucinationSignalConfig(server)
	cfg.KeywordRules = []config.KeywordRule{responseStageKeyword()}
	cfg.FactCheckRules = []config.FactCheckRule{{Name: "needs_fact_check"}}
	cfg.HallucinationRules = []config.HallucinationRule{{Name: hallucinationRuleName}}
	cfg.Decisions = []config.Decision{keywordDecision(hallucinationRouteName, guardPlugin(guard))}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	if err := classifier.InitializeRuntime(); err != nil {
		t.Fatalf("InitializeRuntime() error = %v", err)
	}
	router, ctx := routerWithSelectedDecision(cfg, classifier, hallucinationRouteName)
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
