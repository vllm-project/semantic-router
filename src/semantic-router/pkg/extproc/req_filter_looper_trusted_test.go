package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

// Looper trusted facts are evaluated per model call: each hop carries its
// request-graph stage, and the hop pipeline gates that call's own request.

func trustedLooperHopRouter(t *testing.T, stageRoles []string) (*OpenAIRouter, *config.Decision) {
	t.Helper()
	toolsCfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  config.TrustedEnforcementAuthoritative,
		TrustSources: []string{config.TrustedSourceOperatorPolicy},
		StageRoles:   stageRoles,
	})
	decision := config.Decision{
		Name:      "trusted-looper",
		ModelRefs: []config.ModelRef{{Model: "panel-a"}},
		Plugins:   []config.DecisionPlugin{mustToolsDecisionPlugin(t, toolsCfg)},
	}
	router := &OpenAIRouter{
		Cache:         &spyCache{},
		ToolsDatabase: loadedTrustedFactsToolsDB(t),
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{decision}},
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{"panel-a": {PreferredEndpoints: []string{"panel-backend"}}},
				VLLMEndpoints: []config.VLLMEndpoint{{
					Name: "panel-backend", Address: "127.0.0.1", Port: 8000, Type: "vllm", Weight: 1,
				}},
			},
		},
	}
	router.CredentialResolver = newTestCredentialResolver(router.Config)
	return router, &router.Config.Decisions[0]
}

func TestLooperHopGatesToolsAtItsOwnStage(t *testing.T) {
	all := []string{config.TrustedStageCandidate, config.TrustedStageVerifier, config.TrustedStageAdvisor, config.TrustedStageFinal}
	cases := []struct {
		name      string
		stage     llmprotocol.TrustedStage
		roles     []string
		wantTools bool
		want      llmprotocol.TrustedOutcome
	}{
		{"candidate call under a final-only policy", llmprotocol.TrustedStageCandidate, []string{config.TrustedStageFinal}, false, llmprotocol.TrustedDeny},
		{"final call under a final-only policy", llmprotocol.TrustedStageFinal, []string{config.TrustedStageFinal}, true, llmprotocol.TrustedAllow},
		{"candidate call under a candidate-only policy", llmprotocol.TrustedStageCandidate, []string{config.TrustedStageCandidate}, true, llmprotocol.TrustedAllow},
		{"final call under a candidate-only policy", llmprotocol.TrustedStageFinal, []string{config.TrustedStageCandidate}, false, llmprotocol.TrustedDeny},
		{"verifier call outside its roles", llmprotocol.TrustedStageVerifier, []string{config.TrustedStageCandidate, config.TrustedStageFinal}, false, llmprotocol.TrustedDeny},
		{"unlabeled call", "", all, false, llmprotocol.TrustedDeny},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			router, decision := trustedLooperHopRouter(t, tc.roles)
			request := trustedFactsTestRequest()
			request.Model = "panel-a"
			ctx := &RequestContext{
				LooperRequest:       true,
				Hop:                 &routing.Hop{Decision: decision.Name, Stage: string(tc.stage)},
				VSRSelectedDecision: decision,
				SourceFormat:        llmprotocol.OpenAIChatV1,
				SemanticRequest:     request,
				Headers:             map[string]string{},
			}

			response, err := router.handleLooperInternalRequestWithPlugins("panel-a", ctx)
			require.NoError(t, err)
			var outbound map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &outbound))
			_, hasTools := outbound["tools"]
			require.Equal(t, tc.wantTools, hasTools, "provider-bound hop body: %s", outbound)
			if !tc.wantTools {
				require.NotContains(t, outbound, "tool_choice")
			}
			require.Len(t, ctx.pendingTrustedFactsOutcomes, 1)
			outcome := ctx.pendingTrustedFactsOutcomes[0]
			require.Equal(t, string(tc.want), outcome.Verdict)
			require.Equal(t, string(tc.stage), outcome.Metadata["stage"])
		})
	}
}

// trustedLooperClientRequest offers a function tool, so every Looper stage
// that forwards the client's tools has one to forward.
const trustedLooperClientRequest = `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"What is the capital of France?"}],` +
	`"tools":[{"type":"function","function":{"name":"lookup_capital","parameters":{"type":"object","properties":{}}}}]}`

func trustedLooperToolsPlugin(mode string, roles ...string) string {
	return fmt.Sprintf(`
      plugins:
        - type: tools
          configuration:
            enabled: true
            mode: %s
            trusted_facts:
              enabled: true
              enforcement: authoritative
              trust_sources: [operator-policy]
              stage_roles: [%s]`, mode, strings.Join(roles, ", "))
}

// trustedLooperAlgorithms are the tool-bearing Looper algorithms, with the
// stage of each model's calls in their fixtures. A model the map omits never
// receives tools: Fusion's panel runs without them whatever the policy,
// because Fusion keeps only a final call's tool use.
var trustedLooperAlgorithms = []struct {
	name     string
	decision string
	stageOf  map[string]string
}{
	{name: "confidence", stageOf: map[string]string{"model-a": config.TrustedStageCandidate, "model-b": config.TrustedStageCandidate}, decision: `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: confidence
        confidence:
          confidence_method: avg_logprob
          threshold: 0.6
          escalation_order: small_to_large`},
	{name: "fusion", stageOf: map[string]string{"judge": config.TrustedStageFinal}, decision: `
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: fusion
        fusion:
          model: judge
          analysis_models: [model-a, model-b, model-c]
          analysis_mode: one_call
          max_concurrent: 3
          min_successful_responses: 3
          on_error: skip`},
	{name: "workflows", stageOf: map[string]string{"model-a": config.TrustedStageCandidate, "model-b": config.TrustedStageCandidate, "judge": config.TrustedStageFinal}, decision: `
      modelRefs: [{model: model-a}, {model: model-b}, {model: judge}]
      algorithm:
        type: workflows
        workflows:
          mode: static
          roles:
            - {name: thinker, models: [model-a]}
            - {name: worker, models: [model-b]}
          final: {model: judge}`},
}

// Every call of a real request graph, served in process through the hop
// pipeline, receives tools only when its own stage is authorized: a
// final-only policy keeps them from candidate calls, and a candidate-only
// policy keeps them for candidates and away from the final call.
func TestLooperTrustedFactsFollowEachCallsStage(t *testing.T) {
	policies := map[string][]string{
		"final_only":     {config.TrustedStageFinal},
		"candidate_only": {config.TrustedStageCandidate},
	}
	for _, algorithm := range trustedLooperAlgorithms {
		for policy, roles := range policies {
			t.Run(algorithm.name+"/"+policy, func(t *testing.T) {
				calls := serveTrustedLooper(t, algorithm.decision+trustedLooperToolsPlugin(config.ToolsPluginModePassthrough, roles...))
				called := map[string]bool{}
				for _, call := range calls {
					called[call.Model] = true
					stage, carries := algorithm.stageOf[call.Model]
					want := carries && containsString(roles, stage)
					require.Equal(t, want, looperCallCarriesTools(t, call), "%s call (stage %q) under stage_roles %v", call.Model, stage, roles)
				}
				for model := range algorithm.stageOf {
					require.True(t, called[model], "fixture never called %s", model)
				}
			})
		}
	}
}

// Mode none holds on every call even when every stage is authorized and
// availability only narrows.
func TestLooperTrustedFactsKeepModeNoneOnEveryCall(t *testing.T) {
	for _, algorithm := range trustedLooperAlgorithms {
		t.Run(algorithm.name, func(t *testing.T) {
			plugin := trustedLooperToolsPlugin(config.ToolsPluginModeNone, config.TrustedStageCandidate, config.TrustedStageFinal)
			for _, call := range serveTrustedLooper(t, algorithm.decision+plugin) {
				require.False(t, looperCallCarriesTools(t, call), "%s call carried tools under mode none", call.Model)
			}
		})
	}
}

func serveTrustedLooper(t *testing.T, decision string) []looperCall {
	t.Helper()
	h := newLooperHarness(t, looperFixture{name: "trusted-looper", decision: decision, request: trustedLooperClientRequest})
	record := h.serve(t, looperAdapters["sessions"](h.router), trustedLooperClientRequest)
	require.Equal(t, http.StatusOK, record.Response.Status, "client response: %s", record.Response.Body)
	require.NotEmpty(t, record.Calls)
	return record.Calls
}

func looperCallCarriesTools(t *testing.T, call looperCall) bool {
	t.Helper()
	var body map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(call.Body, &body))
	_, ok := body["tools"]
	return ok
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

// stageRecordingHops records the stage of every hop it passes on.
type stageRecordingHops struct {
	inner  graph.Caller
	mu     sync.Mutex
	stages map[string]map[string]bool // decision -> stages
}

func (h *stageRecordingHops) Call(ctx context.Context, req *graph.HopRequest) (*graph.HopResponse, error) {
	h.mu.Lock()
	if h.stages[req.Hop.Decision] == nil {
		h.stages[req.Hop.Decision] = map[string]bool{}
	}
	h.stages[req.Hop.Decision][req.Hop.Stage] = true
	h.mu.Unlock()
	return h.inner.Call(ctx, req)
}

// Every hop that runs a decision's plugin chain names its stage, so no
// Looper call reaches the trusted-facts gate unlabeled. Only a helper call
// outside any decision, such as a prompt selector's, may name none.
func TestEveryLooperDecisionHopNamesItsStage(t *testing.T) {
	want := map[string][]string{
		"ratings":                {"candidate"},
		"ratings-partial-stream": {"candidate"},
		"confidence":             {"candidate"},
		"remom":                  {"candidate", "final"},
		"fusion-one-call":        {"candidate", "final"},
		"fusion-separate-quorum": {"advisor", "candidate", "final"},
		"workflows-static":       {"candidate", "final"},
	}
	for _, fixture := range looperFixtures {
		t.Run(fixture.name, func(t *testing.T) {
			h := newLooperHarness(t, fixture)
			recorder := &stageRecordingHops{inner: h.router.looperHops(), stages: map[string]map[string]bool{}}
			h.router.hopCaller = recorder
			h.serve(t, looperAdapters["sessions"](h.router), fixture.request)

			var stages []string
			for decision, seen := range recorder.stages {
				for stage := range seen {
					if decision == "" {
						continue
					}
					stages = append(stages, stage)
				}
			}
			sort.Strings(stages)
			require.Equal(t, want[fixture.name], stages, "stages of %s's decision hops", fixture.name)
		})
	}
}
