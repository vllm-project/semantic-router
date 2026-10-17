package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// runTrustedFactsLooper executes a tool-bearing confidence Looper decision,
// requires a successful response, checks the gate outcome reached Replay, and
// returns how many upstream model calls carried tools. Confidence forwards the
// original tools to its models, unlike ratings or ReMoM which strip them.
func runTrustedFactsLooper(t *testing.T, stageRoles []string, want llmprotocol.TrustedOutcome) (calls, callsWithTools int) {
	t.Helper()
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]json.RawMessage
		_ = json.NewDecoder(r.Body).Decode(&payload)
		mu.Lock()
		calls++
		if _, ok := payload["tools"]; ok {
			callsWithTools++
		}
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "chatcmpl-trusted", "object": "chat.completion", "created": 1, "model": "backend-model",
			"choices": []map[string]interface{}{{
				"index": 0, "message": map[string]interface{}{"role": "assistant", "content": "ok"}, "finish_reason": "stop",
				// Confidence scores each model by avg_logprob and fails without it.
				"logprobs": map[string]interface{}{"content": []map[string]interface{}{{"token": "ok", "logprob": -0.1}}},
			}},
			"usage": map[string]int{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
		})
	}))
	defer server.Close()

	toolsCfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  config.TrustedEnforcementAuthoritative,
		TrustSources: []string{config.TrustedSourceOperatorPolicy},
		StageRoles:   stageRoles,
	})
	decision := &config.Decision{
		Name:      "trusted-looper",
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{Type: "confidence"},
		Plugins:   []config.DecisionPlugin{mustToolsDecisionPlugin(t, toolsCfg)},
	}
	router := &OpenAIRouter{
		Config:         &config.RouterConfig{Looper: config.LooperConfig{Endpoint: server.URL}},
		ReplayRecorder: routerreplay.NewRecorder(store.NewMemoryStore(10, 0)),
		ToolsDatabase:  loadedTrustedFactsToolsDB(t),
	}
	replayConfig := config.DefaultRouterReplayPluginConfig()
	request := trustedFactsTestRequest()
	ctx := &RequestContext{
		RequestID:                "trusted-looper",
		Headers:                  map[string]string{},
		SourceFormat:             llmprotocol.OpenAIChatV1,
		SemanticRequest:          request,
		RouterReplayPluginConfig: &replayConfig,
		VSRSelectedDecision:      decision,
	}
	response, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
	if err != nil {
		t.Fatalf("handleLooperExecution: %v", err)
	}
	immediate := response.GetImmediateResponse()
	if code := immediate.GetStatus().GetCode(); code != typev3.StatusCode_OK {
		t.Fatalf("Looper status = %v, want OK; body: %s", code, immediate.GetBody())
	}
	requireTrustedFactsReplayOutcome(t, router.ReplayRecorder, ctx, want)
	return calls, callsWithTools
}

// A final-only policy must not send tools on Confidence's candidate attempts,
// even though those attempts are the ones that produce the answer.
func TestHandleLooperExecutionTrustedFactsDeniesCandidateAttemptsWhenOnlyFinalAuthorized(t *testing.T) {
	calls, withTools := runTrustedFactsLooper(t, []string{config.TrustedStageFinal}, llmprotocol.TrustedDeny)
	if calls == 0 {
		t.Fatal("Looper made no upstream calls")
	}
	if withTools != 0 {
		t.Fatalf("final-only policy must strip tools from candidate attempts, %d/%d calls carried tools", withTools, calls)
	}
}

// A Looper call that names no stage matches no configured role, so the gate
// denies its tools instead of guessing a stage for it.
func TestLooperToolStageGateDeniesUnlabeledAttempts(t *testing.T) {
	toolsCfg := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:      true,
		Enforcement:  config.TrustedEnforcementAuthoritative,
		TrustSources: []string{config.TrustedSourceOperatorPolicy},
		StageRoles:   []string{config.TrustedStageCandidate, config.TrustedStageFinal},
	})
	decision := &config.Decision{Name: "trusted-looper", Plugins: []config.DecisionPlugin{mustToolsDecisionPlugin(t, toolsCfg)}}
	router := &OpenAIRouter{ToolsDatabase: loadedTrustedFactsToolsDB(t)}
	gate := router.looperToolStageGate(trustedFactsTestRequest(), decision, &RequestContext{})
	if gate == nil {
		t.Fatal("trusted-facts decision produced no Looper gate")
	}
	var req openai.ChatCompletionNewParams
	if err := json.Unmarshal([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}]}`), &req); err != nil {
		t.Fatalf("decode request: %v", err)
	}

	if got := gate("", &req); len(got.Tools) != 0 {
		t.Fatalf("unlabeled attempt kept %d tools", len(got.Tools))
	}
	if got := gate(llmprotocol.TrustedStageFinal, &req); len(got.Tools) != 1 {
		t.Fatalf("authorized final attempt kept %d tools, want 1", len(got.Tools))
	}
}

func TestHandleLooperExecutionTrustedFactsAllowsAuthorizedCandidateAttempts(t *testing.T) {
	calls, withTools := runTrustedFactsLooper(t, []string{config.TrustedStageCandidate}, llmprotocol.TrustedAllow)
	if calls == 0 || withTools != calls {
		t.Fatalf("candidate-authorized policy must keep tools on candidate attempts, %d/%d calls carried tools", withTools, calls)
	}
}
