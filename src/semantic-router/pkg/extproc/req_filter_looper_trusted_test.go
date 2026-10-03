package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// runTrustedFactsLooper executes a tool-bearing confidence Looper decision and
// returns how many upstream model calls carried tools. Confidence forwards the
// original tools to its models, unlike ratings or ReMoM which strip them.
func runTrustedFactsLooper(t *testing.T, stageRoles []string) (calls, callsWithTools int) {
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
	if _, err := router.handleLooperExecution(context.Background(), request, decision, ctx); err != nil {
		t.Fatalf("handleLooperExecution: %v", err)
	}
	return calls, callsWithTools
}

func TestHandleLooperExecutionTrustedFactsDeniesExcludedFinalStage(t *testing.T) {
	calls, withTools := runTrustedFactsLooper(t, []string{config.TrustedStageCandidate})
	if calls == 0 {
		t.Fatal("Looper made no upstream calls")
	}
	if withTools != 0 {
		t.Fatalf("policy excluding the final stage must strip tools before Looper, %d/%d calls carried tools", withTools, calls)
	}
}

func TestHandleLooperExecutionTrustedFactsAllowsAuthorizedFinalStage(t *testing.T) {
	calls, withTools := runTrustedFactsLooper(t, []string{config.TrustedStageFinal})
	if calls == 0 || withTools != calls {
		t.Fatalf("authorized final stage must keep tools, %d/%d calls carried tools", withTools, calls)
	}
}
