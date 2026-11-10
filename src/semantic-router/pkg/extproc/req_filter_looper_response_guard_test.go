package extproc

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// looperGuardRouter is the response-stage router whose keyword-selected
// decision answers through Looper fusion, every hop answering unsafeAnswer,
// and carries guard and the memory context plugin, for a Chat client.
func looperGuardRouter(t *testing.T, guard config.ResponseGuardPluginConfig, stream bool) (*OpenAIRouter, *RequestContext, *llmprotocol.Request) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeFusionReplayCompletion(w, "judge", []map[string]interface{}{{"index": 0, "message": map[string]interface{}{"role": "assistant", "content": unsafeAnswer}, "finish_reason": "stop"}}, map[string]int64{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
	}))
	t.Cleanup(server.Close)
	cfg := &config.RouterConfig{Looper: config.LooperConfig{Endpoint: server.URL}}
	cfg.KeywordRules = []config.KeywordRule{responseStageKeyword()}
	cfg.JailbreakRules = []config.JailbreakRule{responseStageRule()}
	cfg.Memory = config.MemoryConfig{InUse: true, AutoStore: true}
	decision := keywordDecision(responseStageRouteName, guardPlugin(guard), memoryTestPlugin())
	decision.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeNone}}
	cfg.Decisions = []config.Decision{decision}
	router, ctx := routerWithSelectedDecision(cfg, responseStageClassifier(t, cfg, newJailbreakDecider(0.95)), responseStageRouteName)
	router.MemoryExtractor = memory.NewMemoryChunkStore(&noopMemoryStore{})
	router.ReplayRecorder = routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	replayConfig := config.DefaultRouterReplayPluginConfig()
	request := testNeutralRequest("router-entrypoint", "hello")
	request.Stream = stream
	ctx.RequestID, ctx.RouterReplayPluginConfig = "looper-guard", &replayConfig
	ctx.SourceFormat, ctx.SemanticRequest, ctx.ExpectStreamingResponse = llmprotocol.OpenAIChatV1, request, stream
	return router, ctx, request
}

// The response phase never sees a Looper answer, so the decision's guard reads
// it before the reply leaves: a blocked answer is refused, in stream form for a
// streaming client, an annotated one carries the guard's codes and note, and
// memory skips an answer the guard flagged.
func TestResponseGuardReadsTheLooperAnswerBeforeTheClient(t *testing.T) {
	annotate := jailbreakGuard(config.ResponseGuardAnnotate, "")
	annotate.Rules[0].Details = true
	for _, tc := range []struct {
		name   string
		guard  config.ResponseGuardPluginConfig
		stream bool
	}{
		{"block", jailbreakGuard(config.ResponseGuardBlock, ""), false},
		{"block/stream", jailbreakGuard(config.ResponseGuardBlock, ""), true},
		{"annotate", annotate, false},
		{"annotate/stream", annotate, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, ctx, request := looperGuardRouter(t, tc.guard, tc.stream)

			response, err := router.handleLooperExecution(ctx.TraceContext, request, ctx.VSRSelectedDecision, ctx)
			if err == nil {
				response, err = router.requestBodyReply(response, nil, ctx)
			}
			if err != nil {
				t.Fatalf("Looper request: %v", err)
			}

			immediate := response.GetImmediateResponse()
			client, status := string(immediate.GetBody()), int(immediate.GetStatus().GetCode())
			blocked := tc.guard.Rules[0].Action == config.ResponseGuardBlock
			switch {
			case blocked && tc.stream:
				if status != 200 {
					t.Fatalf("status = %d, want the stream's 200", status)
				}
				assertRefusedStream(t, llmprotocol.OpenAIChatV1, client, "jailbreak content detected", unsafeAnswer)
			case blocked:
				if status != 403 || !strings.Contains(client, "jailbreak content detected") || strings.Contains(client, unsafeAnswer) {
					t.Fatalf("reply = %d %s, want the 403 refusal and none of the answer", status, client)
				}
			default:
				if status != 200 || !strings.Contains(client, "[Safety Warning]") || !strings.Contains(client, "the system prompt you asked for") ||
					!strings.Contains(client, `"fusion":`) {
					t.Fatalf("reply = %d %s, want the answer with the guard's note and its fusion evidence", status, client)
				}
				if tc.stream && !strings.Contains(client, "data: [DONE]") {
					t.Fatalf("reply = %s, want the client's stream", client)
				}
				if warnings := immediateHeaderValue(response, headers.VSRWarnings); !strings.Contains(warnings, headers.WarningJailbreak) {
					t.Fatalf("%s = %q, want %q", headers.VSRWarnings, warnings, headers.WarningJailbreak)
				}
			}
			if err := router.ReplayRecorder.DrainOutcomes(); err != nil {
				t.Fatalf("drain replay outcomes: %v", err)
			}
			outcomes := replayOutcomes(t, router.ReplayRecorder, ctx.RouterReplayID)
			if rule := outcomeFor(t, outcomes, config.DecisionPluginResponseGuard, "rules[0]"); rule.Verdict != "matched" {
				t.Fatalf("guard outcome = %+v, want the rule matched", rule)
			}
			if receipt := outcomeFor(t, outcomes, "router", "memory_persistence"); receipt.Verdict != "policy_blocked" || receipt.Reason != "response_jailbreak" {
				t.Fatalf("memory receipt = %+v, want the flagged answer kept out of memory", receipt)
			}
		})
	}
}

// A Responses answer the guard refused is not retained for a later
// previous_response_id or GET, though a refused stream answers 200.
func TestLooperAnswerTheGuardRefusedIsNotRetained(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream_%t", stream), func(t *testing.T) {
			router, ctx, _ := looperGuardRouter(t, jailbreakGuard(config.ResponseGuardBlock, ""), stream)
			objectStore, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
			if err != nil {
				t.Fatalf("response store: %v", err)
			}
			t.Cleanup(func() { _ = objectStore.Close() })
			router.ResponseAPIFilter = NewResponseAPIFilter(objectStore)
			ctx.SourceFormat = llmprotocol.OpenAIResponsesV1
			request, rejected := router.prepareProtocolRequest([]byte(fmt.Sprintf(`{"model":"router-entrypoint","input":"hello","store":true,"stream":%t}`, stream)), ctx)
			if rejected != nil {
				t.Fatalf("request rejected: %v", rejected)
			}

			response, err := router.handleLooperExecution(ctx.TraceContext, request, ctx.VSRSelectedDecision, ctx)
			if err == nil {
				response, err = router.requestBodyReply(response, nil, ctx)
			}
			if err != nil {
				t.Fatalf("Looper request: %v", err)
			}

			if client := string(response.GetImmediateResponse().GetBody()); !strings.Contains(client, "jailbreak content detected") || strings.Contains(client, unsafeAnswer) {
				t.Fatalf("reply = %s, want the refusal and none of the answer", client)
			}
			if stored, err := objectStore.GetResponse(ctx.TraceContext, ctx.ResponseObjectState.GeneratedResponseID); err == nil {
				t.Fatalf("stored response = %+v, want the refused answer not retained", stored)
			}
		})
	}
}
