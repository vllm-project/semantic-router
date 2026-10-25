package extproc

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Neither client resends anything the Router counts as session history: Codex
// sends store=false without previous_response_id, and the Chat client sends one
// user message per request. With a shared prompt_cache_key, the second request
// must favor the model that served the first; without one, neither model wins.
func TestPromptCacheKeyKeepsCacheAffinityForStatelessClients(t *testing.T) {
	chatTurns := []json.RawMessage{
		json.RawMessage(`{"model":"vllm-sr/auto","prompt_cache_key":"shared-prefix-1","messages":[{"role":"system","content":"Long shared instructions."},{"role":"user","content":"First question"}]}`),
		json.RawMessage(`{"model":"vllm-sr/auto","prompt_cache_key":"shared-prefix-1","messages":[{"role":"system","content":"Long shared instructions."},{"role":"user","content":"Second question"}]}`),
	}
	clients := []struct {
		name   string
		format llmprotocol.WireFormat
		turns  []json.RawMessage
	}{
		{name: "codex responses tool loop", format: llmprotocol.OpenAIResponsesV1, turns: loadCodexToolLoopTurns(t)},
		{name: "stateless chat completions", format: llmprotocol.OpenAIChatV1, turns: chatTurns},
	}
	for _, client := range clients {
		for _, withKey := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/key=%t", client.name, withKey), func(t *testing.T) {
				affinity := secondTurnCacheAffinity(t, client.format, client.turns, withKey)
				t.Logf("second turn: %+v", affinity)
				favored := affinity.Adjustments["model-a"] > affinity.Adjustments["model-b"]
				if favored != withKey {
					t.Fatalf("model-a favored=%t, want %t: %+v", favored, withKey, affinity)
				}
			})
		}
	}
}

// secondTurnCacheAffinity sends every turn through Router ingress, records the
// first as served by model-a, and scores the last against equal base scores.
func secondTurnCacheAffinity(
	t *testing.T,
	format llmprotocol.WireFormat,
	turns []json.RawMessage,
	withKey bool,
) selection.CacheAffinityResult {
	t.Helper()
	sessiontelemetry.ResetForTesting()
	sessiontelemetry.ResetLastModelForTesting()
	t.Cleanup(sessiontelemetry.ResetForTesting)
	t.Cleanup(sessiontelemetry.ResetLastModelForTesting)

	router := &OpenAIRouter{}
	refs := []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}}
	var ctx *RequestContext
	for turn, body := range turns {
		if !withKey {
			body = withoutJSONField(t, body, "prompt_cache_key")
		}
		ctx = &RequestContext{SourceFormat: format, RequestID: fmt.Sprintf("turn-%d", turn+1), TraceContext: t.Context()}
		if request, immediate := router.prepareProtocolRequest(body, ctx); immediate != nil || request == nil {
			t.Fatalf("turn %d rejected at ingress: %+v", turn+1, ctx.ImmediateProtocolError)
		}
		if turn == 0 {
			ctx.RequestModel = "model-a"
			recordSessionTurn(ctx, responseUsageMetrics{promptTokens: 1200, completionTokens: 40}, sessiontelemetry.TurnPricing{})
		}
	}
	return selection.ComputeCacheAffinityAdjustments(
		router.buildCacheAffinityContext(ctx, refs),
		refs,
		map[string]float64{"model-a": 0.5, "model-b": 0.5},
	)
}

func loadCodexToolLoopTurns(t *testing.T) []json.RawMessage {
	t.Helper()
	raw, err := os.ReadFile("../protocolcodec/testdata/clients/codex-cli-0.156.1-tool-loop.json")
	if err != nil {
		t.Fatal(err)
	}
	var turns []json.RawMessage
	if err := json.Unmarshal(raw, &turns); err != nil {
		t.Fatal(err)
	}
	return turns
}

func withoutJSONField(t *testing.T, body json.RawMessage, field string) json.RawMessage {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatal(err)
	}
	delete(fields, field)
	stripped, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	return stripped
}
