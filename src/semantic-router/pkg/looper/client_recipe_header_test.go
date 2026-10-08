package looper

import (
	"context"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
)

func TestExecuteWithLatencyCarriesTheParentRecipeOnEveryHop(t *testing.T) {
	caller := &hopCaller{resp: &graph.HopResponse{Status: 200, Body: []byte(
		`{"id":"chatcmpl-recipe","object":"chat.completion","created":1,"model":"model-a",` +
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)}}
	request := &Request{
		OriginalRequest: &openai.ChatCompletionNewParams{
			Model:    "entrypoint",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
		},
		ModelRefs:    []config.ModelRef{{Model: "model-a"}},
		DecisionName: "shared-route",
		RecipeName:   "named-recipe",
	}
	cfg := &config.LooperConfig{}
	looper := newBaseLooper(cfg, borrowClient(NewHopClient(cfg, caller)))
	response, err := ExecuteWithLatency(context.Background(), looper, request)
	if err != nil || response == nil {
		t.Fatalf("ExecuteWithLatency: %v", err)
	}
	if hop := caller.last.Hop; hop.Recipe != "named-recipe" || hop.Decision != "shared-route" {
		t.Fatalf("hop = %+v, want the parent's recipe and decision", hop)
	}
}
