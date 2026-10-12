package extproc

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

func gatingRouter(pluginType string) (*OpenAIRouter, *config.Decision) {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{
		"text-model":    {Capabilities: []string{"text"}},
		"image-model":   {Capabilities: []string{"vision"}},
		"labeled-model": {Capabilities: []string{"chat", "code"}},
	}
	payload := map[string]interface{}{"enabled": true}
	if pluginType == config.DecisionPluginRAG {
		payload["backend"] = "external_api"
	}
	cfg.Decisions = []config.Decision{{
		Name:      "gated",
		ModelRefs: []config.ModelRef{{Model: "text-model"}},
		Plugins:   []config.DecisionPlugin{{Type: pluginType, Configuration: config.MustStructuredPayload(payload)}},
	}}
	return &OpenAIRouter{Config: cfg}, &cfg.Decisions[0]
}

func gatingRequest(parts ...llmprotocol.Content) *llmprotocol.Request {
	return &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: parts}}}
}

var (
	gatingText  = llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "describe this"}
	gatingImage = llmprotocol.Content{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/a.png"}
)

func TestRAGRunsWhenRequestAndBackendCoverIt(t *testing.T) {
	router, decision := gatingRouter(config.DecisionPluginRAG)
	ctx := &RequestContext{VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingText), PluginBackendModel: "text-model"}
	if _, runs := router.resolveRAGPluginConfig(ctx, "gated"); !runs || len(ctx.PluginSkips) != 0 {
		t.Fatalf("runs=%v skips=%v", runs, ctx.PluginSkips)
	}
}

func TestRAGSkipsWithTypedReasonWhenBackendLacksText(t *testing.T) {
	router, decision := gatingRouter(config.DecisionPluginRAG)
	ctx := &RequestContext{VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingText), PluginBackendModel: "image-model"}
	if _, runs := router.resolveRAGPluginConfig(ctx, "gated"); runs {
		t.Fatal("RAG ran on a backend that declares no text")
	}
	want := []pluginSkip{{Plugin: config.DecisionPluginRAG, Reason: pluginruntime.SkipBackendLacksCapability}}
	if !reflect.DeepEqual(ctx.PluginSkips, want) {
		t.Fatalf("skips=%v want %v", ctx.PluginSkips, want)
	}
}

func TestRAGSkipsWhenRequestHasNoText(t *testing.T) {
	router, decision := gatingRouter(config.DecisionPluginRAG)
	ctx := &RequestContext{VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingImage), PluginBackendModel: "text-model"}
	if _, runs := router.resolveRAGPluginConfig(ctx, "gated"); runs {
		t.Fatal("RAG ran on a request with no text")
	}
	if len(ctx.PluginSkips) != 1 || ctx.PluginSkips[0].Reason != pluginruntime.SkipRequestDoesNotRequire {
		t.Fatalf("skips=%v", ctx.PluginSkips)
	}
}

func TestBackendWithoutRecognizedDeclarationIsNotGated(t *testing.T) {
	router, decision := gatingRouter(config.DecisionPluginRAG)
	for _, model := range []string{"labeled-model", "unlisted-model", ""} {
		ctx := &RequestContext{VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingText), PluginBackendModel: model}
		if _, runs := router.resolveRAGPluginConfig(ctx, "gated"); !runs {
			t.Fatalf("model %q was gated without a recognized declaration", model)
		}
	}
}

func TestMemoryReportsSkipThroughItsOutcome(t *testing.T) {
	router, decision := gatingRouter(config.DecisionPluginMemory)
	ctx := &RequestContext{
		VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingText),
		PluginBackendModel: "image-model", Headers: map[string]string{},
	}
	if err := router.handleMemoryRetrieval(ctx, "describe this", ctx.SemanticRequest); err != nil {
		t.Fatal(err)
	}
	if ctx.MemoryStatus != "skipped" || ctx.MemoryReason != string(pluginruntime.SkipBackendLacksCapability) {
		t.Fatalf("status=%q reason=%q", ctx.MemoryStatus, ctx.MemoryReason)
	}
}

func TestMediaBlocksSurviveEnrichmentUnchanged(t *testing.T) {
	router, _ := gatingRouter(config.DecisionPluginRAG)
	request := gatingRequest(gatingText, gatingImage)
	before := request.Messages[0].Content[1]
	ctx := &RequestContext{}
	if err := router.injectAsSystemPrompt(request, "retrieved", ctx); err != nil {
		t.Fatal(err)
	}
	router.injectRetrievedMemories(ctx, request, []*memory.RetrieveResult{{Memory: &memory.Memory{ID: "m1", Content: "likes diagrams"}, Score: 0.9}})
	if len(request.Messages) != 2 {
		t.Fatalf("memory was not injected: %d messages", len(request.Messages))
	}
	if got := request.Messages[1].Content[1]; !reflect.DeepEqual(got, before) {
		t.Fatalf("image block changed: %+v != %+v", got, before)
	}
}
