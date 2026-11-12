package extproc

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

func gatingRouter(sourceType string) (*OpenAIRouter, *config.Decision) {
	cfg := &config.RouterConfig{}
	cfg.ModelConfig = map[string]config.ModelParams{
		"text-model":    {Capabilities: []string{"text"}},
		"image-model":   {Capabilities: []string{"vision"}},
		"labeled-model": {Capabilities: []string{"chat", "code"}},
	}
	source := map[string]interface{}{"type": sourceType}
	if sourceType == config.ContextSourceRAG {
		source["backend"] = "external_api"
	}
	payload := map[string]interface{}{"enabled": true, "sources": []interface{}{source}}
	cfg.Decisions = []config.Decision{{
		Name:      "gated",
		ModelRefs: []config.ModelRef{{Model: "text-model"}},
		Plugins:   []config.DecisionPlugin{{Type: config.DecisionPluginContext, Configuration: config.MustStructuredPayload(payload)}},
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
	router, decision := gatingRouter(config.ContextSourceRAG)
	ctx := &RequestContext{VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingText), PluginBackendModel: "text-model"}
	if runs, _ := router.pluginApplies(ctx, config.DecisionPluginContext); !runs || len(ctx.PluginSkips) != 0 {
		t.Fatalf("runs=%v skips=%v", runs, ctx.PluginSkips)
	}
}

func TestRAGRunsWhenBackendCardOmitsText(t *testing.T) {
	router, decision := gatingRouter(config.ContextSourceRAG)
	ctx := &RequestContext{VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingText), PluginBackendModel: "image-model"}
	if runs, _ := router.pluginApplies(ctx, config.DecisionPluginContext); !runs || len(ctx.PluginSkips) != 0 {
		t.Fatalf("runs=%v skips=%v: text is assumed for every model", runs, ctx.PluginSkips)
	}
}

func TestRAGSkipsWhenRequestHasNoText(t *testing.T) {
	router, decision := gatingRouter(config.ContextSourceRAG)
	ctx := &RequestContext{VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingImage), PluginBackendModel: "text-model"}
	if runs, _ := router.pluginApplies(ctx, config.DecisionPluginContext); runs {
		t.Fatal("RAG ran on a request with no text")
	}
	if len(ctx.PluginSkips) != 1 || ctx.PluginSkips[0].Reason != pluginruntime.SkipRequestDoesNotRequire {
		t.Fatalf("skips=%v", ctx.PluginSkips)
	}
}

func TestBackendCardNeverGatesTextPlugins(t *testing.T) {
	router, decision := gatingRouter(config.ContextSourceRAG)
	for _, model := range []string{"text-model", "image-model", "labeled-model", "unlisted-model", ""} {
		ctx := &RequestContext{VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingText), PluginBackendModel: model}
		if runs, _ := router.pluginApplies(ctx, config.DecisionPluginContext); !runs {
			t.Fatalf("model %q was gated although the request has text", model)
		}
	}
}

func TestMemoryReportsSkipThroughItsOutcome(t *testing.T) {
	router, decision := gatingRouter(config.ContextSourceMemory)
	ctx := &RequestContext{
		VSRSelectedDecision: decision, SemanticRequest: gatingRequest(gatingImage),
		PluginBackendModel: "text-model", Headers: map[string]string{},
	}
	if err := router.retrieveContextSources(ctx, "describe this", ctx.SemanticRequest); err != nil {
		t.Fatal(err)
	}
	if ctx.MemoryStatus != "skipped" || ctx.MemoryReason != string(pluginruntime.SkipRequestDoesNotRequire) {
		t.Fatalf("status=%q reason=%q", ctx.MemoryStatus, ctx.MemoryReason)
	}
}

func TestMediaBlocksSurviveEnrichmentUnchanged(t *testing.T) {
	router, decision := gatingRouter(config.ContextSourceRAG)
	request := gatingRequest(gatingText, gatingImage)
	before := request.Messages[0].Content[1]
	ctx := &RequestContext{SemanticRequest: request}
	plugin := decision.GetContextConfig()
	if err := router.injectContext(ctx, &config.ContextPluginConfig{Enabled: true, Injection: &config.ContextInjectionConfig{Mode: config.ContextInjectionSystemPrompt}}, config.ContextSourceRAG, "retrieved"); err != nil {
		t.Fatal(err)
	}
	router.injectRetrievedMemories(ctx, plugin, []*memory.RetrieveResult{{Memory: &memory.Memory{ID: "m1", Content: "likes diagrams"}, Score: 0.9}})
	if len(request.Messages) != 3 {
		t.Fatalf("memory was not injected: %d messages", len(request.Messages))
	}
	if got := request.Messages[0].Content[1]; !reflect.DeepEqual(got, before) {
		t.Fatalf("image block changed: %+v != %+v", got, before)
	}
}
