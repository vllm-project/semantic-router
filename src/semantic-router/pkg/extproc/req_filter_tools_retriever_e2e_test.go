package extproc

import (
	"context"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// stubRetriever is a test double that returns a fixed RetrievalResult.
type stubRetriever struct {
	strategyID string
	results    []tools.ToolSimilarity
	confidence float32
}

func (s *stubRetriever) Retrieve(_ context.Context, _ tools.RetrievalInput) (tools.RetrievalResult, error) {
	return tools.RetrievalResult{
		Tools:      s.results,
		Confidence: s.confidence,
		StrategyID: s.strategyID,
	}, nil
}

// makeToolsRouter builds a minimal OpenAIRouter wired for tool selection tests.
func makeToolsRouter(t *testing.T, reg *tools.Registry) *OpenAIRouter {
	t.Helper()
	db := tools.NewToolsDatabase(tools.ToolsDatabaseOptions{
		Enabled:             true,
		SimilarityThreshold: 0.5,
	})
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			ToolSelection: config.ToolSelection{
				Tools: config.ToolsConfig{
					TopK:            2,
					FallbackToEmpty: true,
				},
			},
		},
		ToolsDatabase: db,
		ToolsRegistry: reg,
	}
}

// makeToolsPluginConfig builds a per-decision tools config for a given strategy.
func makeToolsPluginConfig(strategy string) *config.ToolsPluginConfig {
	return &config.ToolsPluginConfig{
		Enabled:  true,
		Mode:     config.ToolsPluginModePassthrough,
		Strategy: strategy,
	}
}

// sampleTool returns a minimal ChatCompletionToolParam for use in tests.
func sampleTool(name string) tools.ToolSimilarity {
	return tools.ToolSimilarity{
		Entry: tools.ToolEntry{
			Tool: openai.ChatCompletionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name: name,
				},
			},
		},
		Similarity: 0.9,
	}
}

func TestRetrieverE2E_RegisteredStrategyToolsReachRequest(t *testing.T) {
	reg := tools.NewRegistry()
	reg.Register("custom", &stubRetriever{
		strategyID: "custom",
		results:    []tools.ToolSimilarity{sampleTool("search"), sampleTool("calculate")},
		confidence: 0.95,
	})

	router := makeToolsRouter(t, reg)
	toolsCfg := makeToolsPluginConfig("custom")

	req := &openai.ChatCompletionNewParams{}
	req.ToolChoice.OfAuto.Value = "auto"
	resp := &ext_proc.ProcessingResponse{}

	err := router.handleToolSelection(req, "find the weather in Paris", nil, &resp, &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "test-decision",
			Plugins: []config.DecisionPlugin{
				mustToolsDecisionPlugin(t, toolsCfg),
			},
		},
	})
	if err != nil {
		t.Fatalf("handleToolSelection returned unexpected error: %v", err)
	}
	if len(req.Tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(req.Tools))
	}
	if req.Tools[0].Function.Name != "search" {
		t.Errorf("expected first tool to be 'search', got %q", req.Tools[0].Function.Name)
	}
}

func TestRetrieverE2E_UnknownStrategyFallsBackToDBWithSuffix(t *testing.T) {
	reg := tools.NewRegistry()
	// register only "default", not "bm25"
	// reg.Register("default", &stubRetriever{
	// 	strategyID: "default",
	// 	results:    []tools.ToolSimilarity{sampleTool("fallback-tool")},
	// 	confidence: 0.5,
	// })

	router := makeToolsRouter(t, reg)
	toolsCfg := makeToolsPluginConfig("bm25")

	req := &openai.ChatCompletionNewParams{}
	req.ToolChoice.OfAuto.Value = "auto"
	resp := &ext_proc.ProcessingResponse{}

	err := router.handleToolSelection(req, "some query", nil, &resp, &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "test-decision",
			Plugins: []config.DecisionPlugin{
				mustToolsDecisionPlugin(t, toolsCfg),
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	strategyHeader := getHeaderValue(resp, "x-vsr-tools-strategy")
	if strategyHeader != "bm25-fallback" {
		t.Errorf("expected strategy header 'bm25-fallback', got %q", strategyHeader)
	}
}

func TestRetrieverE2E_NilRegistryFallsBackWithSuffix(t *testing.T) {
	router := makeToolsRouter(t, nil) // nil registry
	toolsCfg := makeToolsPluginConfig("embedding")

	req := &openai.ChatCompletionNewParams{}
	req.ToolChoice.OfAuto.Value = "auto"
	resp := &ext_proc.ProcessingResponse{}

	err := router.handleToolSelection(req, "some query", nil, &resp, &RequestContext{
		VSRSelectedDecision: &config.Decision{
			Name: "test-decision",
			Plugins: []config.DecisionPlugin{
				mustToolsDecisionPlugin(t, toolsCfg),
			},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	strategyHeader := getHeaderValue(resp, "x-vsr-tools-strategy")
	if strategyHeader != "embedding-fallback" {
		t.Errorf("expected strategy header 'embedding-fallback', got %q", strategyHeader)
	}
}

func TestRetrieverE2E_ConversationTriggeredRouteAppliesToolsStrategy(t *testing.T) {
	router := makeConversationTriggeredToolsRouter(t)
	req := newConversationTriggeredToolsRequest()
	ctx := &RequestContext{
		Headers:   map[string]string{},
		RequestID: "conversation-triggered-tools",
	}

	assertConversationTriggeredDecisionMatch(t, router, req, ctx)

	userContent, nonUserMessages := extractUserAndNonUserContent(req)
	resp := &ext_proc.ProcessingResponse{}
	err := router.handleToolSelection(req, userContent, nonUserMessages, &resp, ctx)
	if err != nil {
		t.Fatalf("handleToolSelection returned unexpected error: %v", err)
	}

	if len(req.Tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(req.Tools))
	}
	if req.Tools[0].Function.Name != "search" {
		t.Fatalf("expected first tool to be search, got %q", req.Tools[0].Function.Name)
	}

	strategyHeader := getHeaderValue(resp, "x-vsr-tools-strategy")
	if strategyHeader != "custom" {
		t.Fatalf("expected strategy header custom, got %q", strategyHeader)
	}
}

func makeConversationTriggeredToolsRouter(t *testing.T) *OpenAIRouter {
	t.Helper()

	compiled := mustCompileConversationTriggeredToolsDSL(t)

	reg := tools.NewRegistry()
	reg.Register("custom", &stubRetriever{
		strategyID: "custom",
		results:    []tools.ToolSimilarity{sampleTool("search"), sampleTool("calculate")},
		confidence: 0.95,
	})

	router := makeToolsRouter(t, reg)
	router.Config.ConversationRules = compiled.ConversationRules
	router.Config.Decisions = compiled.Decisions
	router.Classifier = &classification.Classifier{Config: router.Config}
	return router
}

func mustCompileConversationTriggeredToolsDSL(t *testing.T) *config.RouterConfig {
	t.Helper()

	compiled, errs := dsl.Compile(`
SIGNAL conversation multi_turn_tool_flow {
  feature: {
    type: "count"
    source: { type: "assistant_tool_cycle" }
  }
  predicate: { gte: 1 }
}

ROUTE agentic_tools {
  PRIORITY 180
  WHEN conversation("multi_turn_tool_flow")
  MODEL "agent-model"
  PLUGIN tools {
    enabled: true
    mode: "passthrough"
    semantic_selection: true
    strategy: "custom"
    dynamic_retrieval: {
      enabled: true
      strategy: "hybrid_history"
      history_window: 8
    }
  }
}
`)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	return compiled
}

func newConversationTriggeredToolsRequest() *openai.ChatCompletionNewParams {
	req := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Find the deployment runbook."),
			assistantToolCallMessage("search_docs"),
			openai.ToolMessage("deployment runbook content", "call-search-docs"),
			openai.UserMessage("Summarize the deployment steps."),
		},
	}
	req.ToolChoice.OfAuto.Value = "auto"
	return req
}

func assertConversationTriggeredDecisionMatch(
	t *testing.T,
	router *OpenAIRouter,
	req *openai.ChatCompletionNewParams,
	ctx *RequestContext,
) {
	t.Helper()

	history := extractSignalConversationHistory(req)
	signalInput := router.prepareSignalEvaluationInput(history)

	signals, err := router.evaluateSignalsForDecision("auto", signalInput, history.nonUserMessages, ctx)
	if err != nil {
		t.Fatalf("evaluateSignalsForDecision returned unexpected error: %v", err)
	}
	if len(signals.MatchedConversationRules) != 1 || signals.MatchedConversationRules[0] != "multi_turn_tool_flow" {
		t.Fatalf("matched conversation rules = %v, want [multi_turn_tool_flow]", signals.MatchedConversationRules)
	}

	result, fallbackModel := router.runDecisionEngine("auto", ctx, signals)
	if fallbackModel != "" {
		t.Fatalf("expected no fallback model, got %q", fallbackModel)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("expected a matched decision")
	}
	if result.Decision.Name != "agentic_tools" {
		t.Fatalf("matched decision = %q, want agentic_tools", result.Decision.Name)
	}

	router.applyDecisionResultToContext(result, ctx)

	toolsCfg := ctx.VSRSelectedDecision.GetToolsConfig()
	if toolsCfg == nil {
		t.Fatal("selected decision missing tools config")
	}
	if toolsCfg.EffectiveStrategy() != "custom" {
		t.Fatalf("tools strategy = %q, want custom", toolsCfg.EffectiveStrategy())
	}
	if !toolsCfg.DynamicRetrievalEnabled() {
		t.Fatal("expected dynamic retrieval to remain attached to the selected decision")
	}
}

// mustToolsDecisionPlugin encodes a ToolsPluginConfig into a DecisionPlugin.
func mustToolsDecisionPlugin(t *testing.T, cfg *config.ToolsPluginConfig) config.DecisionPlugin {
	t.Helper()
	payload, err := config.NewStructuredPayload(cfg)
	if err != nil {
		t.Fatalf("NewStructuredPayload: %v", err)
	}
	return config.DecisionPlugin{Type: config.DecisionPluginTools, Configuration: payload}
}

// getHeaderValue extracts a header value from the ProcessingResponse for assertions.
func getHeaderValue(resp *ext_proc.ProcessingResponse, key string) string {
	if resp == nil {
		return ""
	}
	rb := resp.GetRequestBody()
	if rb == nil || rb.GetResponse() == nil || rb.GetResponse().HeaderMutation == nil {
		return ""
	}
	for _, h := range rb.GetResponse().HeaderMutation.SetHeaders {
		if h.Header != nil && h.Header.Key == key {
			return h.Header.Value
		}
	}
	return ""
}
