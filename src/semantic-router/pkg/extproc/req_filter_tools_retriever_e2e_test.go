package extproc

import (
	"context"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
