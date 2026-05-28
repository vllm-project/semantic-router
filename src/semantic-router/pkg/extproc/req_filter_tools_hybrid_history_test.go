package extproc

import (
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// hybridHistoryCandidate constructs a ToolSimilarity with the given name and
// similarity score. Description is intentionally distinct from the test query
// so lexical overlap stays zero and the weighted-mode score depends only on
// the similarity component.
func hybridHistoryCandidate(name string, similarity float32) tools.ToolSimilarity {
	return tools.ToolSimilarity{
		Entry: tools.ToolEntry{
			Tool: openai.ChatCompletionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name: name,
				},
			},
			Description: "irrelevant payload " + name,
		},
		Similarity: similarity,
	}
}

// requestWithABAToolHistory builds a *ChatCompletionNewParams whose messages
// carry an alternating assistant tool_call history of [toolA, toolB, toolA].
// This is the minimum shape that gives historyTransitionScore a non-zero
// (toolA -> toolB) edge to observe.
func requestWithABAToolHistory(toolA, toolB string) *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("turn 1"),
			assistantToolCallMessage(toolA),
			openai.ToolMessage("result", "call-"+toolA),
			openai.UserMessage("turn 2"),
			assistantToolCallMessage(toolB),
			openai.ToolMessage("result", "call-"+toolB),
			openai.UserMessage("turn 3"),
			assistantToolCallMessage(toolA),
			openai.ToolMessage("result", "call-"+toolA),
			openai.UserMessage("please continue the workflow"),
		},
	}
}

// TestFindToolsForQueryExt_HybridHistoryPromotesHistoryConsistentTool asserts
// that the hybrid_history wiring is alive end-to-end inside extproc:
// when retrieval_strategy=hybrid_history is configured and the request carries
// an assistant tool_call history of [tool_a, tool_b, tool_a], the ranker must
// promote tool_b (history-consistent next step) above tool_a despite tool_a
// having a higher raw similarity score.
//
// Regression nail for the wiring lost in PR #1841: prior to restoring it,
// findToolsForQueryExt called FilterAndRankTools with the conversation history
// hard-coded to nil/0, so the strategy had no observable effect even when the
// caller opted in via config.
func TestFindToolsForQueryExt_HybridHistoryPromotesHistoryConsistentTool(t *testing.T) {
	reg := tools.NewRegistry()
	reg.Register("default", &stubRetriever{
		strategyID: "default",
		results: []tools.ToolSimilarity{
			hybridHistoryCandidate("tool_a", 0.95), // higher sim, A->A is the unseen edge
			hybridHistoryCandidate("tool_b", 0.55), // lower sim, A->B is the observed edge
		},
		confidence: 0.95,
	})

	router := makeToolsRouter(t, reg)
	toolsCfg := makeToolsPluginConfig("default")
	advanced := &config.AdvancedToolFilteringConfig{
		Enabled:           true,
		RetrievalStrategy: config.ToolRetrievalStrategyHybridHistory,
	}

	req := requestWithABAToolHistory("tool_a", "tool_b")

	selected, _, _, _, err := router.findToolsForQueryExt(
		req,
		"please continue the workflow",
		"",
		&RequestContext{},
		toolsCfg,
		2,
		advanced,
		toolsCfg.EffectiveStrategy(),
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("findToolsForQueryExt returned unexpected error: %v", err)
	}
	if len(selected) != 2 {
		t.Fatalf("expected 2 selected tools, got %d", len(selected))
	}
	if selected[0].Function.Name != "tool_b" {
		t.Fatalf("hybrid_history must promote history-consistent tool_b above higher-similarity tool_a; got order [%q, %q]",
			selected[0].Function.Name, selected[1].Function.Name)
	}
}

// TestFindToolsForQueryExt_DefaultStrategyIgnoresToolHistory asserts that the
// default (weighted/unset) retrieval path ignores assistant tool_call history
// and ranks by similarity, matching pre-fix behavior. This pins down the
// symmetric concern that a future change cannot accidentally flip the default
// path into hybrid mode.
func TestFindToolsForQueryExt_DefaultStrategyIgnoresToolHistory(t *testing.T) {
	reg := tools.NewRegistry()
	reg.Register("default", &stubRetriever{
		strategyID: "default",
		results: []tools.ToolSimilarity{
			hybridHistoryCandidate("tool_a", 0.95),
			hybridHistoryCandidate("tool_b", 0.55),
		},
		confidence: 0.95,
	})

	router := makeToolsRouter(t, reg)
	toolsCfg := makeToolsPluginConfig("default")
	advanced := &config.AdvancedToolFilteringConfig{
		Enabled: true,
		// RetrievalStrategy left empty -> EffectiveToolRetrievalStrategy returns "weighted"
	}

	req := requestWithABAToolHistory("tool_a", "tool_b")

	selected, _, _, _, err := router.findToolsForQueryExt(
		req,
		"please continue the workflow",
		"",
		&RequestContext{},
		toolsCfg,
		2,
		advanced,
		toolsCfg.EffectiveStrategy(),
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("findToolsForQueryExt returned unexpected error: %v", err)
	}
	if len(selected) != 2 {
		t.Fatalf("expected 2 selected tools, got %d", len(selected))
	}
	if selected[0].Function.Name != "tool_a" {
		t.Fatalf("default (weighted) strategy must rank by similarity and ignore tool history; got order [%q, %q]",
			selected[0].Function.Name, selected[1].Function.Name)
	}
}
