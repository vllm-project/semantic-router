package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const contextPluginDSL = `
SIGNAL domain test { description: "test" }
ROUTE docs (description = "Answer from the documents") {
  PRIORITY 100
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN context {
    enabled: true
    sources: [{ type: "rag", backend: "vectorstore", backend_config: { vector_store_id: "vs_docs" }, match: { recall: { method: "embedding", top_k: 12, min_similarity: 0.4 }, judge: { preset: "relevance", top_k: 4, min_score: 0.6 } }, max_context_length: 0, on_failure: "block", cache_results: true, cache_ttl_seconds: 120 }, { type: "memory", store: { auto: false }, hybrid_search: true, hybrid_mode: "rrf" }]
    compress: { mode: "auto", targets: { tool_outputs: { mode: "extractive", min_tokens: 2000, target_tokens: 800 } }, match: { recall: { method: "bm25" } }, failure_mode: "fail_closed" }
    budget: { max_tokens: 6000, reserve_output_tokens: "auto" }
    injection: { mode: "system_prompt" }
  }
}
`

func contextPluginOf(t *testing.T, cfg *config.RouterConfig) *config.ContextPluginConfig {
	t.Helper()
	for i := range cfg.Decisions {
		if plugin := cfg.Decisions[i].GetContextConfig(); plugin != nil {
			return plugin
		}
	}
	t.Fatal("no decision has a context plugin")
	return nil
}

// The context plugin compiles with every source, compress, budget and
// injection field, and survives decompiling, both as text and as an AST.
func TestContextPluginSurvivesDSLCompileAndRoundTrip(t *testing.T) {
	cfg := mustCompile(t, contextPluginDSL)
	plugin := contextPluginOf(t, cfg)
	rag, memory := plugin.Source(config.ContextSourceRAG), plugin.Source(config.ContextSourceMemory)
	switch {
	case rag == nil || rag.Match == nil || rag.Match.Recall.TopK != 12 || rag.Match.Judge.TopK != 4 || *rag.Match.Judge.MinScore != 0.6:
		t.Fatalf("rag source %+v", rag)
	case rag.MaxContextLength == nil || *rag.MaxContextLength != 0 || rag.OnFailure != config.RAGOnFailureBlock || *rag.CacheTTLSeconds != 120:
		t.Fatalf("rag retrieval fields %+v", rag)
	case memory == nil || memory.AutoStore() == nil || *memory.AutoStore() || memory.HybridMode != "rrf":
		t.Fatalf("memory source %+v", memory)
	case plugin.Compress.Match.Recall.Method != config.MatchRecallBM25 || plugin.Compress.FailureMode != config.ContextCompressionFailureClosed:
		t.Fatalf("compress %+v", plugin.Compress)
	case plugin.Budget.MaxTokens.Value != 6000 || !plugin.Budget.ReserveOutputTokens.Auto || plugin.InjectionMode() != config.ContextInjectionSystemPrompt:
		t.Fatalf("budget %+v, injection %q", plugin.Budget, plugin.InjectionMode())
	}
	source, err := Decompile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	fromAST, errs := CompileAST(DecompileToAST(cfg))
	if len(errs) != 0 {
		t.Fatalf("AST round-trip: %v", errs)
	}
	for _, current := range []*config.RouterConfig{mustCompile(t, source), fromAST} {
		if got := contextPluginOf(t, current); !reflect.DeepEqual(got, plugin) {
			t.Fatalf("round trip changed the plugin:\n%+v\nwant\n%+v", got, plugin)
		}
	}
}

// The plugins context replaces fail to compile with what to write instead.
func TestRemovedContextPluginsFailToCompile(t *testing.T) {
	for pluginType, replacement := range map[string]string{
		"rag":                 "a rag source of the decision's context plugin",
		"memory":              "a memory source of the decision's context plugin",
		"context_compression": "the compress block and budget of the decision's context plugin",
	} {
		input := strings.Replace(contextPluginDSL, "PLUGIN context {", "PLUGIN "+pluginType+" {", 1)
		_, errs := Compile(input)
		if len(errs) == 0 || !strings.Contains(errs[0].Error(), `plugin "`+pluginType+`" was removed; write `+replacement) {
			t.Fatalf("%s compiles with %v", pluginType, errs)
		}
	}
}

func mustCompile(t *testing.T, input string) *config.RouterConfig {
	t.Helper()
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	return cfg
}
