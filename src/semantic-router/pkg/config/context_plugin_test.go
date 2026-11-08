package config

import (
	"reflect"
	"strings"
	"testing"
)

func vectorStoreSource() ContextSource {
	return ContextSource{Type: ContextSourceRAG, Backend: "vectorstore", BackendConfig: MustStructuredPayload(map[string]interface{}{"vector_store_id": "vs_docs"})}
}

func tokenLimit(value int) *CompressionTokenLimit {
	return &CompressionTokenLimit{Value: value}
}

// A context plugin retrieves its sources, each kept by its match block,
// compresses within its budget and injects; the checks name the field.
func TestContextPluginValidation(t *testing.T) {
	full := func() *ContextPluginConfig {
		return &ContextPluginConfig{
			Enabled: true,
			Sources: []ContextSource{vectorStoreSource(), {Type: ContextSourceMemory, Store: &ContextMemoryStoreConfig{}, HybridSearch: true, HybridMode: "rrf"}},
			Compress: &ContextCompressConfig{
				Targets: &ContextCompressionTargetsConfig{ToolOutputs: ContextCompressionTargetConfig{Mode: ContextCompressionTargetExtractive, MinTokens: 2000, TargetTokens: 1000}},
				Match:   &MatchConfig{Recall: &MatchRecall{Method: MatchRecallBM25}},
			},
			Budget:    &ContextBudgetConfig{MaxTokens: tokenLimit(6000), ReserveOutputTokens: &CompressionTokenLimit{Auto: true}},
			Injection: &ContextInjectionConfig{Mode: ContextInjectionSystemPrompt},
		}
	}
	if err := validateContextPlugin(matchConfig(), "p", full()); err != nil {
		t.Fatalf("a complete plugin is refused: %v", err)
	}
	cases := map[string]struct {
		edit func(*ContextPluginConfig)
		want string
	}{
		"nothing to do":       {func(p *ContextPluginConfig) { p.Sources, p.Compress, p.Budget, p.Injection = nil, nil, nil, nil }, "p: an enabled context plugin retrieves sources, compresses the request or both"},
		"two rag sources":     {func(p *ContextPluginConfig) { p.Sources = append(p.Sources, vectorStoreSource()) }, "p.sources[2]: a context plugin has at most one rag source"},
		"unknown source type": {func(p *ContextPluginConfig) { p.Sources[0].Type = "web" }, "p.sources[0].type must be rag or memory"},
		"memory field on rag": {func(p *ContextPluginConfig) { p.Sources[0].Store = &ContextMemoryStoreConfig{} }, "p.sources[0].store: a rag source takes no store"},
		"rag field on memory": {func(p *ContextPluginConfig) { p.Sources[1].OnFailure = RAGOnFailureBlock }, "p.sources[1].on_failure: a memory source takes no on_failure"},
		"rag without backend": {func(p *ContextPluginConfig) { p.Sources[0].Backend = "" }, "p.sources[0]: backend is required"},
		"vector store id": {func(p *ContextPluginConfig) {
			p.Sources[0].BackendConfig = MustStructuredPayload(map[string]interface{}{})
		}, "backend_config.vector_store_id is required"},
		"mcp unavailable": {func(p *ContextPluginConfig) {
			p.Sources[0] = ContextSource{Type: ContextSourceRAG, Backend: "mcp", BackendConfig: MustStructuredPayload(map[string]interface{}{"server_name": "s", "tool_name": "t"})}
		}, "the router has no MCP tool invoker"},
		"unknown on_failure": {func(p *ContextPluginConfig) { p.Sources[0].OnFailure = "retry" }, "p.sources[0]: on_failure must be skip, block or warn"},
		"bm25 over a vector store without a lexical index": {func(p *ContextPluginConfig) {
			p.Sources[0].Match = &MatchConfig{Recall: &MatchRecall{Method: MatchRecallBM25, TopK: 5}}
		}, "p.sources[0].match.recall.method: bm25 recall reads the chunks the in-memory vector store keeps; set global.stores.vector_store.backend_type: memory, or recall with embedding"},
		"judge only over an external collection": {func(p *ContextPluginConfig) {
			p.Sources[0] = ContextSource{Type: ContextSourceRAG, Backend: "milvus", BackendConfig: MustStructuredPayload(map[string]interface{}{"collection": "docs"}), Match: &MatchConfig{Judge: &MatchJudge{Preset: "relevance"}}}
		}, "p.sources[0].match: a judge without recall reads a local knowledge base, a vectorstore source; a milvus source recalls with embedding"},
		"bm25 over memories": {func(p *ContextPluginConfig) {
			p.Sources[1].Match = &MatchConfig{Recall: &MatchRecall{Method: MatchRecallBM25, TopK: 5}}
		}, "p.sources[1].match.recall.method: a memory source recalls with embedding only"},
		"hybrid mode alone":     {func(p *ContextPluginConfig) { p.Sources[1].HybridSearch = false }, "p.sources[1].hybrid_mode fuses the scores of hybrid_search"},
		"unknown hybrid mode":   {func(p *ContextPluginConfig) { p.Sources[1].HybridMode = "rerank" }, `p.sources[1].hybrid_mode "rerank" is not supported`},
		"budget alone":          {func(p *ContextPluginConfig) { p.Compress = nil }, "p.budget bounds the request compress produces; add compress or remove budget"},
		"zero max tokens":       {func(p *ContextPluginConfig) { p.Budget.MaxTokens = tokenLimit(0) }, "p.budget.max_tokens must be at least 1, or auto"},
		"injection alone":       {func(p *ContextPluginConfig) { p.Sources = nil }, "p.injection places what the sources retrieve"},
		"unknown injection":     {func(p *ContextPluginConfig) { p.Injection.Mode = "user_message" }, "p.injection.mode must be tool_role or system_prompt"},
		"compress mode":         {func(p *ContextPluginConfig) { p.Compress.Mode = "never" }, "p.compress.mode must be auto or always"},
		"compress target order": {func(p *ContextPluginConfig) { p.Compress.Targets.ToolOutputs.TargetTokens = 3000 }, "p.compress.targets.tool_outputs.target_tokens must be less than min_tokens"},
		"recovery store": {func(p *ContextPluginConfig) {
			p.Compress.Recovery = &ContextCompressionRecoveryConfig{Enabled: true, Store: "redis"}
		}, "p.compress.recovery needs the response cache's redis connection"},
		"compress recall only": {func(p *ContextPluginConfig) {
			p.Compress.Match = &MatchConfig{Recall: &MatchRecall{Method: MatchRecallEmbedding}}
		}, "p.compress.match.recall.method: context compression recalls with bm25 only"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			plugin := full()
			tc.edit(plugin)
			err := validateContextPlugin(matchConfig(), "p", plugin)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %v, want %q", err, tc.want)
			}
		})
	}
	disabled := full()
	disabled.Enabled, disabled.Sources[0].Backend = false, "mcp"
	if err := validateContextPlugin(matchConfig(), "p", disabled); err != nil {
		t.Fatalf("a disabled plugin is not checked: %v", err)
	}
}

// A local knowledge base, a vectorstore source over the in-memory vector
// store, recalls by bm25 or is judged whole, and a memory source is judged
// without recall: both read their items listed, so neither needs a query
// embedding.
func TestContextSourcesJudgedWithoutEmbeddingRecall(t *testing.T) {
	cfg := matchConfig()
	cfg.VectorStore = &VectorStoreConfig{Enabled: true, BackendType: "memory"}
	judge := &MatchJudge{Preset: "relevance", MaxCandidates: 32}
	for name, source := range map[string]ContextSource{
		"bm25 over a local knowledge base": func() ContextSource {
			s := vectorStoreSource()
			s.Match = &MatchConfig{Recall: &MatchRecall{Method: MatchRecallBM25, TopK: 20}, Judge: judge}
			return s
		}(),
		"a local knowledge base judged whole": func() ContextSource {
			s := vectorStoreSource()
			s.Match = &MatchConfig{Judge: judge}
			return s
		}(),
		"memories judged whole": {Type: ContextSourceMemory, Match: &MatchConfig{Judge: judge}},
	} {
		plugin := &ContextPluginConfig{Enabled: true, Sources: []ContextSource{source}}
		if err := validateContextPlugin(cfg, "p", plugin); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

// A memory source without a block recalls with the memory store's defaults,
// themselves the memory row of the match defaults when unset.
func TestMemorySourceRecallsWithTheStoreDefaults(t *testing.T) {
	cfg := &RouterConfig{}
	if got, want := cfg.MemoryMatch(&ContextSource{Type: ContextSourceMemory}), MatchMemory.Default(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unset store defaults recall with %+v, want %+v", got.Recall, want.Recall)
	}
	cfg.Memory.DefaultRetrievalLimit, cfg.Memory.DefaultSimilarityThreshold = 8, 0.5
	got := cfg.MemoryMatch(&ContextSource{Type: ContextSourceMemory})
	if got.Recall.TopK != 8 || *got.Recall.MinSimilarity != 0.5 || got.Judge != nil {
		t.Fatalf("the store's defaults recall with %+v", got.Recall)
	}
	written := &MatchConfig{Recall: &MatchRecall{TopK: 3}}
	if got := cfg.MemoryMatch(&ContextSource{Type: ContextSourceMemory, Match: written}); got.Recall.TopK != 3 || got.Recall.MinSimilarity != nil {
		t.Fatalf("a written block replaces the defaults whole: %+v", got.Recall)
	}
}

// The judges a decision prepares are its context plugin's: a rag source's
// default relevance judge and compress's, but not a recall-only block.
func TestContextPluginMatchJudges(t *testing.T) {
	plugin := ContextPluginConfig{Enabled: true, Sources: []ContextSource{vectorStoreSource(), {Type: ContextSourceMemory}}, Compress: &ContextCompressConfig{}}
	decision := Decision{Plugins: []DecisionPlugin{{Type: DecisionPluginContext, Configuration: MustStructuredPayload(plugin)}}}
	if judges := decision.MatchJudges(); len(judges) != 2 || judges[0].TopK != 6 || judges[1].Preset != "relevance" || judges[1].TopK != 0 {
		t.Fatalf("judges %+v, want the rag default and compress's", judges)
	}
	plugin.Compress.Match = &MatchConfig{Recall: &MatchRecall{Method: MatchRecallBM25}}
	decision.Plugins[0].Configuration = MustStructuredPayload(plugin)
	if judges := decision.MatchJudges(); len(judges) != 1 {
		t.Fatalf("a recall-only compress block judges nothing: %+v", judges)
	}
	plugin.Enabled = false
	decision.Plugins[0].Configuration = MustStructuredPayload(plugin)
	if judges := decision.MatchJudges(); len(judges) != 0 {
		t.Fatalf("a disabled plugin judges nothing: %+v", judges)
	}
}
