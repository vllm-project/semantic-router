package config

import (
	"slices"
	"testing"
)

func TestModelRuntimeEmbeddingDeploymentsFollowConsumerOwnership(t *testing.T) {
	kbSignal := func(cfg *RouterConfig) {
		cfg.KBRules = []KBSignalRule{{KB: "catalog-kb"}}
	}
	localBinding := func(cfg *RouterConfig) {
		cfg.ModelBindings = map[string]ModelBinding{"embedding": {Deployment: "local", Contract: "embedding.v1"}}
	}
	plugin := func(kind string, enabled bool) DecisionPlugin {
		return DecisionPlugin{Type: kind, Configuration: MustStructuredPayload(map[string]any{"enabled": enabled})}
	}
	namedPlugin := func(cfg *RouterConfig, kind string, reachable bool) {
		cfg.Recipes = []RoutingRecipe{{Name: DefaultRecipeName}, {Name: "named", Profile: RoutingProfile{
			Decisions: []Decision{{Plugins: []DecisionPlugin{plugin(kind, true)}}},
		}}}
		if reachable {
			cfg.Entrypoints = []EntrypointMapping{{Recipe: "named", ModelNames: []string{"public"}}}
		}
	}
	cases := []struct {
		name  string
		setup func(*RouterConfig)
		want  []string
	}{
		{name: "unused KB and embedding catalog", setup: func(*RouterConfig) {}},
		{name: "KB inherits global binding", setup: kbSignal, want: []string{"global"}},
		{name: "KB owns local binding", setup: func(cfg *RouterConfig) { kbSignal(cfg); localBinding(cfg) }, want: []string{"local"}},
		{name: "KB projection owns local binding", setup: func(cfg *RouterConfig) {
			localBinding(cfg)
			cfg.Projections.Scores = []ProjectionScore{{Name: "quality", Inputs: []ProjectionScoreInput{{Type: ProjectionInputKBMetric, KB: "catalog-kb"}}}}
		}, want: []string{"local"}},
		{name: "unreachable named KB", setup: func(cfg *RouterConfig) { kbSignal(cfg); moveTestRoutingToUnmappedRecipe(cfg) }},
		{name: "disabled memory plugin", setup: func(cfg *RouterConfig) {
			cfg.Decisions = []Decision{{Plugins: []DecisionPlugin{plugin("memory", false)}}}
		}},
		{name: "reachable named tools", setup: func(cfg *RouterConfig) { namedPlugin(cfg, "tool_selection", true) }, want: []string{"global"}},
		{name: "reachable named memory", setup: func(cfg *RouterConfig) { namedPlugin(cfg, "memory", true) }, want: []string{"global"}},
		{name: "unreachable named memory", setup: func(cfg *RouterConfig) { namedPlugin(cfg, "memory", false) }},
		{name: "explicit embedding API", setup: func(cfg *RouterConfig) { cfg.API.Embeddings.Enabled = true }, want: []string{"global"}},
		{name: "explicit tools service", setup: func(cfg *RouterConfig) { cfg.Tools.Enabled = true }, want: []string{"global"}},
		{name: "explicit memory service", setup: func(cfg *RouterConfig) { cfg.Memory.Enabled = true }, want: []string{"global"}},
		{name: "explicit vector store", setup: func(cfg *RouterConfig) { cfg.VectorStore = &VectorStoreConfig{Enabled: true} }, want: []string{"global"}},
		{name: "semantic response cache", setup: func(cfg *RouterConfig) {
			cfg.SemanticCache.Enabled, cfg.SemanticCache.EmbeddingModel = true, "mmbert"
			cfg.Decisions = []Decision{cacheDemandDecision("semantic", true)}
		}, want: []string{"global"}},
		{name: "exact response cache", setup: func(cfg *RouterConfig) {
			cfg.SemanticCache.Enabled, cfg.SemanticCache.EmbeddingModel = true, "mmbert"
			cfg.Decisions = []Decision{cacheDemandDecision("exact", true)}
		}},
		{name: "ML uses another model", setup: func(cfg *RouterConfig) {
			localBinding(cfg)
			cfg.ModelSelection.Enabled = true
			cfg.ModelSelection.ML = MLSelectionConfig{ModelsPath: "selectors", ModelType: "qwen3"}
			cfg.Decisions = []Decision{{Algorithm: &AlgorithmConfig{Type: "knn"}}}
		}},
		{name: "shared memory uses another model", setup: func(cfg *RouterConfig) {
			cfg.Memory.Enabled, cfg.Memory.EmbeddingModel = true, "qwen3"
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{KnowledgeBases: []KnowledgeBaseConfig{{Name: "catalog-kb"}}}
			cfg.EmbeddingConfig.ModelType = "mmbert"
			cfg.ModelDeployments = map[string]ModelDeployment{
				"global": {Provider: ModelRuntimeProvider, Artifact: "test/global", Device: "cpu"},
				"local":  {Provider: ModelRuntimeProvider, Artifact: "test/local", Device: "cpu"},
			}
			cfg.GlobalModelBindings = map[string]ModelBinding{"embedding": {Deployment: "global", Contract: "embedding.v1"}}
			tc.setup(cfg)
			if _, err := CompileModelBindings(cfg); err != nil {
				t.Fatalf("invalid test configuration: %v", err)
			}
			used := ModelRuntimeDeploymentsInUse(cfg)
			got := make([]string, 0, len(used))
			for name := range used {
				got = append(got, name)
			}
			slices.Sort(got)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("started deployments = %v, want %v", got, tc.want)
			}
			if len(cfg.KnowledgeBases) != 1 {
				t.Fatal("demand planning mutated the KB catalog")
			}
		})
	}
}
