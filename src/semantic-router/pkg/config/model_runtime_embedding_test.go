package config

import "testing"

func TestModelRuntimePlansImplicitEmbeddingConsumers(t *testing.T) {
	for _, consumer := range []string{"recipe", "tools", "memory", "cache", "vector_store", "api"} {
		t.Run(consumer, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.EmbeddingConfig.ModelType = "mmbert"
			cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
			cfg.EmbeddingModels.UseCPU = true
			switch consumer {
			case "recipe":
				cfg.EmbeddingRules = []EmbeddingRule{{Name: "meaning", Candidates: []string{"hello"}}}
			case "tools":
				cfg.Tools.Enabled = true
			case "memory":
				cfg.Memory.Enabled = true
			case "cache":
				cfg.SemanticCache.Enabled = true
			case "vector_store":
				cfg.VectorStore = &VectorStoreConfig{Enabled: true}
			case "api":
				cfg.API.Embeddings.Enabled = true
			}
			used := ModelRuntimeDeploymentsInUse(cfg)
			deployment, ok := used["@embedding.mmbert"]
			if !ok || len(used) != 1 {
				t.Fatalf("%s embedding missing from startup plan: %v", consumer, used)
			}
			if deployment.Artifact != "vllm-sr/Vela-1.0-Encoder-307M-Embedding" || deployment.Device != "cpu" || deployment.Input.Overflow != "truncate" {
				t.Fatalf("planned embedding differs from runtime preparation: %+v", deployment)
			}
		})
	}
}

func TestModelRuntimeDoesNotPlanUnusedImplicitEmbedding(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 0 {
		t.Fatalf("unused embedding module started a runtime: %v", used)
	}
}

func TestModelRuntimePlansEmbeddingWithClassifier(t *testing.T) {
	cfg := taskBindingConfig()
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	cfg.Tools.Enabled = true
	used := ModelRuntimeDeploymentsInUse(cfg)
	if len(used) != 2 || used["vela-domain"].Artifact == "" || used["@embedding.mmbert"].Artifact == "" {
		t.Fatalf("classifier and embedding must be planned together: %v", used)
	}
}

func TestModelRuntimeEmbeddingPlanRespectsExplicitAndRemoteBindings(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	cfg.Tools.Enabled = true
	cfg.EmbeddingConfig.Backend = EmbeddingBackendOpenAICompatible
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 0 {
		t.Fatalf("remote embedding backend must not start local defaults: %v", used)
	}
	cfg.ModelDeployments = map[string]ModelDeployment{"chosen": {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Embedding", Device: "cpu"}}
	cfg.GlobalModelBindings = map[string]ModelBinding{"embedding": {Deployment: "chosen", Contract: "embedding.v1"}}
	used := ModelRuntimeDeploymentsInUse(cfg)
	if len(used) != 1 || used["chosen"].Artifact == "" {
		t.Fatalf("explicit binding must replace the implicit primary even on a remote backend: %v", used)
	}
}

func TestModelRuntimeEmbeddingPlanKeepsSecondaryGlobalConsumer(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	cfg.Qwen3ModelPath = t.TempDir()
	cfg.Tools.Enabled = true
	cfg.Memory.Enabled = true
	cfg.Memory.EmbeddingModel = "qwen3"
	cfg.ModelDeployments = map[string]ModelDeployment{"chosen": {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Embedding", Device: "cpu"}}
	cfg.GlobalModelBindings = map[string]ModelBinding{"embedding": {Deployment: "chosen", Contract: "embedding.v1"}}
	used := ModelRuntimeDeploymentsInUse(cfg)
	if len(used) != 2 || used["chosen"].Artifact == "" || used["@embedding.qwen3"].Artifact != cfg.Qwen3ModelPath {
		t.Fatalf("explicit primary must retain the memory consumer's secondary model: %v", used)
	}
}

func TestModelRuntimeEmbeddingPlanExcludesDormantRecipe(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.EmbeddingConfig.ModelType = "mmbert"
	cfg.MmBertModelPath = "models/Vela-1.0-Encoder-307M-Embedding"
	cfg.EmbeddingRules = []EmbeddingRule{{Name: "dormant", Candidates: []string{"hello"}}}
	moveTestRoutingToUnmappedRecipe(cfg)
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 0 {
		t.Fatalf("dormant recipe must not start its embedding: %v", used)
	}
	cfg.Tools.Enabled = true
	if used := ModelRuntimeDeploymentsInUse(cfg); len(used) != 1 || used["@embedding.mmbert"].Artifact == "" {
		t.Fatalf("global tools must still prepare their embedding: %v", used)
	}
}
