package config

import "testing"

// The defaults pin no memory width, so a store without one takes its model's
// default: mmBERT's 256 view, or another model's complete output. A pinned
// width would have to be one every model serves.
func TestMemoryWithoutAWidthTakesItsModelsDefault(t *testing.T) {
	for model, want := range map[string]int{"mmbert": 256, "multimodal": 0, "qwen3": 0} {
		cfg, err := ParseYAMLBytes([]byte("version: v0.3\nglobal:\n  stores:\n    memory:\n      enabled: true\n      embedding_model: " + model + "\n"))
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Memory.Milvus.Dimension != 0 {
			t.Fatalf("%s: defaults pinned a Milvus width: %d", model, cfg.Memory.Milvus.Dimension)
		}
		var memory []EmbeddingRequirement
		for _, r := range EmbeddingRequirements(cfg, model, true) {
			if r.Consumer == "memory" {
				memory = append(memory, r)
			}
		}
		if len(memory) != 1 || memory[0].Model != model || memory[0].Dimension != want {
			t.Fatalf("%s: memory demand %+v, want width %d", model, memory, want)
		}
	}
}

func TestOmniStorageRequirementsLeaveDefaultWidthToPreparedModel(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.EmbeddingModel = "multimodal"
	cfg.SemanticCache.Enabled = true
	cfg.Memory = MemoryConfig{Enabled: true, EmbeddingModel: "multimodal"}
	for _, backend := range []string{"milvus", "valkey", "qdrant"} {
		cfg.Memory.Backend = backend
		cfg.Memory.Valkey = &MemoryValkeyConfig{}
		cfg.Memory.Qdrant = &MemoryQdrantConfig{}
		cfg.Memory.Milvus.Dimension = 0
		seen := map[string]bool{}
		for _, r := range EmbeddingRequirements(cfg, "multimodal", true) {
			if r.Consumer == "response cache" || r.Consumer == "memory" {
				if r.Dimension != 0 || r.Layer != 0 {
					t.Fatalf("%s guessed output width: %+v", backend, r)
				}
				seen[r.Consumer] = true
			}
		}
		if !seen["response cache"] || !seen["memory"] {
			t.Fatalf("missing demand: %+v", seen)
		}
		cfg.Memory.Milvus.Dimension = 128
		cfg.Memory.Valkey.Dimension = 128
		cfg.Memory.Qdrant.Dimension = 128
		for _, r := range EmbeddingRequirements(cfg, "multimodal", true) {
			if r.Consumer == "memory" && r.Dimension != 128 {
				t.Fatalf("explicit width lost before capability validation: %+v", r)
			}
		}
	}
}
