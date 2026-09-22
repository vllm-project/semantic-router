package config

import "testing"

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
