package config

import "testing"

func TestEmbeddingRequirementsLeaveMemoryDimensionForModelContract(t *testing.T) {
	cfg := &RouterConfig{
		Memory: MemoryConfig{
			Enabled:        true,
			EmbeddingModel: "mmbert",
			Milvus:         MemoryMilvusConfig{Dimension: 0},
		},
	}

	requirements := EmbeddingRequirements(cfg, "mmbert", true)
	if len(requirements) != 1 || requirements[0].Consumer != "memory" {
		t.Fatalf("memory requirements = %+v, want one memory requirement", requirements)
	}
	if requirements[0].Dimension != 0 {
		t.Fatalf("omitted memory dimension = %d, want 0 for contract resolution", requirements[0].Dimension)
	}

	cfg.Memory.Milvus.Dimension = 256
	requirements = EmbeddingRequirements(cfg, "mmbert", true)
	if len(requirements) != 1 || requirements[0].Dimension != 256 {
		t.Fatalf("configured memory dimension = %+v, want 256", requirements)
	}
}
