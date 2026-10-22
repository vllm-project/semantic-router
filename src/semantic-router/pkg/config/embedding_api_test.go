package config

import "testing"

func TestEmbeddingAPIAddsOnlyGlobalDemand(t *testing.T) {
	cfg := &RouterConfig{}
	if needed := EmbeddingModelsNeeded(cfg.ConfigForGlobalModelServices(), "mmbert", true); len(needed) != 0 {
		t.Fatalf("disabled API loaded models: %v", needed)
	}
	cfg.API.Embeddings.Enabled = true
	if needed := EmbeddingModelsNeeded(cfg.ConfigForGlobalModelServices(), "mmbert", true); len(needed) != 1 || !needed["mmbert"] {
		t.Fatalf("enabled API did not declare global demand: %v", needed)
	}
	if needed := EmbeddingModelsNeeded(cfg, "mmbert", false); len(needed) != 0 {
		t.Fatalf("global API caused recipe loading: %v", needed)
	}
	requirements := EmbeddingRequirements(cfg.ConfigForGlobalModelServices(), "mmbert", true)
	if len(requirements) != 1 || !requirements[0].SharedService || requirements[0].Consumer != "embedding API" {
		t.Fatalf("missing API preparation requirements: %+v", requirements)
	}
}
