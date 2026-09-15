package config

import "testing"

func TestDormantDefaultKeepsSharedConsumersWithoutRoutingEmbeddings(t *testing.T) {
	cfg := &RouterConfig{RouterOptions: RouterOptions{AutoModelNames: []string{}}}
	cfg.EmbeddingRules = []EmbeddingRule{{Name: "unused"}}
	if needed := EmbeddingModelsNeeded(cfg, "mmbert", true); len(needed) != 0 {
		t.Fatalf("unreachable default owns routing embeddings: %v", needed)
	}
	if requirements := EmbeddingRequirements(cfg, "mmbert", true); len(requirements) != 0 {
		t.Fatalf("unreachable default needs routing capabilities: %+v", requirements)
	}
	cfg.Tools.Enabled = true
	if needed := EmbeddingModelsNeeded(cfg, "mmbert", true); !needed["mmbert"] {
		t.Fatalf("default shared tool consumer lost embeddings: %v", needed)
	}
	if len(cfg.EmbeddingRules) != 1 {
		t.Fatal("consumer planning changed canonical signals")
	}
}
