package config

import (
	"testing"
)

func TestEmbeddingModelsNeededSelectorsAndExecutorsLoadNoEmbedding(t *testing.T) {
	for _, entry := range DecisionAlgorithmCatalog() {
		t.Run(entry.Type, func(t *testing.T) {
			cfg := &RouterConfig{Entrypoints: []EntrypointMapping{{ModelNames: []string{"auto"}, Recipe: DefaultRecipeName}}}
			cfg.ModelSelection.Enabled = true
			cfg.Decisions = []Decision{{Algorithm: &AlgorithmConfig{Type: entry.Type}}}
			if got := EmbeddingModelsNeeded(cfg, "mmbert", false); len(got) != 0 {
				t.Fatalf("algorithm %s loads an embedding: %v", entry.Type, got)
			}
		})
	}
}
