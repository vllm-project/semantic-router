package config

import "fmt"

// A remote embedding backend must not silently ignore a requested long
// context. The served model's card checks capacity.
func validateClassifierContextLimits(cfg *RouterConfig) error {
	if cfg.EmbeddingConfig.FullContext && (cfg.UsesRemoteEmbeddings() || cfg.EmbeddingConfig.ModelType != "mmbert") {
		return fmt.Errorf("embedding_config.full_context currently requires the native mmbert model type")
	}
	return nil
}
