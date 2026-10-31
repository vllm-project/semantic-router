package config

// validateVectorStoreContracts refuses a supported format that no text
// extractor reads. The upload route would accept such a file and the
// ingestion would fail on it later, so the configuration names it as a
// mistake. The formats are checked wherever the configuration is loaded or
// replaced, not only when the vector store runtime starts.
func validateVectorStoreContracts(cfg *RouterConfig) error {
	if cfg == nil || cfg.VectorStore == nil || !cfg.VectorStore.Enabled {
		return nil
	}
	return validateVectorStoreSupportedFormats(cfg.VectorStore.SupportedFormats)
}
