package config

import "fmt"

// Local classifier budgets are explicit; a remote backend must not silently
// ignore a requested long context. The served model's card checks capacity.
func validateClassifierContextLimits(cfg *RouterConfig) error {
	if cfg.EmbeddingConfig.FullContext && (cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() || cfg.EmbeddingConfig.ModelType != "mmbert") {
		return fmt.Errorf("embedding_config.full_context currently requires the native mmbert model type")
	}
	checks := []struct {
		name      string
		limit     int
		supported bool
	}{
		{"classifier.domain", cfg.CategoryModel.MaxSequenceLength, cfg.CategoryModel.Backend == nil},
		{"classifier.pii", cfg.PIIModel.MaxSequenceLength, cfg.PIIModel.Backend == nil},
		{"prompt_guard", cfg.PromptGuard.MaxSequenceLength, cfg.PromptGuard.Backend == nil},
		{"feedback_detector", cfg.FeedbackDetector.MaxSequenceLength, true},
		{"hallucination_mitigation.fact_check", cfg.HallucinationMitigation.FactCheckModel.MaxSequenceLength, true},
	}
	if cfg.ModalityDetector.Classifier != nil {
		checks = append(checks, struct {
			name      string
			limit     int
			supported bool
		}{"modality_detector.classifier", cfg.ModalityDetector.Classifier.MaxSequenceLength, true})
	}
	for _, check := range checks {
		if check.limit < 0 {
			return fmt.Errorf("global.model_catalog.modules.%s.max_sequence_length must be nonnegative", check.name)
		}
		if check.limit > 0 && !check.supported {
			return fmt.Errorf("global.model_catalog.modules.%s.max_sequence_length requires the local model", check.name)
		}
	}
	return nil
}
