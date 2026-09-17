package config

import "fmt"

// Local classifier budgets are explicit; incompatible backends must not silently
// ignore a requested long context. Actual model capacity is checked on loading.
func validateClassifierContextLimits(cfg *RouterConfig) error {
	if cfg.EmbeddingConfig.FullContext && (cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() || cfg.EmbeddingConfig.ModelType != "mmbert") {
		return fmt.Errorf("embedding_config.full_context currently requires the native mmbert model type")
	}
	variant, _ := cfg.CategoryModel.EffectiveVariant()
	checks := []struct {
		name      string
		limit     int
		supported bool
	}{
		{"classifier.domain", cfg.CategoryModel.MaxSequenceLength, cfg.CategoryModel.Backend == nil && variant == CategoryVariantMmBERT32K},
		{"classifier.pii", cfg.PIIModel.MaxSequenceLength, cfg.PIIModel.Backend == nil && cfg.PIIModel.UseMmBERT32K},
		{"prompt_guard", cfg.PromptGuard.MaxSequenceLength, cfg.PromptGuard.Protocol == "" && cfg.PromptGuard.Variant == PromptGuardVariantMmBERT32K},
		{"feedback_detector", cfg.FeedbackDetector.MaxSequenceLength, cfg.FeedbackDetector.UseMmBERT32K},
		{"hallucination_mitigation.fact_check", cfg.HallucinationMitigation.FactCheckModel.MaxSequenceLength, cfg.HallucinationMitigation.FactCheckModel.UseMmBERT32K},
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
			return fmt.Errorf("global.model_catalog.modules.%s.max_sequence_length requires the local mmbert32k backend", check.name)
		}
	}
	return nil
}
