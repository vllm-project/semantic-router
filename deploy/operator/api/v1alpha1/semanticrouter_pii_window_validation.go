package v1alpha1

import "fmt"

// Validate only declared geometry here. Model capacity and the tokenizer's
// special-token overhead are checked by the native loader.
func (r *SemanticRouter) validatePIIWindow() error {
	if r.Spec.Config.Classifier == nil || r.Spec.Config.Classifier.PIIModel == nil {
		return nil
	}
	cfg := r.Spec.Config.Classifier.PIIModel
	if cfg.Backend != nil && cfg.UseMmBERT32K {
		return fmt.Errorf("config.classifier.pii_model.backend cannot be combined with use_mmbert_32k")
	}
	if cfg.MaxSequenceLength < 0 {
		return fmt.Errorf("config.classifier.pii_model.max_sequence_length must be nonnegative")
	}
	if (cfg.MaxSequenceLength > 0 || cfg.Window != nil) && (cfg.Backend != nil || !cfg.UseMmBERT32K) {
		return fmt.Errorf("config.classifier.pii_model.max_sequence_length and window require local use_mmbert_32k")
	}
	if cfg.Window == nil {
		return nil
	}
	limit := cfg.MaxSequenceLength
	if limit == 0 {
		limit = 512
	}
	if cfg.Window.Size <= 0 || cfg.Window.Size > limit {
		return fmt.Errorf("config.classifier.pii_model.window.size must be positive and at most max_sequence_length (%d)", limit)
	}
	if cfg.Window.Overlap < 0 || cfg.Window.Overlap >= cfg.Window.Size {
		return fmt.Errorf("config.classifier.pii_model.window.overlap must be nonnegative and smaller than window.size")
	}
	return nil
}
