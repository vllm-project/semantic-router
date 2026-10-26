package v1alpha1

import "fmt"

// Validate only declared geometry here. Model capacity and the tokenizer's
// special-token overhead are checked by the model runtime.
func (r *SemanticRouter) validatePIIWindow() error {
	if r.Spec.Config.Classifier == nil || r.Spec.Config.Classifier.PIIModel == nil {
		return nil
	}
	cfg := r.Spec.Config.Classifier.PIIModel
	if cfg.MaxSequenceLength < 0 {
		return fmt.Errorf("config.classifier.pii_model.max_sequence_length must be nonnegative")
	}
	if (cfg.MaxSequenceLength > 0 || cfg.Window != nil) && cfg.Backend != nil {
		return fmt.Errorf("config.classifier.pii_model.max_sequence_length and window apply to the local model; omit them with backend")
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
