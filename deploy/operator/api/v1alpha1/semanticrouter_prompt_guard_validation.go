package v1alpha1

import "fmt"

// Admission mirrors the explicit local-token contract. Model capacity and
// tokenizer-specific special-token overhead remain runtime checks.
func (r *SemanticRouter) validatePromptGuardContext() error {
	cfg := r.Spec.Config.PromptGuard
	if cfg == nil {
		return nil
	}
	if cfg.MaxSequenceLength < 0 {
		return fmt.Errorf("config.prompt_guard.max_sequence_length must be nonnegative")
	}
	local := cfg.Protocol == "" && (cfg.Variant == "" || cfg.Variant == "mmbert32k")
	if (cfg.MaxSequenceLength > 0 || cfg.Window != nil) && !local {
		return fmt.Errorf("config.prompt_guard.max_sequence_length and window require the local mmbert32k variant")
	}
	if cfg.Window == nil {
		return nil
	}
	limit := cfg.MaxSequenceLength
	if limit == 0 {
		limit = 512
	}
	if cfg.Window.Size <= 0 || cfg.Window.Size > limit {
		return fmt.Errorf("config.prompt_guard.window.size must be positive and at most max_sequence_length (%d)", limit)
	}
	if cfg.Window.Overlap < 0 || cfg.Window.Overlap >= cfg.Window.Size {
		return fmt.Errorf("config.prompt_guard.window.overlap must be nonnegative and smaller than window.size")
	}
	seen := make(map[string]bool)
	for _, label := range cfg.PositiveLabels {
		if label == "" || seen[label] {
			return fmt.Errorf("config.prompt_guard.positive_labels must be nonempty and unique for windowed inference")
		}
		seen[label] = true
	}
	return nil
}
