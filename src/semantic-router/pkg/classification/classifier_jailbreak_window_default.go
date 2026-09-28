package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Resolve only the implicit local default, on the preparation-owned config
// copy. An explicit deployment, document budget or window retains its policy.
func (m *classifierModelRuntime) resolveDefaultJailbreakWindow() error {
	guard := m.cfg.PromptGuard
	if _, bound := m.plan.Lookup(m.recipe, "prompt_guard"); bound ||
		!guard.Enabled || guard.Variant != config.PromptGuardVariantMmBERT32K || guard.Backend != nil ||
		guard.MaxSequenceLength != 0 || guard.Window != nil ||
		!isDefaultModelArtifact(guard.ModelID, config.DefaultSystemModels().PromptGuard) {
		return nil
	}
	model := config.GetModelByPath(config.DefaultSystemModels().PromptGuard)
	if model == nil || model.MaxContextLength <= 0 {
		return fmt.Errorf("default Guard document budget is absent from the model registry")
	}
	guard.MaxSequenceLength = model.MaxContextLength
	guard.Window = &config.SequenceHeadWindowConfig{Size: 512, Overlap: 255}
	if err := guard.ValidateWindow(); err != nil {
		return err
	}
	m.cfg.PromptGuard = guard
	logging.ComponentEvent("classifier", "jailbreak_default_window_resolved", map[string]interface{}{
		"recipe": m.recipe, "document_max_tokens": guard.MaxSequenceLength,
		"window_size": guard.Window.Size, "window_overlap": guard.Window.Overlap,
		"document_overflow": "reject",
	})
	return nil
}
