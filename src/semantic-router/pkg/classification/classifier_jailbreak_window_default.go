package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Resolve only the local Vela 1.0 Guard a deprecated module path names, on
// the preparation-owned config copy. A deployment the configuration names,
// a document budget or a window retains its policy; Vela 2.0 reads the whole
// text itself.
func (m *classifierModelRuntime) resolveDefaultJailbreakWindow() error {
	guard := m.cfg.PromptGuard
	vela1Guard := config.Vela1SystemModels().PromptGuard
	if !guard.Enabled || guard.MaxSequenceLength != 0 || guard.Window != nil ||
		!m.translatedLocalModel("prompt_guard", vela1Guard) {
		return nil
	}
	model := config.GetModelByPath(vela1Guard)
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
