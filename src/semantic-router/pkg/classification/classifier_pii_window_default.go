package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Resolve only the implicit local module. Named deployments and every explicit
// budget/backend/window retain their operator-authored policy.
func (m *classifierModelRuntime) resolveDefaultPIIWindow() error {
	pii := m.cfg.PIIModel
	if _, bound := m.plan.Lookup(m.recipe, "pii_classifier"); bound || !pii.Active() || pii.Backend != nil || !pii.UseMmBERT32K || pii.MaxSequenceLength != 0 || pii.Window != nil {
		return nil
	}
	model := config.GetModelByPath(config.DefaultSystemModels().PIIClassifier)
	if model == nil || model.MaxContextLength < 512 {
		return fmt.Errorf("default PII document capacity is unavailable")
	}
	pii.MaxSequenceLength = model.MaxContextLength
	pii.Window = &config.SequenceHeadWindowConfig{Size: 512, Overlap: 255}
	m.cfg.PIIModel = pii
	return nil
}
