package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Resolve only the implicit local Vela 1.0 PII module. Named deployments and
// every explicit budget/backend/window retain their operator-authored policy;
// Vela 2.0 reads the whole text itself.
func (m *classifierModelRuntime) resolveDefaultPIIWindow() error {
	pii := m.cfg.PIIModel
	vela1PII := config.Vela1SystemModels().PIIClassifier
	if _, bound := m.plan.Lookup(m.recipe, "pii_classifier"); bound || !pii.Active() || pii.Backend != nil || pii.MaxSequenceLength != 0 || pii.Window != nil ||
		!isDefaultModelArtifact(pii.ModelID, vela1PII) {
		return nil
	}
	model := config.GetModelByPath(vela1PII)
	if model == nil || model.MaxContextLength < 512 {
		return fmt.Errorf("default PII document capacity is unavailable")
	}
	pii.MaxSequenceLength = model.MaxContextLength
	pii.Window = &config.SequenceHeadWindowConfig{Size: 512, Overlap: 255}
	m.cfg.PIIModel = pii
	return nil
}
