package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Resolve only the local Vela 1.0 PII model a deprecated module path names.
// Named deployments and every explicit budget or window retain their
// operator-authored policy; Vela 2.0 reads the whole text itself.
func (m *classifierModelRuntime) resolveDefaultPIIWindow() error {
	pii := m.cfg.PIIModel
	vela1PII := config.Vela1SystemModels().PIIClassifier
	if !pii.Active() || pii.MaxSequenceLength != 0 || pii.Window != nil ||
		!m.translatedLocalModel("pii_classifier", vela1PII) {
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
