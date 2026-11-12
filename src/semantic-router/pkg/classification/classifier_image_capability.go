package classification

import (
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// ValidateImageClassification checks only prepared, routed image rules and
// their observed local model-runtime capabilities. It performs no inference.
func (c *Classifier) ValidateImageClassification() error {
	if c == nil || c.Config == nil {
		return fmt.Errorf("%w: image classification requires a prepared classifier", binding.ErrCapability)
	}
	found := false
	for _, asked := range c.preparedSystemOneRules() {
		rule := asked.rule
		if rule.EffectiveStage() != config.SystemOneStageRequest || !rule.Input.Image {
			continue
		}
		found = true
		name, deployment := c.systemOneRuleDeployment(rule)
		if !deployment.IsModelRuntime() {
			return fmt.Errorf("%w: image classification requires a local model-runtime deployment", binding.ErrCapability)
		}
		card, ready := c.decisionTaskCard(name)
		if !ready || !slices.Contains(card.ReadModalities(), "image") {
			return fmt.Errorf("%w: image classification requires a prepared image-capable deployment", binding.ErrCapability)
		}
	}
	if !found {
		return fmt.Errorf("%w: image classification requires a routed image-reading rule", binding.ErrCapability)
	}
	return nil
}
