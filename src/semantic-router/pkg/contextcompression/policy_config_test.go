package contextcompression

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPolicyFromConfigAppliesRequestTargetToGlobalBudget(t *testing.T) {
	override := 4096
	policy := PolicyFromConfig(
		&config.ContextCompressionPluginConfig{Enabled: true},
		&override,
	)
	if policy.Budget.TargetAuto || policy.Budget.TargetTokens != override {
		t.Fatalf("global target override = %#v", policy.Budget)
	}
	if policy.Targets.ToolOutputs.TargetTokens != 1000 {
		t.Fatalf(
			"request target changed item target: %#v",
			policy.Targets.ToolOutputs,
		)
	}
}
