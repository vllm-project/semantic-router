package contextcompression

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPolicyFromConfigAppliesRequestTargetToGlobalBudget(t *testing.T) {
	override := 4096
	policy := PolicyFromConfig(&config.ContextCompressConfig{}, nil, &override)
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

// An explicit max_tokens starts compression and is its target; auto, or no
// budget, resolves both from the model's window.
func TestPolicyFromConfigReadsTheContextBudget(t *testing.T) {
	policy := PolicyFromConfig(&config.ContextCompressConfig{}, &config.ContextBudgetConfig{
		MaxTokens:           &config.CompressionTokenLimit{Value: 6000},
		ReserveOutputTokens: &config.CompressionTokenLimit{Auto: true},
	}, nil)
	if budget := policy.Budget; budget.TriggerAuto || budget.TargetAuto || budget.TriggerTokens != 6000 || budget.TargetTokens != 6000 || !budget.ReserveAuto {
		t.Fatalf("explicit budget = %#v", budget)
	}
	for _, budget := range []*config.ContextBudgetConfig{nil, {MaxTokens: &config.CompressionTokenLimit{Auto: true}}} {
		if got := PolicyFromConfig(&config.ContextCompressConfig{}, budget, nil).Budget; !got.TriggerAuto || !got.TargetAuto || !got.ReserveAuto {
			t.Fatalf("auto budget %#v = %#v", budget, got)
		}
	}
}
