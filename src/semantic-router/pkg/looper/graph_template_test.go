package looper

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestTemplateBoundsTheRunWithTheCallBudget(t *testing.T) {
	for _, algorithm := range []string{
		config.DecisionAlgorithmFusion,
		config.DecisionAlgorithmReMoM,
		config.DecisionAlgorithmWorkflows,
	} {
		t.Run(algorithm, func(t *testing.T) {
			program, err := Template(&config.LooperConfig{}, algorithm, nil)
			if err != nil {
				t.Fatalf("Template: %v", err)
			}
			if program.Limits.MaxHops != config.MaxUpstreamCallsPerRequest {
				t.Fatalf("MaxHops = %d, want %d", program.Limits.MaxHops, config.MaxUpstreamCallsPerRequest)
			}
		})
	}
}
