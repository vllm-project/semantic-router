package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestNewWorkflowStateServiceIfEnabled(t *testing.T) {
	t.Run("nil config", func(t *testing.T) {
		if got := newWorkflowStateServiceIfEnabled(nil); got != nil {
			t.Fatal("expected nil service for nil config")
		}
	})

	t.Run("no workflows", func(t *testing.T) {
		cfg := &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{{Name: "static"}},
			},
		}
		if got := newWorkflowStateServiceIfEnabled(cfg); got != nil {
			t.Fatal("expected nil service when no workflows decision exists")
		}
	})

	t.Run("flat workflows decision", func(t *testing.T) {
		cfg := &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{workflowTestDecision()},
			},
		}
		svc := newWorkflowStateServiceIfEnabled(cfg)
		if svc == nil {
			t.Fatal("expected a generation-owned workflow state service")
		}
		t.Cleanup(func() { _ = svc.Close() })
	})

	t.Run("recipe-only workflows decision", func(t *testing.T) {
		cfg := &config.RouterConfig{
			Recipes: []config.RoutingRecipe{{
				Name: "agent",
				Profile: config.RoutingProfile{
					Decisions: []config.Decision{workflowTestDecision()},
				},
			}},
		}
		svc := newWorkflowStateServiceIfEnabled(cfg)
		if svc == nil {
			t.Fatal("expected a generation-owned service when workflows live only on a recipe")
		}
		t.Cleanup(func() { _ = svc.Close() })
	})
}

func workflowTestDecision() config.Decision {
	includeTrace := true
	return config.Decision{
		Name: "workflow_decision",
		ModelRefs: []config.ModelRef{
			{Model: "worker-model"},
			{Model: "verifier-model"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmWorkflows,
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeStatic,
				Roles: []config.WorkflowRoleConfig{
					{Name: "worker", Models: []string{"worker-model"}, Prompt: "Use tools when needed, then solve."},
				},
				Final:                        config.WorkflowFinalConfig{Model: "verifier-model"},
				MaxSteps:                     2,
				MaxParallel:                  1,
				IncludeIntermediateResponses: &includeTrace,
			},
		},
	}
}
