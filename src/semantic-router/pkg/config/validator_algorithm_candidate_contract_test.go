package config

import (
	"strings"
	"testing"
)

func TestValidateDecisionMinimumCandidatesAllowsModelFreeRecipeAsset(t *testing.T) {
	err := validateDecisionAlgorithmConfig("model-free", nil, &AlgorithmConfig{
		Type:              DecisionAlgorithmStatic,
		MinimumCandidates: 3,
	})
	if err != nil {
		t.Fatalf("model-free Recipe contract was rejected: %v", err)
	}
}

func TestValidateDecisionMinimumCandidatesRejectsUndersizedMaterialization(t *testing.T) {
	err := validateDecisionAlgorithmConfig(
		"panel",
		[]ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		&AlgorithmConfig{Type: DecisionAlgorithmStatic, MinimumCandidates: 3},
	)
	if err == nil || !strings.Contains(err.Error(), "requires at least 3 unique modelRefs, got 2") {
		t.Fatalf("expected minimum candidate error, got %v", err)
	}
}

func TestValidateDecisionFusionRejectsImpossibleQuorum(t *testing.T) {
	err := validateDecisionAlgorithmConfig(
		"fusion-panel",
		[]ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		&AlgorithmConfig{
			Type: DecisionAlgorithmFusion,
			Fusion: &FusionAlgorithmConfig{
				MinSuccessfulResponses: 3,
			},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "exceeds the configured panel size 2") {
		t.Fatalf("expected fusion quorum error, got %v", err)
	}
}

func TestValidateDecisionDynamicWorkflowRejectsImpossibleQuorum(t *testing.T) {
	err := validateDecisionAlgorithmConfig(
		"workflow",
		[]ModelRef{{Model: "model-a"}},
		&AlgorithmConfig{
			Type: DecisionAlgorithmWorkflows,
			Workflows: &WorkflowsAlgorithmConfig{
				Mode:                   WorkflowModeDynamic,
				Planner:                WorkflowPlannerConfig{Model: "model-a"},
				MinSuccessfulResponses: 2,
			},
		},
	)
	if err == nil || !strings.Contains(err.Error(), "exceeds the configured worker pool size 1") {
		t.Fatalf("expected workflow quorum error, got %v", err)
	}
}

func TestValidateStaticWorkflowRejectsRoleBelowQuorum(t *testing.T) {
	err := ValidateWorkflowsAlgorithmConfig(&WorkflowsAlgorithmConfig{
		Mode:                   WorkflowModeStatic,
		MinSuccessfulResponses: 2,
		Roles: []WorkflowRoleConfig{{
			Name:   "worker",
			Models: []string{"model-a"},
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "exceeds roles[0] model count 1") {
		t.Fatalf("expected static workflow quorum error, got %v", err)
	}
}

func TestValidateWorkflowRejectsQuorumAboveEffectiveMaxParallel(t *testing.T) {
	err := ValidateWorkflowsAlgorithmConfig(&WorkflowsAlgorithmConfig{
		Mode:                   WorkflowModeDynamic,
		Planner:                WorkflowPlannerConfig{Model: "planner"},
		MinSuccessfulResponses: DefaultWorkflowMaxParallel + 1,
	})
	if err == nil || !strings.Contains(err.Error(), "exceeds max_parallel=2") {
		t.Fatalf("expected max parallel quorum error, got %v", err)
	}
}
