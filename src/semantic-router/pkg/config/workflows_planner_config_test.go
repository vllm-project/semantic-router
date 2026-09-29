package config

import (
	"strings"
	"testing"
)

func TestDynamicPlannerOmissionPreservesWorkerMinimum(t *testing.T) {
	for _, planner := range []WorkflowPlannerConfig{{}, {Model: "coordinator"}} {
		algorithm := &AlgorithmConfig{
			Type:              DecisionAlgorithmWorkflows,
			MinimumCandidates: 2,
			Workflows:         &WorkflowsAlgorithmConfig{Mode: WorkflowModeDynamic, Planner: planner, MinSuccessfulResponses: 2, MaxParallel: 2},
		}
		if err := validateDecisionAlgorithmConfig("flow", []ModelRef{{Model: "worker-a"}, {Model: "worker-b"}}, algorithm); err != nil {
			t.Fatal(err)
		}
		if algorithm.Workflows.Planner.Model != planner.Model {
			t.Fatal("validation invented or replaced a planner")
		}
		err := validateDecisionAlgorithmConfig("flow", []ModelRef{{Model: "worker-a"}, {Model: "worker-a"}}, algorithm)
		if err == nil || !strings.Contains(err.Error(), "unique modelRefs") {
			t.Fatalf("planner default bypassed distinct worker minimum: %v", err)
		}
	}
	if err := ValidateWorkflowsAlgorithmConfig(&WorkflowsAlgorithmConfig{Mode: WorkflowModeDynamic, Planner: WorkflowPlannerConfig{MaxCompletionTokens: -1}}); err == nil {
		t.Fatal("invalid planner output budget accepted")
	}
}
