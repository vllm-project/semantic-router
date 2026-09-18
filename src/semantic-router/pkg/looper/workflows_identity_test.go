package looper

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestValidateWorkflowPlanRejectsAmbiguousAccessIdentities(t *testing.T) {
	for _, tc := range []struct {
		name string
		ids  []string
	}{
		{"duplicate steps", []string{"research", "research"}},
		{"trimmed steps", []string{"research", " research "}},
		{"default after explicit", []string{"step-2", ""}},
		{"explicit after default", []string{"", "step-1"}},
		{"step after agent", []string{"research", "research:0:worker"}},
		{"agent after step", []string{"research:0:worker", "research"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := &workflowPlan{}
			for _, id := range tc.ids {
				plan.Steps = append(plan.Steps, workflowPlanStep{ID: id, Models: []string{"worker"}})
			}
			err := validateWorkflowPlan(plan, []string{"worker"}, workflowsExecutionConfig{MaxSteps: 3, MaxParallel: 1})
			if err == nil || !strings.Contains(err.Error(), "access identity") {
				t.Fatalf("expected access identity collision error, got %v", err)
			}
		})
	}
}

func TestValidateWorkflowPlanRejectsNormalizedStaticRoleCollision(t *testing.T) {
	cfg := workflowsExecutionConfig{
		MaxSteps: 3, MaxParallel: 1,
		Roles: []config.WorkflowRoleConfig{
			{Name: "Code Review", Models: []string{"worker"}},
			{Name: "code_review", Models: []string{"worker"}},
		},
	}
	plan, err := buildStaticWorkflowPlan(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateWorkflowPlan(plan, []string{"worker"}, cfg); err == nil {
		t.Fatal("expected normalized static role ID collision to be rejected")
	}
}

func TestWorkflowValidatedAccessListsPreserveVisibility(t *testing.T) {
	for _, tc := range []struct {
		name   string
		access []string
		want   int
	}{
		{"omitted", nil, 2},
		{"empty", []string{}, 0},
		{"step", []string{"solve"}, 2},
		{"agent", []string{"solve:1:worker-b"}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := &workflowPlan{Steps: []workflowPlanStep{
				{ID: " solve ", Models: []string{"worker-a", "worker-b"}},
				{ID: "review", Models: []string{"worker-a"}, AccessList: tc.access},
			}}
			err := validateWorkflowPlan(plan, []string{"worker-a", "worker-b"}, workflowsExecutionConfig{MaxSteps: 3, MaxParallel: 2})
			if err != nil {
				t.Fatalf("valid plan rejected: %v", err)
			}
			prior := []workflowStepResult{{
				step: plan.Steps[0],
				responses: []*ModelResponse{
					{Model: "worker-a", Content: "A"},
					{Model: "worker-b", Content: "B"},
				},
			}}
			visible := workflowVisibleStepResults(plan.Steps[1], prior)
			count := 0
			for _, result := range visible {
				count += len(result.responses)
			}
			if count != tc.want {
				t.Fatalf("visible responses = %d, want %d", count, tc.want)
			}
			if tc.name == "agent" && visible[0].responses[0].Model != "worker-b" {
				t.Fatal("agent access list selected the wrong worker")
			}
		})
	}
}
